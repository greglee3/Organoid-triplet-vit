from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRAIN_SCRIPT = PROJECT_ROOT / "05_final_tuning/02_ordinal_aware_triplet/train_pilot.py"
PYTHON = PROJECT_ROOT / ".venv/bin/python"
STAGE08_ROOT = (
    PROJECT_ROOT
    / "runs/05_final_tuning/PROFESSOR_FEEDBACK/04_triplet_final_tuning/08_lambda_retuning"
)
SCREEN_ROOT = STAGE08_ROOT / "screening_seed_4242"
SELECTION_PATH = STAGE08_ROOT / "summary/LAMBDA_SCREENING_SELECTION.json"
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs/05_final_tuning/PROFESSOR_FEEDBACK/04_triplet_final_tuning/09_lambda_followup"
)
STATUS_PATH = OUTPUT_ROOT / "FOLLOWUP_STATUS.json"
LOCK_PATH = OUTPUT_ROOT / "followup.lock"
BASELINE_MULTISEED_ROOT = (
    PROJECT_ROOT
    / "runs/05_final_tuning/PROFESSOR_FEEDBACK/04_triplet_final_tuning/02_lr_wd_multiseed_confirmation/runs"
)
SEEDS = (27182, 16180, 57721)
SCREEN_LAMBDAS = (0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_status(status: str, **extra: object) -> None:
    payload = {
        "status": status,
        "updated_at_unix": time.time(),
        "test_status": "NOT_ACCESSED",
        **extra,
    }
    write_json(STATUS_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def lambda_key(value: float) -> str:
    return "lambda_" + format(value, ".10g").replace(".", "p")


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def acquire_lock() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = read_json(LOCK_PATH)
        pid = int(existing.get("pid", -1))
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            LOCK_PATH.unlink()
            return acquire_lock()
        raise RuntimeError(f"Follow-up watcher already active: PID={pid}")
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "created_at_unix": time.time()}, stream)


def wait_for_screening() -> dict:
    screening_pid_path = STAGE08_ROOT / "training.pid"
    update_status("waiting_for_lambda_screening")
    while not SELECTION_PATH.is_file():
        if screening_pid_path.is_file():
            pid = int(screening_pid_path.read_text(encoding="utf-8").strip())
            try:
                os.kill(pid, 0)
            except ProcessLookupError as error:
                raise RuntimeError(
                    "Lambda screening stopped without producing its selection file"
                ) from error
        time.sleep(60)
    return read_json(SELECTION_PATH)


def gpu_processes() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def wait_for_idle_gpu(action: str) -> None:
    while True:
        processes = gpu_processes()
        if not processes:
            time.sleep(15)
            if not gpu_processes():
                return
        update_status("waiting_for_idle_gpu", action=action, gpu_processes=processes)
        time.sleep(60)


def training_command(
    output: Path,
    triplet_lambda: float,
    seed: int,
    reference_baseline: Path,
) -> list[str]:
    return [
        str(PYTHON),
        "-u",
        str(TRAIN_SCRIPT),
        "--data",
        str(PROJECT_ROOT / "data/Asano_Full_Data.zip"),
        "--output",
        str(output),
        "--ordinal-lambda",
        "0",
        "--lambdas",
        format(triplet_lambda, ".10g"),
        "--mining-mode",
        "semi-hard",
        "--adjacent-margin",
        "0.05",
        "--far-margin",
        "0.05",
        "--embedding-head",
        "shared",
        "--projection-dim",
        "128",
        "--epochs",
        "20",
        "--folds",
        "3",
        "--classes-per-batch",
        "3",
        "--samples-per-class",
        "8",
        "--workers",
        "8",
        "--lr",
        "3e-5",
        "--head-lr-multiplier",
        "10",
        "--optimizer",
        "adamw",
        "--weight-decay",
        "0.01",
        "--warmup-epochs",
        "2",
        "--scheduler",
        "cosine",
        "--scheduler-milestones",
        "10,15",
        "--scheduler-gamma",
        "0.1",
        "--triplet-ce-only-epochs",
        "0",
        "--triplet-ramp-epochs",
        "0",
        "--augmentation",
        "paper",
        "--seed",
        str(seed),
        "--min-c-specificity",
        "0.90",
        "--max-c-recall-drop",
        "0.02",
        "--max-c-precision-drop",
        "0.03",
        "--reference-baseline-metrics",
        str(reference_baseline),
        "--device",
        "cuda",
    ]


def flatten(metrics: dict) -> dict:
    embedding = metrics["embedding_metrics"]
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "c_recall": float(metrics["c_recall"]),
        "c_precision": float(metrics["c_precision"]),
        "c_specificity": float(metrics["c_specificity"]),
        "c_to_a": int(metrics["c_to_a"]),
        "c_to_b": int(metrics["c_to_b"]),
        "silhouette": float(embedding["silhouette_cosine"]),
        "gap": float(embedding["between_minus_within_distance"]),
    }


def eligible(candidate: dict, baseline: dict) -> bool:
    return (
        candidate["c_specificity"] >= 0.90
        and candidate["c_recall"] >= baseline["c_recall"] - 0.02
        and candidate["c_precision"] >= baseline["c_precision"] - 0.03
    )


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def summarize_boundary() -> float:
    baseline = flatten(read_json(SCREEN_ROOT / "lambda_0/oof_metrics.json"))
    rows = []
    for value in SCREEN_LAMBDAS:
        item = flatten(read_json(SCREEN_ROOT / lambda_key(value) / "oof_metrics.json"))
        rows.append(
            {
                "lambda": value,
                **item,
                "delta_f1_vs_ce": item["macro_f1"] - baseline["macro_f1"],
                "eligible": value > 0 and eligible(item, baseline),
            }
        )
    candidates = [row for row in rows if row["eligible"]]
    if not candidates:
        raise RuntimeError("No positive lambda passed the boundary safety gates")
    best_f1 = max(row["macro_f1"] for row in candidates)
    band = [row for row in candidates if row["macro_f1"] >= best_f1 - 0.002]
    selected = min(band, key=lambda row: row["lambda"])

    summary_root = OUTPUT_ROOT / "boundary_summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    with (summary_root / "LAMBDA_BOUNDARY_RESULTS.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        summary_root / "LAMBDA_BOUNDARY_SELECTION.json",
        {
            "rows": rows,
            "selected_lambda": selected["lambda"],
            "selection_rule": "safety gates; highest F1; within 0.2 pp choose smaller lambda",
            "test_status": "NOT_ACCESSED",
        },
    )
    report = [
        "# Lambda 0.02 경계 확인 결과",
        "",
        "| Lambda | Accuracy | Macro-F1 | Delta F1 | C recall | C precision | C specificity | C->A | C->B | 적격 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['lambda']:g} | {pct(row['accuracy'])} | {pct(row['macro_f1'])} | "
            f"{100 * row['delta_f1_vs_ce']:+.2f} pp | {pct(row['c_recall'])} | "
            f"{pct(row['c_precision'])} | {pct(row['c_specificity'])} | "
            f"{row['c_to_a']} | {row['c_to_b']} | {'✓' if row['eligible'] else '—'} |"
        )
    report.extend(["", f"선택 lambda: **{selected['lambda']:g}**"])
    (summary_root / "LAMBDA_BOUNDARY_RESULTS_KO.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return float(selected["lambda"])


def run_boundary() -> None:
    value = 0.02
    metric_path = SCREEN_ROOT / lambda_key(value) / "oof_metrics.json"
    if not metric_path.is_file():
        wait_for_idle_gpu("lambda_0.02_boundary")
        update_status("running_boundary", selected_lambda=value, seed=4242)
        subprocess.run(
            training_command(
                SCREEN_ROOT,
                value,
                4242,
                SCREEN_ROOT / "lambda_0/oof_metrics.json",
            ),
            check=True,
        )
    selected = summarize_boundary()
    update_status(
        "boundary_complete",
        tested_lambda=value,
        selected_lambda_after_boundary=selected,
    )


def mean_sd(values: list[float]) -> tuple[float, float]:
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def run_multiseed(value: float) -> None:
    key = lambda_key(value)
    run_root = OUTPUT_ROOT / "multiseed" / key
    rows = []
    for seed in SEEDS:
        baseline_path = (
            BASELINE_MULTISEED_ROOT
            / f"seed_{seed}/lr_3e-5__wd_0p01/lambda_0/oof_metrics.json"
        )
        condition_root = run_root / f"seed_{seed}"
        metric_path = condition_root / key / "oof_metrics.json"
        if not metric_path.is_file():
            wait_for_idle_gpu(f"lambda_{value:g}_seed_{seed}")
            update_status("running_multiseed", selected_lambda=value, seed=seed)
            subprocess.run(
                training_command(condition_root, value, seed, baseline_path),
                check=True,
            )
        baseline = flatten(read_json(baseline_path))
        triplet = flatten(read_json(metric_path))
        rows.append(
            {
                "seed": seed,
                "lambda": value,
                "ce_macro_f1": baseline["macro_f1"],
                **{f"triplet_{name}": metric for name, metric in triplet.items()},
                "delta_f1": triplet["macro_f1"] - baseline["macro_f1"],
                "eligible": eligible(triplet, baseline),
            }
        )

    output = run_root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "LAMBDA_MULTISEED_RESULTS.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregates = {}
    for key_name in (
        "triplet_accuracy",
        "triplet_macro_f1",
        "delta_f1",
        "triplet_c_recall",
        "triplet_c_precision",
        "triplet_c_specificity",
    ):
        aggregates[f"{key_name}_mean"], aggregates[f"{key_name}_sd"] = mean_sd(
            [float(row[key_name]) for row in rows]
        )
    aggregates["positive_delta_seeds"] = sum(row["delta_f1"] > 0 for row in rows)
    aggregates["eligible_all_seeds"] = all(row["eligible"] for row in rows)
    write_json(
        output / "LAMBDA_MULTISEED_SELECTION.json",
        {
            "selected_lambda": value,
            "rows": rows,
            "aggregates": aggregates,
            "test_status": "NOT_ACCESSED",
        },
    )
    report = [
        f"# Lambda {value:g} 독립 seed 확인 결과",
        "",
        "| Seed | CE F1 | Triplet F1 | Delta F1 | Accuracy | C recall | C precision | C specificity | C->A | C->B | 적격 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['seed']} | {pct(row['ce_macro_f1'])} | "
            f"{pct(row['triplet_macro_f1'])} | {100 * row['delta_f1']:+.2f} pp | "
            f"{pct(row['triplet_accuracy'])} | {pct(row['triplet_c_recall'])} | "
            f"{pct(row['triplet_c_precision'])} | {pct(row['triplet_c_specificity'])} | "
            f"{row['triplet_c_to_a']} | {row['triplet_c_to_b']} | "
            f"{'✓' if row['eligible'] else '—'} |"
        )
    report.extend(
        [
            "",
            f"평균 Triplet F1: **{pct(aggregates['triplet_macro_f1_mean'])}**",
            f"평균 CE 대비 변화: **{100 * aggregates['delta_f1_mean']:+.2f} pp**",
            f"Triplet 승리 seed: **{aggregates['positive_delta_seeds']}/3**",
        ]
    )
    (output / "LAMBDA_MULTISEED_RESULTS_KO.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    update_status(
        "multiseed_complete",
        selected_lambda=value,
        aggregates=aggregates,
    )


def main() -> None:
    acquire_lock()
    try:
        selection = wait_for_screening()
        selected = selection.get("selected_lambda")
        if selected is None:
            raise RuntimeError("Lambda screening produced no eligible selection")
        selected_value = float(selected)
        boundary = bool(selection.get("boundary_selected", False))
        if boundary:
            run_boundary()
        else:
            run_multiseed(selected_value)
    except Exception as error:
        update_status("failed", error=repr(error))
        raise
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
