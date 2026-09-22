from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


METRICS = [
    "accuracy",
    "macro_f1",
    "a_f1",
    "b_f1",
    "c_f1",
    "c_recall",
    "c_precision",
    "c_specificity",
    "c_to_a",
    "c_to_b",
    "silhouette",
    "gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paired five-seed CE-ViT vs Triplet-ViT OOF results."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=str, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def class_f1(confusion: list[list[int]], index: int) -> float:
    true_positive = float(confusion[index][index])
    predicted = float(sum(row[index] for row in confusion))
    actual = float(sum(confusion[index]))
    precision = safe_div(true_positive, predicted)
    recall = safe_div(true_positive, actual)
    return safe_div(2.0 * precision * recall, precision + recall)


def flatten(metrics: dict[str, Any]) -> dict[str, float]:
    confusion = metrics["confusion_matrix"]
    embedding = metrics["embedding_metrics"]
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "a_f1": class_f1(confusion, 0),
        "b_f1": class_f1(confusion, 1),
        "c_f1": class_f1(confusion, 2),
        "c_recall": float(metrics["c_recall"]),
        "c_precision": float(metrics["c_precision"]),
        "c_specificity": float(metrics["c_specificity"]),
        "c_to_a": float(metrics["c_to_a"]),
        "c_to_b": float(metrics["c_to_b"]),
        "silhouette": float(embedding["silhouette_cosine"]),
        "gap": float(embedding["between_minus_within_distance"]),
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def paired_t_interval(values: list[float]) -> tuple[float, float]:
    # Two-sided 95% Student-t critical values for the planned n=5 analysis.
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
    mean, sd = mean_sd(values)
    degrees = len(values) - 1
    if degrees not in critical:
        raise ValueError("This confirmatory summary expects 2-5 paired seeds")
    half_width = critical[degrees] * sd / math.sqrt(len(values))
    return mean - half_width, mean + half_width


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def mean_sd_pct(values: list[float]) -> str:
    mean, sd = mean_sd(values)
    return f"{mean * 100:.2f}±{sd * 100:.2f}%"


def main() -> None:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Exactly five unique preregistered seeds are required")

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_root = args.run_root / f"seed_{seed}"
        ce_path = seed_root / "lambda_0/oof_metrics.json"
        triplet_path = seed_root / "lambda_0p0025/oof_metrics.json"
        if not ce_path.is_file() or not triplet_path.is_file():
            raise FileNotFoundError(
                f"Incomplete seed {seed}: {ce_path} or {triplet_path} is missing"
            )
        ce = flatten(read_json(ce_path))
        triplet = flatten(read_json(triplet_path))
        rows.append(
            {
                "seed": seed,
                "ce": ce,
                "triplet": triplet,
                "delta": {metric: triplet[metric] - ce[metric] for metric in METRICS},
            }
        )

    summary: dict[str, Any] = {
        "protocol": {
            "seeds": seeds,
            "folds": 5,
            "epochs": 20,
            "lambda_triplet": 0.0025,
            "uniform_margin": 0.10,
            "mining": "semi-hard",
            "embedding": "shared ViT pre-logits 384-D, L2",
            "distance": "cosine",
            "test_status": "NOT_ACCESSED",
        },
        "metrics": {},
    }
    for metric in METRICS:
        ce_values = [row["ce"][metric] for row in rows]
        triplet_values = [row["triplet"][metric] for row in rows]
        deltas = [row["delta"][metric] for row in rows]
        ci_low, ci_high = paired_t_interval(deltas)
        summary["metrics"][metric] = {
            "ce_mean": statistics.mean(ce_values),
            "ce_sd": statistics.stdev(ce_values),
            "triplet_mean": statistics.mean(triplet_values),
            "triplet_sd": statistics.stdev(triplet_values),
            "paired_delta_mean": statistics.mean(deltas),
            "paired_delta_sd": statistics.stdev(deltas),
            "paired_delta_95ci": [ci_low, ci_high],
            "triplet_seed_wins": sum(value > 0 for value in deltas),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "CE_VS_TRIPLET_RESULTS.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=True)

    with (args.output / "PAIRED_SEED_RESULTS.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = ["seed"]
        for metric in METRICS:
            fieldnames.extend([f"ce_{metric}", f"triplet_{metric}", f"delta_{metric}"])
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row: dict[str, Any] = {"seed": row["seed"]}
            for metric in METRICS:
                output_row[f"ce_{metric}"] = row["ce"][metric]
                output_row[f"triplet_{metric}"] = row["triplet"][metric]
                output_row[f"delta_{metric}"] = row["delta"][metric]
            writer.writerow(output_row)

    lines = [
        "# 최종 CE-ViT 대 Triplet-ViT 확인 결과",
        "",
        "## 프로토콜",
        "",
        "- 새 seed 5개 × 5-fold train-only OOF",
        "- 동일 초기화·fold·batch 순서의 paired CE 대조",
        "- Triplet: semi-hard, λ=0.0025, uniform margin=0.10",
        "- 공식 test split: **접근하지 않음**",
        "",
        "## Seed별 핵심 결과",
        "",
        "| Seed | CE Acc | Triplet Acc | ΔAcc | CE Macro-F1 | Triplet Macro-F1 | ΔF1 | C recall CE→T | C specificity CE→T | C→A CE→T |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {pct(row['ce']['accuracy'])} | "
            f"{pct(row['triplet']['accuracy'])} | {pp(row['delta']['accuracy'])} | "
            f"{pct(row['ce']['macro_f1'])} | {pct(row['triplet']['macro_f1'])} | "
            f"{pp(row['delta']['macro_f1'])} | "
            f"{pct(row['ce']['c_recall'])}→{pct(row['triplet']['c_recall'])} | "
            f"{pct(row['ce']['c_specificity'])}→{pct(row['triplet']['c_specificity'])} | "
            f"{int(row['ce']['c_to_a'])}→{int(row['triplet']['c_to_a'])} |"
        )

    lines.extend(
        [
            "",
            "## 5-seed 평균±표준편차",
            "",
            "| 지표 | CE-ViT | Triplet-ViT | Paired Δ | Δ 95% CI | Triplet 승리 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    labels = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro-F1",
        "a_f1": "A F1",
        "b_f1": "B F1",
        "c_f1": "C F1",
        "c_recall": "C recall",
        "c_precision": "C precision",
        "c_specificity": "C specificity",
        "c_to_a": "C→A (장)",
        "c_to_b": "C→B (장)",
        "silhouette": "Silhouette",
        "gap": "Embedding gap",
    }
    for metric in METRICS:
        ce_values = [row["ce"][metric] for row in rows]
        triplet_values = [row["triplet"][metric] for row in rows]
        result = summary["metrics"][metric]
        if metric in {"c_to_a", "c_to_b"}:
            ce_text = f"{statistics.mean(ce_values):.2f}±{statistics.stdev(ce_values):.2f}"
            triplet_text = f"{statistics.mean(triplet_values):.2f}±{statistics.stdev(triplet_values):.2f}"
            delta_text = f"{result['paired_delta_mean']:+.2f}"
            ci_text = (
                f"[{result['paired_delta_95ci'][0]:+.2f}, "
                f"{result['paired_delta_95ci'][1]:+.2f}]"
            )
        elif metric in {"silhouette", "gap"}:
            ce_text = f"{statistics.mean(ce_values):.4f}±{statistics.stdev(ce_values):.4f}"
            triplet_text = f"{statistics.mean(triplet_values):.4f}±{statistics.stdev(triplet_values):.4f}"
            delta_text = f"{result['paired_delta_mean']:+.4f}"
            ci_text = (
                f"[{result['paired_delta_95ci'][0]:+.4f}, "
                f"{result['paired_delta_95ci'][1]:+.4f}]"
            )
        else:
            ce_text = mean_sd_pct(ce_values)
            triplet_text = mean_sd_pct(triplet_values)
            delta_text = pp(result["paired_delta_mean"])
            ci_text = (
                f"[{pp(result['paired_delta_95ci'][0])}, "
                f"{pp(result['paired_delta_95ci'][1])}]"
            )
        lines.append(
            f"| {labels[metric]} | {ce_text} | {triplet_text} | {delta_text} | "
            f"{ci_text} | {result['triplet_seed_wins']}/5 |"
        )

    macro = summary["metrics"]["macro_f1"]
    lines.extend(
        [
            "",
            "## 사전 정의 결론",
            "",
            f"- 평균 paired ΔMacro-F1: **{pp(macro['paired_delta_mean'])}**",
            f"- 95% CI: **[{pp(macro['paired_delta_95ci'][0])}, {pp(macro['paired_delta_95ci'][1])}]**",
            f"- Triplet Macro-F1 승리 seed: **{macro['triplet_seed_wins']}/5**",
            "",
            "해석은 평균 차이, 신뢰구간, seed 승리 일관성을 함께 사용한다. 결과를 본 뒤 seed를 추가·삭제하거나 교체하지 않는다.",
        ]
    )
    (args.output / "CE_VS_TRIPLET_RESULTS_KO.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

