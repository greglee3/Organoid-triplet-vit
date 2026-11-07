from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


SEEDS = (27182, 16180, 57721)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize margin=0.2 independent-seed confirmation.")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.2)
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def mean_sd(rows: list[dict], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows]
    return mean(values), stdev(values)


def main() -> None:
    args = parse_args()
    margin_key = "margin_" + format(args.margin, ".10g").replace(".", "p")
    rows = []
    for seed in SEEDS:
        baseline = load(args.baseline_root / f"seed_{seed}" / "lr_3e-5__wd_0p01" / "lambda_0" / "oof_metrics.json")
        triplet = load(args.confirmation_root / f"seed_{seed}" / margin_key / "lambda_0p0025" / "oof_metrics.json")
        eligible = (
            triplet["c_specificity"] >= 0.90
            and triplet["c_recall"] >= baseline["c_recall"] - 0.02
            and triplet["c_precision"] >= baseline["c_precision"] - 0.03
        )
        rows.append({
            "seed": seed,
            "ce_accuracy": float(baseline["accuracy"]),
            "triplet_accuracy": float(triplet["accuracy"]),
            "ce_macro_f1": float(baseline["macro_f1"]),
            "triplet_macro_f1": float(triplet["macro_f1"]),
            "delta_f1": float(triplet["macro_f1"] - baseline["macro_f1"]),
            "ce_c_recall": float(baseline["c_recall"]),
            "triplet_c_recall": float(triplet["c_recall"]),
            "triplet_c_precision": float(triplet["c_precision"]),
            "triplet_c_specificity": float(triplet["c_specificity"]),
            "ce_c_to_a": int(baseline["c_to_a"]),
            "triplet_c_to_a": int(triplet["c_to_a"]),
            "ce_c_to_b": int(baseline["c_to_b"]),
            "triplet_c_to_b": int(triplet["c_to_b"]),
            "eligible": eligible,
        })

    aggregates = {}
    for key in (
        "ce_accuracy", "triplet_accuracy", "ce_macro_f1", "triplet_macro_f1",
        "delta_f1", "ce_c_recall", "triplet_c_recall", "triplet_c_precision",
        "triplet_c_specificity", "ce_c_to_a", "triplet_c_to_a", "ce_c_to_b", "triplet_c_to_b",
    ):
        avg, sd = mean_sd(rows, key)
        aggregates[f"{key}_mean"] = avg
        aggregates[f"{key}_sd"] = sd
    aggregates["positive_delta_seeds"] = sum(row["delta_f1"] > 0 for row in rows)
    aggregates["eligible_all_seeds"] = all(row["eligible"] for row in rows)
    aggregates["confirmation_pass"] = (
        aggregates["positive_delta_seeds"] >= 2
        and aggregates["delta_f1_mean"] > 0
        and aggregates["eligible_all_seeds"]
    )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "MARGIN_MULTISEED_RESULTS.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "MARGIN_MULTISEED_SUMMARY.json").write_text(
        json.dumps({"rows": rows, "aggregates": aggregates, "test_status": "NOT_ACCESSED"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        f"# Triplet Final Tuning: Margin={args.margin:g} 독립 seed 확인",
        "",
        f"- lambda=0.0025, uniform margin={args.margin:g}, semi-hard Triplet",
        "- seeds 27182, 16180, 57721; 각 3-fold train-only OOF, 20 epochs",
        "- test split: 접근하지 않음",
        "",
        "| Seed | CE F1 | Triplet F1 | Delta F1 | Triplet Acc | C recall CE->T | C precision | C specificity | C->A CE->T | C->B CE->T | 적격 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['seed']} | {pct(row['ce_macro_f1'])} | {pct(row['triplet_macro_f1'])} | "
            f"{100 * row['delta_f1']:+.2f} pp | {pct(row['triplet_accuracy'])} | "
            f"{pct(row['ce_c_recall'])}->{pct(row['triplet_c_recall'])} | "
            f"{pct(row['triplet_c_precision'])} | {pct(row['triplet_c_specificity'])} | "
            f"{row['ce_c_to_a']}->{row['triplet_c_to_a']} | {row['ce_c_to_b']}->{row['triplet_c_to_b']} | "
            f"{'✓' if row['eligible'] else '—'} |"
        )
    report.extend([
        "",
        "## 세 seed 평균",
        "",
        f"- Triplet Accuracy: **{pct(aggregates['triplet_accuracy_mean'])}**",
        f"- Triplet Macro-F1: **{pct(aggregates['triplet_macro_f1_mean'])}**",
        f"- CE 대비 Delta F1: **{100 * aggregates['delta_f1_mean']:+.2f} pp**",
        f"- Triplet 승리 seed: **{aggregates['positive_delta_seeds']}/3**",
        f"- 모든 seed 안전 조건 통과: **{'예' if aggregates['eligible_all_seeds'] else '아니오'}**",
        f"- 독립 seed 확인 통과: **{'예' if aggregates['confirmation_pass'] else '아니오'}**",
    ])
    text = "\n".join(report)
    (args.output / "MARGIN_MULTISEED_RESULTS_KO.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
