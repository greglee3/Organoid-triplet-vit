from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


MODELS = {"CE-ViT": "lambda_0", "Triplet-ViT": "lambda_0p0025"}
TARGETS = (0.900, 0.925, 0.928, 0.950)
COLORS = {"CE-ViT": "#E76F51", "Triplet-ViT": "#2A6FDB"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", required=True)
    return parser.parse_args()


def recall_key(specificity):
    return f"recall_at_spec_{specificity:.3f}".replace(".", "p")


def paired_ci(values):
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = critical[len(values) - 1] * sd / math.sqrt(len(values))
    return mean - half, mean + half


def unique_roc(y_true, score):
    fpr, tpr, thresholds = roc_curve(y_true, score)
    xs, ys, ts = [], [], []
    for value in np.unique(fpr):
        mask = fpr == value
        local_tpr = tpr[mask]
        local_threshold = thresholds[mask]
        index = int(np.argmax(local_tpr))
        xs.append(float(value))
        ys.append(float(local_tpr[index]))
        ts.append(float(local_threshold[index]))
    return np.asarray(xs), np.asarray(ys), np.asarray(ts)


def recall_at_specificity(y_true, score, specificity):
    fpr, tpr, _ = unique_roc(y_true, score)
    return float(np.interp(1.0 - specificity, fpr, tpr))


def empirical_point(y_true, score, specificity):
    fpr, tpr, thresholds = roc_curve(y_true, score)
    eligible = np.flatnonzero(fpr <= 1.0 - specificity + 1e-12)
    best_recall = np.max(tpr[eligible])
    candidates = eligible[np.isclose(tpr[eligible], best_recall)]
    index = int(candidates[np.argmax(fpr[candidates])])
    threshold = float(thresholds[index])
    predicted = score >= threshold
    positives = y_true == 1
    negatives = y_true == 0
    return {
        "target_specificity": specificity,
        "threshold": threshold,
        "achieved_specificity": float(np.mean(~predicted[negatives])),
        "recall": float(np.mean(predicted[positives])),
        "predicted_c": int(predicted.sum()),
    }


def calculate_metrics(y_true, score):
    result = {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "average_precision": float(average_precision_score(y_true, score)),
        "partial_auc_fpr_0p10": float(
            roc_auc_score(y_true, score, max_fpr=0.10)
        ),
    }
    for target in TARGETS:
        result[recall_key(target)] = recall_at_specificity(y_true, score, target)
    return result


def load_pair(run_root, seed):
    pair = {}
    reference = None
    for model, directory in MODELS.items():
        path = run_root / f"seed_{seed}" / directory / "oof_predictions.csv"
        with path.open("r", encoding="utf-8", newline="") as stream:
            frame = sorted(
                csv.DictReader(stream), key=lambda row: int(row["train_index"])
            )
        identity = [
            (int(row["train_index"]), row["sample"], row["true_label"])
            for row in frame
        ]
        if reference is None:
            reference = list(identity)
        elif reference != identity:
            raise ValueError(f"OOF rows do not match for seed {seed}")
        y_true = np.asarray(
            [int(row["true_label"] == "C") for row in frame], dtype=int
        )
        score = np.asarray([float(row["prob_C"]) for row in frame], dtype=float)
        if len(frame) != 1200 or int(y_true.sum()) != 400:
            raise ValueError(f"Unexpected sample count for seed {seed}")
        pair[model] = (y_true, score)
    return pair


def curve_arrays(y_true, score, roc_grid, recall_grid):
    fpr, tpr, _ = unique_roc(y_true, score)
    precision, recall, _ = precision_recall_curve(y_true, score)
    recall = recall[::-1]
    precision = precision[::-1]
    unique_recall = np.unique(recall)
    best_precision = np.asarray(
        [np.max(precision[recall == value]) for value in unique_recall]
    )
    return (
        np.interp(roc_grid, fpr, tpr),
        np.interp(recall_grid, unique_recall, best_precision),
    )


def save_curve_figure(curves, rows, output):
    roc_grid = np.linspace(0, 1, 501)
    recall_grid = np.linspace(0, 1, 501)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    for model in MODELS:
        roc_values = np.stack([item[0] for item in curves[model]])
        pr_values = np.stack([item[1] for item in curves[model]])
        roc_mean = roc_values.mean(axis=0)
        roc_sd = roc_values.std(axis=0, ddof=1)
        pr_mean = pr_values.mean(axis=0)
        pr_sd = pr_values.std(axis=0, ddof=1)
        auc_mean = statistics.mean(row[model]["roc_auc"] for row in rows)
        ap_mean = statistics.mean(
            row[model]["average_precision"] for row in rows
        )
        axes[0].plot(
            roc_grid,
            roc_mean,
            color=COLORS[model],
            linewidth=2,
            label=f"{model} (AUC {auc_mean:.3f})",
        )
        axes[0].fill_between(
            roc_grid,
            np.clip(roc_mean - roc_sd, 0, 1),
            np.clip(roc_mean + roc_sd, 0, 1),
            color=COLORS[model],
            alpha=0.14,
            linewidth=0,
        )
        axes[1].plot(
            recall_grid,
            pr_mean,
            color=COLORS[model],
            linewidth=2,
            label=f"{model} (AP {ap_mean:.3f})",
        )
        axes[1].fill_between(
            recall_grid,
            np.clip(pr_mean - pr_sd, 0, 1),
            np.clip(pr_mean + pr_sd, 0, 1),
            color=COLORS[model],
            alpha=0.14,
            linewidth=0,
        )
    axes[0].plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
    axes[0].axvspan(0, 0.10, color="#2A9D8F", alpha=0.07)
    axes[0].set(
        title="Category C: ROC curve",
        xlabel="False-positive rate (1 - specificity)",
        ylabel="True-positive rate (recall)",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axes[1].set(
        title="Category C: precision-recall curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(loc="lower left", fontsize=9)
    fig.suptitle("Paired 5-seed train-only OOF analysis", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "C_ROC_PR_CURVES.png", dpi=350, bbox_inches="tight")
    fig.savefig(output / "C_ROC_PR_CURVES.pdf", bbox_inches="tight")
    plt.close(fig)


def save_specificity_figure(rows, output):
    x = np.arange(len(TARGETS), dtype=float)
    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    for offset, model in zip((-0.08, 0.08), MODELS):
        values = [
            [row[model][recall_key(target)] for row in rows]
            for target in TARGETS
        ]
        means = np.asarray([statistics.mean(item) for item in values]) * 100
        sds = np.asarray([statistics.stdev(item) for item in values]) * 100
        axis.errorbar(
            x + offset,
            means,
            yerr=sds,
            marker="o",
            linewidth=2,
            capsize=4,
            color=COLORS[model],
            label=model,
        )
    axis.set_xticks(x)
    axis.set_xticklabels([f"{target * 100:.1f}%" for target in TARGETS])
    axis.set_xlabel("Matched C specificity")
    axis.set_ylabel("C recall (%)")
    axis.set_title("C recall at identical specificity", fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        output / "C_RECALL_AT_MATCHED_SPECIFICITY.png",
        dpi=350,
        bbox_inches="tight",
    )
    fig.savefig(output / "C_RECALL_AT_MATCHED_SPECIFICITY.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Exactly five unique locked seeds are required")
    args.output.mkdir(parents=True, exist_ok=True)
    roc_grid = np.linspace(0, 1, 501)
    recall_grid = np.linspace(0, 1, 501)
    rows = []
    operating_rows = []
    curves = {model: [] for model in MODELS}

    for seed in seeds:
        pair = load_pair(args.run_root, seed)
        row = {"seed": seed}
        for model, (y_true, score) in pair.items():
            row[model] = calculate_metrics(y_true, score)
            curves[model].append(
                curve_arrays(y_true, score, roc_grid, recall_grid)
            )
            for target in TARGETS:
                operating_rows.append(
                    {"seed": seed, "model": model, **empirical_point(
                        y_true, score, target
                    )}
                )
        rows.append(row)

    metric_labels = {
        "roc_auc": "C ROC-AUC",
        "average_precision": "C average precision (PR-AUC)",
        "partial_auc_fpr_0p10": (
            "C standardized partial AUC (specificity >=90%)"
        ),
        recall_key(0.900): "C recall at specificity 90.0%",
        recall_key(0.925): "C recall at specificity 92.5%",
        recall_key(0.928): "C recall at specificity 92.8%",
        recall_key(0.950): "C recall at specificity 95.0%",
    }
    summary = {
        "protocol": {
            "seeds": seeds,
            "folds": 5,
            "samples_per_seed": 1200,
            "c_samples_per_seed": 400,
            "comparison": "paired train-only OOF probabilities",
            "target_specificities": list(TARGETS),
            "test_status": "NOT_ACCESSED",
        },
        "metrics": {},
    }
    for metric, label in metric_labels.items():
        ce = [row["CE-ViT"][metric] for row in rows]
        triplet = [row["Triplet-ViT"][metric] for row in rows]
        delta = [right - left for left, right in zip(ce, triplet)]
        low, high = paired_ci(delta)
        summary["metrics"][metric] = {
            "label": label,
            "ce_mean": statistics.mean(ce),
            "ce_sd": statistics.stdev(ce),
            "triplet_mean": statistics.mean(triplet),
            "triplet_sd": statistics.stdev(triplet),
            "paired_delta_mean": statistics.mean(delta),
            "paired_delta_sd": statistics.stdev(delta),
            "paired_delta_95ci": [low, high],
            "triplet_seed_wins": sum(value > 0 for value in delta),
        }

    with (args.output / "THRESHOLD_MATCHED_RESULTS.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    fields = ["seed"]
    for metric in metric_labels:
        fields.extend(
            [f"ce_{metric}", f"triplet_{metric}", f"delta_{metric}"]
        )
    with (args.output / "THRESHOLD_MATCHED_PER_SEED.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {"seed": row["seed"]}
            for metric in metric_labels:
                ce = row["CE-ViT"][metric]
                triplet = row["Triplet-ViT"][metric]
                output[f"ce_{metric}"] = ce
                output[f"triplet_{metric}"] = triplet
                output[f"delta_{metric}"] = triplet - ce
            writer.writerow(output)
    with (args.output / "EMPIRICAL_OPERATING_POINTS.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(operating_rows[0]))
        writer.writeheader()
        writer.writerows(operating_rows)

    lines = [
        "# CE-ViT 대 Triplet-ViT 동일 specificity 분석",
        "",
        "## 고정 프로토콜",
        "",
        "- 기존 새 seed 5개 × 5-fold train-only OOF 확률만 사용",
        "- 재학습 및 추가 튜닝 없음",
        "- CE-ViT: lambda_triplet=0",
        "- Triplet-ViT: semi-hard, cosine, lambda_triplet=0.0025, margin=0.10",
        "- C 대 A/B one-vs-rest 분석",
        "- 공식 test split: 접근하지 않음",
        "",
        "## 5-seed 평균±표준편차",
        "",
        "| 지표 | CE-ViT | Triplet-ViT | Paired delta | delta 95% CI | Triplet 승리 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in metric_labels:
        result = summary["metrics"][metric]
        lines.append(
            f"| {result['label']} | "
            f"{result['ce_mean'] * 100:.2f}±{result['ce_sd'] * 100:.2f}% | "
            f"{result['triplet_mean'] * 100:.2f}±{result['triplet_sd'] * 100:.2f}% | "
            f"{result['paired_delta_mean'] * 100:+.2f} pp | "
            f"[{result['paired_delta_95ci'][0] * 100:+.2f}, "
            f"{result['paired_delta_95ci'][1] * 100:+.2f}] pp | "
            f"{result['triplet_seed_wins']}/5 |"
        )

    key_metrics = [
        "average_precision",
        "partial_auc_fpr_0p10",
        recall_key(0.900),
        recall_key(0.928),
    ]
    supported = any(
        summary["metrics"][metric]["paired_delta_95ci"][0] > 0
        for metric in key_metrics
    )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- 동일 specificity의 recall은 C를 과도하게 예측하는 정도를 맞춘 뒤 C 검출력을 비교한다.",
            "- standardized partial AUC는 specificity 90% 이상인 구간의 ROC 성능을 요약한다.",
            "- 신뢰구간은 다섯 학습 seed의 변동성이며 독립 생물학 표본의 불확실성은 아니다.",
            "",
            (
                "**결론:** 적어도 하나의 고-specificity 지표에서 paired 95% CI가 "
                "0보다 높아 Triplet의 C 판별 개선을 지지한다."
                if supported
                else "**결론:** 고-specificity 지표의 paired 95% CI가 0을 명확히 "
                "넘지 않아 C recall 상승을 Triplet 고유의 판별력 향상으로 확정하기 "
                "어렵다. 예측 성향 변화 가능성을 함께 보고해야 한다."
            ),
            "",
            "결과 확인 뒤 threshold, seed 또는 하이퍼파라미터를 변경하지 않는다.",
        ]
    )
    report = args.output / "THRESHOLD_MATCHED_RESULTS_KO.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_curve_figure(curves, rows, args.output)
    save_specificity_figure(rows, args.output)
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
