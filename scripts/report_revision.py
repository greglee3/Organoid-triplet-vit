"""Recompute current ten-seed results from published, image-aligned probabilities.

CPU only: no torch, training, checkpoint loading, or network access. All reported
metrics are computed separately within each seed before averaging across seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("A", "B", "C")
MODELS = ("CE-only ViT", "Triplet-ViT", "EfficientNetV2-S", "50:50 ensemble")
SEEDS = (70384, 12731, 76313, 98831, 80324, 104729, 130363, 155921, 180749, 205019)
METRICS = ("accuracy", "macro_f1", "c_recall", "c_precision", "c_specificity", "c_f1")
COLUMNS = ("seed", "model", "sample", "true_label", "prob_A", "prob_B", "prob_C")


def load_predictions(path, expected_seeds=SEEDS, expected_class_count=100):
    """Validate and align all groups by sample ID, preserving original CSV order.

    The first group's sample order is retained for deterministic bootstrap
    resampling. Other groups may have different row order, but must contain
    exactly the same unique identities and labels.
    """
    groups = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"Prediction CSV requires columns: {', '.join(COLUMNS)}")
        for line, row in enumerate(reader, start=2):
            try:
                seed, model, sample = int(row["seed"]), row["model"], row["sample"]
                label = CLASSES.index(row["true_label"])
                probability = [float(row[f"prob_{name}"]) for name in CLASSES]
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid prediction at CSV line {line}") from error
            if model not in MODELS or not sample:
                raise ValueError(f"Unknown model or empty sample at CSV line {line}")
            group = groups.setdefault((seed, model), {})
            if sample in group:
                raise ValueError(f"Duplicate sample {sample!r} for seed/model {(seed, model)}")
            group[sample] = (label, probability)
    if not groups:
        raise ValueError("Prediction CSV is empty")
    seeds = tuple(expected_seeds)
    expected_groups = {(seed, model) for seed in seeds for model in MODELS}
    if set(groups) != expected_groups:
        raise ValueError(f"Seed/model coverage mismatch: missing={expected_groups - set(groups)}, "
                         f"unexpected={set(groups) - expected_groups}")
    first = next(iter(groups.values()))
    samples = tuple(first)
    labels = np.asarray([first[sample][0] for sample in samples], dtype=int)
    counts = np.bincount(labels, minlength=3)
    if np.any(counts == 0):
        raise ValueError("Every class must be represented")
    if expected_class_count is not None and not np.all(counts == expected_class_count):
        raise ValueError(f"Expected {expected_class_count} images per class, found {counts.tolist()}")
    probabilities = np.empty((len(seeds), len(MODELS), len(samples), 3), dtype=float)
    for seed_index, seed in enumerate(seeds):
        for model_index, model in enumerate(MODELS):
            group = groups[seed, model]
            if set(group) != set(samples):
                raise ValueError(f"Sample identities differ for {(seed, model)}")
            observed = np.asarray([group[sample][0] for sample in samples])
            if not np.array_equal(labels, observed):
                raise ValueError(f"Labels differ for {(seed, model)}")
            probability = np.asarray([group[sample][1] for sample in samples])
            if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
                raise ValueError(f"Probabilities must be finite and in [0, 1]: {(seed, model)}")
            if not np.allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-5):
                raise ValueError(f"Probabilities do not sum to one: {(seed, model)}")
            probabilities[seed_index, model_index] = probability
    expected_ensemble = 0.5 * probabilities[:, 1] + 0.5 * probabilities[:, 2]
    if not np.allclose(probabilities[:, 3], expected_ensemble, rtol=0, atol=1e-7):
        raise ValueError("50:50 ensemble probabilities differ from 0.5 Triplet + 0.5 EfficientNet")
    return {"seeds": seeds, "samples": samples, "labels": labels, "probabilities": probabilities}


def confusion_metrics(matrices):
    """Vectorized six classification metrics, in proportions, from (..., 3, 3)."""
    matrices = np.asarray(matrices)
    true_positive = np.diagonal(matrices, axis1=-2, axis2=-1)
    actual, called = matrices.sum(-1), matrices.sum(-2)
    f1 = np.divide(2 * true_positive, actual + called,
                   out=np.zeros_like(true_positive, dtype=float), where=(actual + called) != 0)
    precision = np.divide(true_positive[..., 2], called[..., 2],
                          out=np.zeros_like(true_positive[..., 2], dtype=float), where=called[..., 2] != 0)
    return np.stack((true_positive.sum(-1) / matrices.sum((-1, -2)), f1.mean(-1),
                     true_positive[..., 2] / actual[..., 2], precision,
                     1 - matrices[..., :2, 2].sum(-1) / actual[..., :2].sum(-1), f1[..., 2]), axis=-1)


def calculate_metrics(labels, probabilities):
    prediction = probabilities.argmax(axis=1)
    matrix = np.bincount(3 * labels + prediction, minlength=9).reshape(3, 3)
    true_positive = np.diag(matrix)
    actual, called = matrix.sum(1), matrix.sum(0)
    f1 = np.divide(2 * true_positive, actual + called,
                   out=np.zeros(3, dtype=float), where=(actual + called) != 0)
    result = dict(zip(METRICS, confusion_metrics(matrix).tolist()))
    result.update(evaluated_samples=len(labels), confusion_matrix=matrix.tolist(),
                  per_class_f1=dict(zip(CLASSES, f1.tolist())),
                  c_to_a=int(matrix[2, 0]), c_to_b=int(matrix[2, 1]), false_c=int(matrix[:2, 2].sum()))
    result["roc_auc_ovr"] = {name: float(roc_auc_score(labels == index, probabilities[:, index]))
                              for index, name in enumerate(CLASSES)}
    result["average_precision_ovr"] = {
        name: float(average_precision_score(labels == index, probabilities[:, index]))
        for index, name in enumerate(CLASSES)}
    result["macro_roc_auc_ovr"] = float(np.mean(list(result["roc_auc_ovr"].values())))
    result["macro_average_precision_ovr"] = float(np.mean(list(result["average_precision_ovr"].values())))
    return result


def flatten_metrics(metrics):
    result = {key: metrics[key] for key in METRICS}
    for group in ("per_class_f1", "roc_auc_ovr", "average_precision_ovr"):
        result.update({f"{group}.{name}": value for name, value in metrics[group].items()})
    for key in ("macro_roc_auc_ovr", "macro_average_precision_ovr", "c_to_a", "c_to_b", "false_c"):
        result[key] = metrics[key]
    return result


def summarize(data):
    rows = []
    for seed_index, seed in enumerate(data["seeds"]):
        for model_index, model in enumerate(MODELS):
            rows.append({"seed": seed, "model": model,
                         "metrics": calculate_metrics(data["labels"], data["probabilities"][seed_index, model_index])})
    if len(data["seeds"]) < 2:
        raise ValueError("At least two seeds are required for sample SD")
    summary, mean_confusion = {}, {}
    for model in MODELS:
        selected = [row["metrics"] for row in rows if row["model"] == model]
        flattened = [flatten_metrics(metrics) for metrics in selected]
        summary[model] = {key: {"mean": float(np.mean([row[key] for row in flattened])),
                                "sd": float(np.std([row[key] for row in flattened], ddof=1))}
                          for key in flattened[0]}
        mean_confusion[model] = np.mean([metrics["confusion_matrix"] for metrics in selected], axis=0).tolist()
    return {"rows": rows, "summary": summary, "mean_confusion_matrices": mean_confusion}


def verify_reference(report, reference, tolerance=1e-10):
    """Compare every recorded reference metric, including its percent summaries."""
    reference_rows = reference["rows"]
    expected = {(row["seed"], row["model"]): row["metrics"] for row in reference_rows}
    if len(expected) != len(reference_rows):
        raise ValueError("Reference contains duplicate seed/model rows")
    observed = {(row["seed"], row["model"]): row["metrics"] for row in report["rows"]}
    if set(expected) != set(observed):
        raise ValueError("Reference seed/model coverage differs from predictions")

    def compare(actual, wanted, location):
        if isinstance(wanted, dict):
            for key, value in wanted.items():
                if key not in actual:
                    raise ValueError(f"Unsupported reference metric: {location}.{key}")
                compare(actual[key], value, f"{location}.{key}")
        elif not np.allclose(actual, wanted, rtol=0, atol=tolerance):
            raise ValueError(f"Reference mismatch: {location}")

    for key, wanted in expected.items():
        compare(observed[key], wanted, str(key))
    for model, metrics in reference.get("summary", {}).items():
        for metric, values in metrics.items():
            if metric not in METRICS:
                raise ValueError(f"Unsupported reference summary metric: {metric}")
            compare({key: 100 * report["summary"][model][metric][key] for key in ("mean", "sd")},
                    values, f"summary.{model}.{metric}")


def holm(p_values):
    adjusted = np.empty(len(p_values), dtype=float)
    previous = 0.0
    for rank, index in enumerate(np.argsort(p_values)):
        previous = max(previous, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(previous, 1.0)
    return adjusted.tolist()


def paired_uncertainty(data, replicates=20000, rng_seed=20261102):
    """Match the original CE/Triplet paired image bootstrap and McNemar method.

    No seed-level t-tests are performed. Samples are resampled within class,
    using identical weights across all fitted seed/model pairs. The interval
    targets the mean of the per-seed metric differences, not pooled predictions.
    """
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    labels = data["labels"]
    predictions = data["probabilities"][:, :2].argmax(-1)
    matrices = np.asarray([[np.bincount(3 * labels + prediction, minlength=9).reshape(3, 3)
                            for prediction in pair] for pair in predictions])
    values = confusion_metrics(matrices) * 100
    differences = values[:, 1] - values[:, 0]
    bootstrap = np.empty((replicates, len(data["seeds"]), len(METRICS)))
    rng = np.random.default_rng(rng_seed)
    for start in range(0, replicates, 500):
        count = min(500, replicates - start)
        weights = np.zeros((count, len(labels)), dtype=np.int32)
        for label in range(3):
            ids = np.flatnonzero(labels == label)
            weights[:, ids] = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=count)
        for seed_index, pair in enumerate(predictions):
            pair_values = []
            for prediction in pair:
                codes = 3 * labels + prediction
                matrix = np.stack([weights[:, codes == code].sum(1) for code in range(9)], axis=1)
                pair_values.append(confusion_metrics(matrix.reshape(count, 3, 3)) * 100)
            bootstrap[start:start + count, seed_index] = pair_values[1] - pair_values[0]
    aggregate = {key: {"delta_pp": float(differences[:, index].mean()),
                       "paired_image_bootstrap_ci95_pp": np.quantile(bootstrap[:, :, index].mean(1), [.025, .975]).tolist()}
                 for index, key in enumerate(METRICS)}
    per_seed = []
    for index, seed in enumerate(data["seeds"]):
        ce, triplet = predictions[index, 0] == labels, predictions[index, 1] == labels
        ce_only, triplet_only = int((ce & ~triplet).sum()), int((~ce & triplet).sum())
        p_value = float(binomtest(ce_only, ce_only + triplet_only, .5).pvalue) if ce_only + triplet_only else 1.0
        per_seed.append({"seed": seed, "ce_only_correct": ce_only, "triplet_only_correct": triplet_only,
                         "mcnemar_exact_two_sided_p": p_value,
                         "metrics": {key: {"ce": float(values[index, 0, metric]),
                                            "triplet": float(values[index, 1, metric]),
                                            "delta_pp": float(differences[index, metric]),
                                            "paired_image_bootstrap_ci95_pp": np.quantile(bootstrap[:, index, metric], [.025, .975]).tolist()}
                                     for metric, key in enumerate(METRICS)}})
    for row, adjusted in zip(per_seed, holm([row["mcnemar_exact_two_sided_p"] for row in per_seed])):
        row["mcnemar_p_holm_across_seeds"] = adjusted
    return {"protocol": {
        "comparison": "Triplet-ViT minus CE-only ViT", "bootstrap_replicates": replicates,
        "bootstrap_rng_seed": rng_seed, "bootstrap_chunk_size": 500,
        "bootstrap": "Class-stratified paired percentile image bootstrap; identical image weights across all seeds/models; average per-seed metric differences. Conditional on class counts and fixed fitted models.",
        "mcnemar": "Exact two-sided binomial McNemar separately per seed; Holm correction across seeds.",
        "limits": "Exploratory post hoc inference. Images assumed independent; no biological cluster IDs. Bootstrap intervals unadjusted across six metrics. Repeated predictions are not independent additional images. No seed-level t-tests or seed-level confidence intervals."},
        "image_bootstrap_summary": aggregate, "per_seed": per_seed}


def mean_roc(data, points=501):
    grid = np.linspace(0, 1, points)
    rows = []
    for model_index, model in enumerate(MODELS):
        for class_index, name in enumerate(CLASSES):
            curves = []
            for probabilities in data["probabilities"][:, model_index]:
                false_positive, true_positive, _ = roc_curve(data["labels"] == class_index, probabilities[:, class_index])
                interpolated = np.interp(grid, false_positive, true_positive)
                interpolated[0], interpolated[-1] = 0.0, 1.0
                curves.append(interpolated)
            mean, sd = np.mean(curves, axis=0), np.std(curves, axis=0, ddof=1)
            rows.extend({"model": model, "class": name, "fpr": float(x),
                         "mean_tpr": float(y), "sd_tpr": float(s)} for x, y, s in zip(grid, mean, sd))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output, report, roc_rows, overwrite=False, figures=True):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not overwrite:
        raise FileExistsError(f"Output is not an empty directory: {output}; choose a new path or use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    (output / "revision_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output / "per_seed_metrics.csv", [{"seed": row["seed"], "model": row["model"],
              **flatten_metrics(row["metrics"])} for row in report["rows"]])
    write_csv(output / "seed_summary.csv", [{"model": model, "metric": metric, **values}
              for model, metrics in report["summary"].items() for metric, values in metrics.items()])
    write_csv(output / "mean_confusion_matrices.csv", [{"model": model, "actual": CLASSES[i],
              "predicted": CLASSES[j], "mean_count": matrix[i][j]}
              for model, matrix in report["mean_confusion_matrices"].items() for i in range(3) for j in range(3)])
    write_csv(output / "classwise_mean_roc.csv", roc_rows)
    if figures:
        plot_figures(output, report, roc_rows)


def plot_figures(output, report, roc_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    colors = ("#64748b", "#2563eb", "#d97706", "#059669")
    for axis, name in zip(axes, CLASSES):
        for model, color in zip(MODELS, colors):
            selected = [row for row in roc_rows if row["model"] == model and row["class"] == name]
            x, y, sd = (np.asarray([row[key] for row in selected]) for key in ("fpr", "mean_tpr", "sd_tpr"))
            auc = report["summary"][model][f"roc_auc_ovr.{name}"]
            axis.plot(x, y, color=color, label=f"{model}: {auc['mean']:.3f} ± {auc['sd']:.3f}")
            axis.fill_between(x, np.maximum(y - sd, 0), np.minimum(y + sd, 1), color=color, alpha=.10)
        axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=.7)
        axis.set(title=f"Class {name}", xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
        axis.legend(loc="lower right", fontsize=7)
    figure.suptitle("Mean seed-specific ROC; shading = ±1 sample SD; legend = per-seed AUC mean ± SD", fontsize=10)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"classwise_mean_roc.{suffix}", dpi=200)
    plt.close(figure)
    figure, axes = plt.subplots(1, 4, figsize=(13, 4.2), constrained_layout=True)
    matrices = report["mean_confusion_matrices"]
    maximum = max(np.max(matrix) for matrix in matrices.values())
    for axis, model in zip(axes, MODELS):
        matrix = np.asarray(matrices[model])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        axis.set(title=model, xlabel="Predicted class", ylabel="Actual class", xticks=range(3), yticks=range(3),
                 xticklabels=CLASSES, yticklabels=CLASSES)
        for i in range(3):
            for j in range(3):
                axis.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                          color="white" if matrix[i, j] > maximum / 2 else "black")
    figure.suptitle("Mean confusion counts across seeds (not pooled independent observations)", fontsize=10)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"mean_confusion_matrices.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "revision/results/test_predictions.csv")
    parser.add_argument("--reference", type=Path, default=ROOT / "revision/results/ten_seed_reference.json")
    parser.add_argument("--output", type=Path, default=ROOT / "generated/revision")
    parser.add_argument("--overwrite", action="store_true", help="Replace this reporter's generated files; never delete other files")
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261102)
    parser.add_argument("--no-bootstrap", action="store_true", help="Skip paired bootstrap/McNemar output")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())) and not args.overwrite:
        parser.error("Output exists and is not empty; select a new --output or explicitly pass --overwrite")
    data = load_predictions(args.predictions)
    report = summarize(data)
    reference = json.loads(args.reference.read_text(encoding="utf-8-sig"))
    verify_reference(report, reference)
    report["protocol"] = {
        "seeds": list(data["seeds"]), "models": list(MODELS), "sample_count": len(data["samples"]),
        "class_counts": np.bincount(data["labels"], minlength=3).tolist(),
        "metric_units": "Proportions 0–1 except confusion/error counts; bootstrap differences use percentage points.",
        "aggregation": "Compute metrics per seed, then arithmetic mean and sample SD (ddof=1). Do not average predictions across seeds.",
        "roc": "Interpolate each seed's classwise ROC onto 501 fixed FPR points; force endpoints (0,0)/(1,1), then mean and sample SD. AUC is computed per seed, not from mean ROC.",
        "reference_verified": True, "ensemble_probability_arithmetic_verified": True,
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "sample_order_sha256": hashlib.sha256(json.dumps(data["samples"], ensure_ascii=False).encode("utf-8")).hexdigest()}
    if not args.no_bootstrap:
        report["paired_uncertainty"] = paired_uncertainty(data, args.bootstrap_replicates, args.bootstrap_seed)
    write_outputs(args.output, report, mean_roc(data), overwrite=args.overwrite, figures=not args.no_figures)
    print("PASS: image alignment, ensemble arithmetic, all per-seed reference metrics and mean/sample SD.")
    for model in MODELS:
        value = report["summary"][model]["accuracy"]
        print(f"{model}: accuracy {100 * value['mean']:.2f} ± {100 * value['sd']:.2f}%")
    print(f"Wrote revision report to {args.output}")


if __name__ == "__main__":
    main()
