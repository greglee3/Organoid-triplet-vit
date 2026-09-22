from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


CLASS_NAMES = ("A", "B", "C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the 50:50 Triplet-ViT/Notebook-EfficientNet ensemble before test.")
    parser.add_argument("--triplet-oof-root", type=Path, required=True)
    parser.add_argument("--efficient-oof-root", type=Path, required=True)
    parser.add_argument("--folds-json", type=Path, required=True)
    parser.add_argument("--triplet-lock-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_prediction_rows(path: Path) -> list[tuple[int, str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    index_key = "train_index" if "train_index" in rows[0] else "index"
    return [(int(row[index_key]), row["sample"], row["true_label"]) for row in rows]


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1, 2], zero_division=0
    )
    c_tp = int(matrix[2, 2])
    c_fp = int(matrix[:2, 2].sum())
    c_tn = int(matrix[:2, :2].sum())
    return {
        "evaluated_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "per_class_precision": dict(zip(CLASS_NAMES, precision.tolist())),
        "per_class_recall": dict(zip(CLASS_NAMES, recall.tolist())),
        "per_class_f1": dict(zip(CLASS_NAMES, class_f1.tolist())),
        "per_class_support": dict(zip(CLASS_NAMES, support.astype(int).tolist())),
        "c_recall": float(recall[2]),
        "c_precision": float(precision[2]),
        "c_specificity": c_tn / (c_tn + c_fp) if c_tn + c_fp else 0.0,
        "c_to_a": int(matrix[2, 0]),
        "c_to_b": int(matrix[2, 1]),
        "confusion_matrix": matrix.tolist(),
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"Ensemble lock already exists: {args.output}")

    fold_manifest = json.loads(args.folds_json.read_text(encoding="utf-8"))
    triplet_config = json.loads(args.triplet_lock_config.read_text(encoding="utf-8"))
    if int(fold_manifest["seed"]) != 42 or int(triplet_config["seed"]) != 42:
        raise RuntimeError("This final ensemble protocol requires the locked seed 42 split")
    if len(fold_manifest["folds"]) != 5 or int(triplet_config["final_folds"]) != 5:
        raise RuntimeError("This final ensemble protocol requires five folds")

    triplet_prob_path = args.triplet_oof_root / "oof_probabilities.npy"
    efficient_prob_path = args.efficient_oof_root / "oof_probabilities.npy"
    triplet_probs = np.load(triplet_prob_path)
    efficient_probs = np.load(efficient_prob_path)
    if triplet_probs.shape != (1200, 3) or efficient_probs.shape != (1200, 3):
        raise RuntimeError(f"Expected aligned (1200, 3) OOF arrays, got {triplet_probs.shape} and {efficient_probs.shape}")
    if np.isnan(triplet_probs).any() or np.isnan(efficient_probs).any():
        raise RuntimeError("OOF probabilities contain missing values")

    triplet_rows = read_prediction_rows(args.triplet_oof_root / "oof_predictions.csv")
    efficient_rows = read_prediction_rows(args.efficient_oof_root / "oof_predictions.csv")
    if triplet_rows != efficient_rows or len(triplet_rows) != 1200:
        raise RuntimeError("Triplet and EfficientNet OOF sample orders do not match")
    label_lookup = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = np.asarray([label_lookup[row[2]] for row in triplet_rows], dtype=np.int64)
    ensemble_probs = 0.5 * triplet_probs + 0.5 * efficient_probs

    efficient_metrics = json.loads((args.efficient_oof_root / "oof_metrics.json").read_text(encoding="utf-8"))
    expected_efficient = {
        "model": "tf_efficientnetv2_s",
        "seed": 42,
        "folds": 5,
        "epochs": 50,
        "optimizer": "AdamW",
        "head_lr": 0.00015,
        "backbone_lr": 0.00015,
        "weight_decay": 0.01,
        "trainable_mode": "all",
        "scheduler": {"name": "step", "step_size": 3, "gamma": 0.97},
        "augmentation": {
            "gridmask_probability": 0.3,
            "gridmask_implementation": "notebook",
            "horizontal_flip_probability": 0.5,
            "vertical_flip_probability": 0.5,
        },
        "checkpoint_rule": "minimum validation cross-entropy",
    }
    observed_pairs = {
        "model": efficient_metrics["model"],
        "folds": efficient_metrics["folds"],
        "epochs": efficient_metrics["epochs"],
        "optimizer": efficient_metrics["optimizer"],
        "head_lr": efficient_metrics["head_lr"],
        "backbone_lr": efficient_metrics["backbone_lr"],
        "weight_decay": efficient_metrics["weight_decay"],
        "trainable_mode": efficient_metrics["trainable"],
        "scheduler": efficient_metrics["scheduler"],
        "augmentation": efficient_metrics["augmentation"],
        "checkpoint_rule": efficient_metrics["checkpoint_rule"],
    }
    for key, expected in expected_efficient.items():
        if key == "seed":
            continue
        if observed_pairs[key] != expected:
            raise RuntimeError(f"EfficientNet setting mismatch for {key}: {observed_pairs[key]!r} != {expected!r}")

    checkpoints = []
    for fold in range(1, 6):
        path = args.efficient_oof_root / "folds" / f"fold_{fold}_best.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints.append({"fold": fold, "path": str(path), "sha256": sha256(path)})

    payload = {
        "lock_status": "FROZEN_BEFORE_ENSEMBLE_TEST",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "One 50:50 public-test evaluation after train-only selection",
        "seed": 42,
        "folds": 5,
        "weight_rule": {"triplet_vit": 0.5, "notebook_efficientnet": 0.5, "combination": "arithmetic mean of softmax probabilities"},
        "weight_selection_source": "Prior 3-seed, 5-fold train-only OOF search; no public-test metric used",
        "triplet": {
            "lock_config": str(args.triplet_lock_config),
            "lock_sha256": sha256(args.triplet_lock_config),
            "oof_root": str(args.triplet_oof_root),
            "oof_probabilities_sha256": sha256(triplet_prob_path),
        },
        "efficientnet": {
            "oof_root": str(args.efficient_oof_root),
            "oof_probabilities_sha256": sha256(efficient_prob_path),
            "config": expected_efficient,
            "checkpoints": checkpoints,
        },
        "aligned_train_only_oof": {
            "triplet_vit": metrics(labels, triplet_probs),
            "notebook_efficientnet": metrics(labels, efficient_probs),
            "ensemble_50_50": metrics(labels, ensemble_probs),
        },
        "test_policy": {
            "evaluation_count": 1,
            "weights_must_not_change_after_test": True,
            "model_hyperparameters_must_not_change_after_test": True,
            "project_public_test_already_seen_for_locked_triplet_vit": True,
        },
        "test_status": "NOT_EVALUATED_FOR_THIS_ENSEMBLE",
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["aligned_train_only_oof"], ensure_ascii=False, indent=2))
    print(f"ensemble_lock={args.output}")


if __name__ == "__main__":
    main()
