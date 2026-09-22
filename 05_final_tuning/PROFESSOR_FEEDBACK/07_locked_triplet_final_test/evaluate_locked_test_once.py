from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asano_data import AsanoDataset  # noqa: E402


TRAINER_PATH = (
    PROJECT_ROOT
    / "05_final_tuning"
    / "02_ordinal_aware_triplet"
    / "train_pilot.py"
)
CLASS_NAMES = ("A", "B", "C")


def load_trainer():
    trainer_directory = str(TRAINER_PATH.parent)
    if trainer_directory not in sys.path:
        sys.path.insert(0, trainer_directory)
    spec = importlib.util.spec_from_file_location("locked_triplet_trainer", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load trainer: {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One locked public-test evaluation of paired CE/Triplet ViT.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--lock-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(checkpoint: dict, config: dict, expected_lambda: float, fold: int) -> None:
    if int(checkpoint["fold"]) != fold:
        raise RuntimeError(f"Fold mismatch: expected {fold}")
    if not math.isclose(float(checkpoint["lambda_triplet"]), expected_lambda, abs_tol=1e-12):
        raise RuntimeError(f"Triplet lambda mismatch in fold {fold}")
    values = checkpoint["args"]
    expected = {
        "seed": config["seed"],
        "folds": config["final_folds"],
        "epochs": config["epochs"],
        "ordinal_lambda": config["lambda_ordinal"],
        "mining_mode": config["mining_mode"],
        "adjacent_margin": config["adjacent_margin"],
        "far_margin": config["far_margin"],
        "classes_per_batch": config["classes_per_batch"],
        "samples_per_class": config["samples_per_class"],
        "lr": config["backbone_lr"],
        "head_lr_multiplier": config["head_lr_multiplier"],
        "optimizer": config["optimizer"],
        "weight_decay": config["weight_decay"],
        "warmup_epochs": config["warmup_epochs"],
        "scheduler": config["scheduler"],
        "augmentation": config["augmentation"],
        "embedding_head": config["embedding_head"],
        "triplet_ce_only_epochs": config["triplet_ce_only_epochs"],
        "triplet_ramp_epochs": config["triplet_ramp_epochs"],
    }
    for key, locked in expected.items():
        observed = values[key]
        if isinstance(locked, float):
            matches = math.isclose(float(observed), locked, rel_tol=0.0, abs_tol=1e-12)
        else:
            matches = observed == locked
        if not matches:
            raise RuntimeError(f"Locked setting mismatch for {key}: {observed!r} != {locked!r}")


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, batch_labels in loader:
        logits, _ = model(images.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def extended_metrics(trainer, labels: np.ndarray, probabilities: np.ndarray) -> dict:
    result = trainer.calculate_metrics(labels, probabilities)
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1, 2],
        zero_division=0,
    )
    result["per_class_precision"] = dict(zip(CLASS_NAMES, precision.tolist()))
    result["per_class_recall"] = dict(zip(CLASS_NAMES, recall.tolist()))
    result["per_class_f1"] = dict(zip(CLASS_NAMES, f1.tolist()))
    result["per_class_support"] = dict(zip(CLASS_NAMES, support.astype(int).tolist()))
    return result


def evaluate_condition(
    trainer,
    run_root: Path,
    condition: str,
    expected_lambda: float,
    config: dict,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    fold_probabilities: list[np.ndarray] = []
    labels_reference: np.ndarray | None = None
    for fold in range(1, int(config["final_folds"]) + 1):
        checkpoint_path = run_root / condition / "folds" / f"fold_{fold}_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, config, expected_lambda, fold)
        model = trainer.ViTSharedMetricModel(
            pretrained=False,
            embedding_head=config["embedding_head"],
            projection_dim=128,
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        probabilities, labels = predict(model, loader, device)
        fold_probabilities.append(probabilities)
        if labels_reference is None:
            labels_reference = labels
        elif not np.array_equal(labels_reference, labels):
            raise RuntimeError("Test label order changed between folds")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if labels_reference is None:
        raise RuntimeError("No test predictions were generated")
    ensemble = np.stack(fold_probabilities).mean(axis=0)
    return extended_metrics(trainer, labels_reference, ensemble), ensemble, labels_reference


def bootstrap_delta(
    labels: np.ndarray,
    ce_probabilities: np.ndarray,
    triplet_probabilities: np.ndarray,
    replicates: int = 10_000,
) -> dict:
    rng = np.random.default_rng(20260820)
    ce_predictions = ce_probabilities.argmax(axis=1)
    triplet_predictions = triplet_probabilities.argmax(axis=1)
    accuracy_deltas = np.empty(replicates, dtype=np.float64)
    f1_deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[selected]
        ce_sample = ce_predictions[selected]
        triplet_sample = triplet_predictions[selected]
        accuracy_deltas[index] = np.mean(triplet_sample == sampled_labels) - np.mean(ce_sample == sampled_labels)
        f1_deltas[index] = f1_score(
            sampled_labels,
            triplet_sample,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        ) - f1_score(
            sampled_labels,
            ce_sample,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    def summarize(values: np.ndarray) -> dict:
        lower, upper = np.quantile(values, [0.025, 0.975])
        return {
            "mean": float(values.mean()),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "probability_positive": float(np.mean(values > 0)),
        }
    return {
        "replicates": replicates,
        "seed": 20260820,
        "accuracy_delta": summarize(accuracy_deltas),
        "macro_f1_delta": summarize(f1_deltas),
    }


def exact_mcnemar(labels: np.ndarray, ce_probabilities: np.ndarray, triplet_probabilities: np.ndarray) -> dict:
    ce_correct = ce_probabilities.argmax(axis=1) == labels
    triplet_correct = triplet_probabilities.argmax(axis=1) == labels
    ce_only = int(np.sum(ce_correct & ~triplet_correct))
    triplet_only = int(np.sum(~ce_correct & triplet_correct))
    discordant = ce_only + triplet_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(ce_only, triplet_only)
        tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "ce_correct_triplet_wrong": ce_only,
        "ce_wrong_triplet_correct": triplet_only,
        "discordant_total": discordant,
        "exact_two_sided_p": p_value,
    }


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def write_predictions(path: Path, dataset: AsanoDataset, labels: np.ndarray, ce: np.ndarray, triplet: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "test_index", "sample", "true_label", "ce_prediction", "triplet_prediction",
            "ce_prob_A", "ce_prob_B", "ce_prob_C",
            "triplet_prob_A", "triplet_prob_B", "triplet_prob_C",
        ])
        for index, ((sample, _), label) in enumerate(zip(dataset.samples, labels)):
            writer.writerow([
                index,
                sample,
                CLASS_NAMES[int(label)],
                CLASS_NAMES[int(ce[index].argmax())],
                CLASS_NAMES[int(triplet[index].argmax())],
                *ce[index].tolist(),
                *triplet[index].tolist(),
            ])


def main() -> None:
    args = parse_args()
    if not args.confirm_final_test:
        raise RuntimeError("Refusing to access the public test without --confirm-final-test")
    completed = args.output / "FINAL_TEST_COMPLETED.flag"
    metrics_path = args.output / "FINAL_TEST_RESULTS.json"
    if completed.exists() or metrics_path.exists():
        raise RuntimeError("Locked public test was already evaluated; refusing to run again")
    config = json.loads(args.lock_config.read_text(encoding="utf-8"))
    if config.get("lock_status") != "FROZEN_BEFORE_FINAL_TEST":
        raise RuntimeError("Lock config is not frozen")

    required = {
        "lambda_0": float(config["paired_conditions"]["ce_vit"]["lambda_triplet"]),
        "lambda_0p0025": float(config["paired_conditions"]["triplet_vit"]["lambda_triplet"]),
    }
    # Validate every checkpoint before constructing the test dataset.
    for condition, expected_lambda in required.items():
        for fold in range(1, int(config["final_folds"]) + 1):
            checkpoint_path = args.run_root / condition / "folds" / f"fold_{fold}_best.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            validate_checkpoint(checkpoint, config, expected_lambda, fold)

    trainer = load_trainer()
    trainer.set_deterministic(int(config["seed"]))
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
    ])
    test_dataset = AsanoDataset(args.data, "test", transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ce_metrics, ce_probabilities, labels = evaluate_condition(
        trainer, args.run_root, "lambda_0", required["lambda_0"], config, test_loader, device
    )
    triplet_metrics, triplet_probabilities, triplet_labels = evaluate_condition(
        trainer,
        args.run_root,
        "lambda_0p0025",
        required["lambda_0p0025"],
        config,
        test_loader,
        device,
    )
    if not np.array_equal(labels, triplet_labels):
        raise RuntimeError("CE and Triplet test label order differs")

    delta_keys = (
        "accuracy", "macro_f1", "c_recall", "c_precision", "c_specificity",
        "c_to_a", "c_to_b", "a_to_c", "severe_a_c_errors", "ordinal_mae",
        "quadratic_weighted_kappa",
    )
    delta = {key: triplet_metrics[key] - ce_metrics[key] for key in delta_keys}
    inference = {
        "paired_bootstrap": bootstrap_delta(labels, ce_probabilities, triplet_probabilities),
        "mcnemar": exact_mcnemar(labels, ce_probabilities, triplet_probabilities),
    }
    payload = {
        "protocol": {
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "lock_sha256": sha256(args.lock_config),
            "seed": config["seed"],
            "folds": config["final_folds"],
            "ensemble": config["ensemble_rule"],
            "test_samples": int(len(test_dataset)),
            "selection_used_train_only_oof": True,
            "test_results_must_not_change_triplet_hyperparameters": True,
        },
        "locked_config": config,
        "ce_vit": ce_metrics,
        "triplet_vit": triplet_metrics,
        "delta_triplet_minus_ce": delta,
        "inference": inference,
        "safety_checks": {
            "c_specificity_at_least_90_percent": triplet_metrics["c_specificity"] >= 0.90,
            "c_recall_drop_no_more_than_2pp": triplet_metrics["c_recall"] >= ce_metrics["c_recall"] - 0.02,
            "c_precision_drop_no_more_than_3pp": triplet_metrics["c_precision"] >= ce_metrics["c_precision"] - 0.03,
        },
        "test_status": "FINAL_LOCKED_EVALUATION_COMPLETED",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(args.output / "test_labels.npy", labels)
    np.save(args.output / "ce_test_probabilities.npy", ce_probabilities)
    np.save(args.output / "triplet_test_probabilities.npy", triplet_probabilities)
    write_predictions(args.output / "FINAL_TEST_PREDICTIONS.csv", test_dataset, labels, ce_probabilities, triplet_probabilities)

    bootstrap = inference["paired_bootstrap"]["macro_f1_delta"]
    mcnemar = inference["mcnemar"]
    report = [
        "# Locked Triplet-ViT 공개 test 결과",
        "",
        "- 공개 test 300장, seed 42, 5-fold softmax 확률 평균",
        "- CE-ViT와 Triplet-ViT는 동일한 초기화·fold·학습 조건",
        "- 이 결과를 근거로 Triplet hyperparameter를 다시 변경하지 않음",
        "",
        "| 모델 | Accuracy | Macro-F1 | A F1 | B F1 | C F1 | C recall | C precision | C specificity | C->A | C->B |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in (("CE-ViT", ce_metrics), ("Triplet-ViT", triplet_metrics)):
        report.append(
            f"| {name} | {pct(metrics['accuracy'])} | {pct(metrics['macro_f1'])} | "
            f"{pct(metrics['per_class_f1']['A'])} | {pct(metrics['per_class_f1']['B'])} | "
            f"{pct(metrics['per_class_f1']['C'])} | {pct(metrics['c_recall'])} | "
            f"{pct(metrics['c_precision'])} | {pct(metrics['c_specificity'])} | "
            f"{metrics['c_to_a']} | {metrics['c_to_b']} |"
        )
    report.extend([
        "",
        "## Triplet - CE 변화",
        "",
        f"- Accuracy: {100 * delta['accuracy']:+.2f} pp",
        f"- Macro-F1: {100 * delta['macro_f1']:+.2f} pp",
        f"- C recall: {100 * delta['c_recall']:+.2f} pp",
        f"- C specificity: {100 * delta['c_specificity']:+.2f} pp",
        f"- C->A: {int(delta['c_to_a']):+d}장",
        f"- C->B: {int(delta['c_to_b']):+d}장",
        "",
        "## 대응 통계",
        "",
        f"- Macro-F1 delta bootstrap 95% CI: [{100 * bootstrap['ci95_lower']:+.2f}, {100 * bootstrap['ci95_upper']:+.2f}] pp",
        f"- Bootstrap P(delta>0): {bootstrap['probability_positive']:.3f}",
        f"- McNemar exact p: {mcnemar['exact_two_sided_p']:.4f}",
        f"- CE만 정답/Triplet 오답: {mcnemar['ce_correct_triplet_wrong']}장",
        f"- CE 오답/Triplet만 정답: {mcnemar['ce_wrong_triplet_correct']}장",
    ])
    report_text = "\n".join(report)
    (args.output / "FINAL_TEST_RESULTS_KO.md").write_text(report_text, encoding="utf-8")
    completed.write_text(
        f"completed_utc={payload['protocol']['evaluated_at_utc']}\nlock_sha256={payload['protocol']['lock_sha256']}\n",
        encoding="utf-8",
    )
    print(report_text)


if __name__ == "__main__":
    main()
