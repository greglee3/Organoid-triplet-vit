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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asano_data import AsanoDataset  # noqa: E402


EFFICIENT_TRAINER = PROJECT_ROOT / "05_final_tuning" / "PROFESSOR_FEEDBACK" / "03_vit_triplet_efficientnet" / "train_efficientnet_oof.py"
CLASS_NAMES = ("A", "B", "C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single locked 50:50 ensemble public-test evaluation.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ensemble-lock", type=Path, required=True)
    parser.add_argument("--triplet-test-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=35)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args()


def load_module(path: Path, name: str):
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1, 2], zero_division=0
    )
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


def bootstrap_delta(labels: np.ndarray, reference: np.ndarray, candidate: np.ndarray, replicates: int = 10_000) -> dict:
    rng = np.random.default_rng(20260820)
    reference_pred = reference.argmax(axis=1)
    candidate_pred = candidate.argmax(axis=1)
    accuracy_delta = np.empty(replicates)
    f1_delta = np.empty(replicates)
    for replicate in range(replicates):
        selected = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[selected]
        ref = reference_pred[selected]
        cand = candidate_pred[selected]
        accuracy_delta[replicate] = np.mean(cand == sampled_labels) - np.mean(ref == sampled_labels)
        f1_delta[replicate] = f1_score(sampled_labels, cand, labels=[0, 1, 2], average="macro", zero_division=0) - f1_score(sampled_labels, ref, labels=[0, 1, 2], average="macro", zero_division=0)

    def summarize(values: np.ndarray) -> dict:
        lower, upper = np.quantile(values, [0.025, 0.975])
        return {"mean": float(values.mean()), "ci95_lower": float(lower), "ci95_upper": float(upper), "probability_positive": float(np.mean(values > 0))}

    return {"replicates": replicates, "seed": 20260820, "accuracy_delta": summarize(accuracy_delta), "macro_f1_delta": summarize(f1_delta)}


def exact_mcnemar(labels: np.ndarray, reference: np.ndarray, candidate: np.ndarray) -> dict:
    reference_correct = reference.argmax(axis=1) == labels
    candidate_correct = candidate.argmax(axis=1) == labels
    reference_only = int(np.sum(reference_correct & ~candidate_correct))
    candidate_only = int(np.sum(~reference_correct & candidate_correct))
    discordant = reference_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(reference_only, candidate_only)
        tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"reference_correct_candidate_wrong": reference_only, "reference_wrong_candidate_correct": candidate_only, "discordant_total": discordant, "exact_two_sided_p": p_value}


def validate_checkpoint(checkpoint: dict, lock: dict, expected_fold: int, path: Path) -> None:
    if int(checkpoint["fold"]) != expected_fold or int(checkpoint["seed"]) != 42:
        raise RuntimeError(f"Checkpoint fold/seed mismatch: {path}")
    if sha256(path) != lock["efficientnet"]["checkpoints"][expected_fold - 1]["sha256"]:
        raise RuntimeError(f"Checkpoint changed after lock: {path}")
    args = checkpoint["args"]
    expected = {
        "epochs": 50, "lr": 0.00015, "backbone_lr": 0.00015, "weight_decay": 0.01,
        "trainable_mode": "all", "scheduler": "step", "step_size": 3, "gamma": 0.97,
        "gridmask_probability": 0.3, "gridmask_implementation": "notebook",
        "horizontal_flip_probability": 0.5, "vertical_flip_probability": 0.5,
    }
    for key, value in expected.items():
        observed = args[key]
        matches = math.isclose(float(observed), value, rel_tol=0, abs_tol=1e-12) if isinstance(value, float) else observed == value
        if not matches:
            raise RuntimeError(f"Checkpoint setting changed for {key}: {observed!r} != {value!r}")


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels = [], []
    for images, batch_labels in loader:
        logits = model(images.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    args = parse_args()
    if not args.confirm_final_test:
        raise RuntimeError("Refusing public test without --confirm-final-test")
    completed = args.output / "FINAL_ENSEMBLE_TEST_COMPLETED.flag"
    results_path = args.output / "FINAL_ENSEMBLE_TEST_RESULTS.json"
    if completed.exists() or results_path.exists():
        raise RuntimeError("This locked ensemble was already tested; refusing to run again")

    lock = json.loads(args.ensemble_lock.read_text(encoding="utf-8"))
    if lock.get("lock_status") != "FROZEN_BEFORE_ENSEMBLE_TEST":
        raise RuntimeError("Ensemble is not frozen")
    if lock["weight_rule"] != {"triplet_vit": 0.5, "notebook_efficientnet": 0.5, "combination": "arithmetic mean of softmax probabilities"}:
        raise RuntimeError("Ensemble weights differ from the locked 50:50 rule")

    triplet_probs = np.load(args.triplet_test_root / "triplet_test_probabilities.npy")
    ce_probs = np.load(args.triplet_test_root / "ce_test_probabilities.npy")
    locked_labels = np.load(args.triplet_test_root / "test_labels.npy")
    if triplet_probs.shape != (300, 3) or ce_probs.shape != (300, 3) or locked_labels.shape != (300,):
        raise RuntimeError("Expected the existing locked 300-image Triplet/CE test arrays")

    efficient_module = load_module(EFFICIENT_TRAINER, "locked_notebook_efficient_trainer")
    checkpoint_root = Path(lock["efficientnet"]["oof_root"]) / "folds"
    checkpoints = []
    for fold in range(1, 6):
        path = checkpoint_root / f"fold_{fold}_best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, lock, fold, path)
        checkpoints.append(checkpoint)

    # Construct split='test' only after all settings and checkpoint hashes pass.
    transform = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])
    test_dataset = AsanoDataset(args.data, "test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_probabilities = []
    labels_reference = None
    for fold, checkpoint in enumerate(checkpoints, start=1):
        model = efficient_module.create_model(pretrained=False, trainable_mode="all").to(device)
        model.load_state_dict(checkpoint["model"])
        probabilities, labels = predict(model, test_loader, device)
        fold_probabilities.append(probabilities)
        if labels_reference is None:
            labels_reference = labels
        elif not np.array_equal(labels_reference, labels):
            raise RuntimeError("Test order changed between EfficientNet folds")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    efficient_probs = np.stack(fold_probabilities).mean(axis=0)
    if labels_reference is None or not np.array_equal(labels_reference, locked_labels):
        raise RuntimeError("EfficientNet test order differs from the locked Triplet test order")
    ensemble_probs = 0.5 * triplet_probs + 0.5 * efficient_probs

    model_metrics = {
        "ce_vit_reference": metrics(locked_labels, ce_probs),
        "triplet_vit": metrics(locked_labels, triplet_probs),
        "notebook_efficientnet": metrics(locked_labels, efficient_probs),
        "triplet_vit_notebook_efficientnet_50_50": metrics(locked_labels, ensemble_probs),
    }
    inference = {
        "ensemble_minus_triplet": {
            "paired_bootstrap": bootstrap_delta(locked_labels, triplet_probs, ensemble_probs),
            "mcnemar": exact_mcnemar(locked_labels, triplet_probs, ensemble_probs),
        },
        "ensemble_minus_ce_vit": {
            "paired_bootstrap": bootstrap_delta(locked_labels, ce_probs, ensemble_probs),
            "mcnemar": exact_mcnemar(locked_labels, ce_probs, ensemble_probs),
        },
    }
    payload = {
        "protocol": {
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "ensemble_lock": str(args.ensemble_lock),
            "ensemble_lock_sha256": sha256(args.ensemble_lock),
            "seed": 42, "folds": 5, "test_samples": 300,
            "combination": "0.5 * Triplet-ViT softmax + 0.5 * Notebook-EfficientNet softmax",
            "selection_used_train_only_oof": True,
            "project_public_test_was_already_seen_for_triplet_vit": True,
            "this_is_the_first_locked_test_of_this_specific_ensemble": True,
            "test_results_must_not_change_this_ensemble": True,
        },
        "models": model_metrics,
        "inference": inference,
        "test_status": "FINAL_LOCKED_ENSEMBLE_EVALUATION_COMPLETED",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(args.output / "efficient_test_probabilities.npy", efficient_probs)
    np.save(args.output / "triplet_eff_50_50_test_probabilities.npy", ensemble_probs)
    np.save(args.output / "test_labels.npy", locked_labels)

    with (args.output / "FINAL_ENSEMBLE_TEST_PREDICTIONS.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["test_index", "sample", "true_label", "triplet_prediction", "efficient_prediction", "ensemble_prediction", "ensemble_prob_A", "ensemble_prob_B", "ensemble_prob_C"])
        for index, ((sample, _), label) in enumerate(zip(test_dataset.samples, locked_labels)):
            writer.writerow([index, sample, CLASS_NAMES[int(label)], CLASS_NAMES[int(triplet_probs[index].argmax())], CLASS_NAMES[int(efficient_probs[index].argmax())], CLASS_NAMES[int(ensemble_probs[index].argmax())], *ensemble_probs[index].tolist()])

    lines = [
        "# Locked Triplet-ViT + Notebook-EfficientNet 50:50 공개 test 결과", "",
        "- 공개 test 300장; seed 42; 각 모델 5-fold 확률 평균 후 50:50 결합",
        "- 결합 비율과 두 모델의 학습 설정은 test 전에 고정됨",
        "- 프로젝트 전체로는 Triplet-ViT test 결과를 이미 확인했으므로 완전히 미공개인 test는 아니지만, 이 특정 앙상블의 첫 고정 평가임", "",
        "| 모델 | Accuracy | Macro-F1 | A F1 | B F1 | C F1 | C recall | C precision | C specificity | C→A | C→B |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in (("CE-ViT 참고", model_metrics["ce_vit_reference"]), ("Triplet-ViT", model_metrics["triplet_vit"]), ("Notebook-EfficientNet", model_metrics["notebook_efficientnet"]), ("Triplet-ViT + Notebook-EfficientNet 50:50", model_metrics["triplet_vit_notebook_efficientnet_50_50"])):
        lines.append(f"| {name} | {pct(result['accuracy'])} | {pct(result['macro_f1'])} | {pct(result['per_class_f1']['A'])} | {pct(result['per_class_f1']['B'])} | {pct(result['per_class_f1']['C'])} | {pct(result['c_recall'])} | {pct(result['c_precision'])} | {pct(result['c_specificity'])} | {result['c_to_a']} | {result['c_to_b']} |")
    ensemble = model_metrics["triplet_vit_notebook_efficientnet_50_50"]
    triplet = model_metrics["triplet_vit"]
    bootstrap = inference["ensemble_minus_triplet"]["paired_bootstrap"]["macro_f1_delta"]
    mcnemar = inference["ensemble_minus_triplet"]["mcnemar"]
    lines.extend(["", "## 50:50 앙상블 - Triplet-ViT 변화", "", f"- Accuracy: {100 * (ensemble['accuracy'] - triplet['accuracy']):+.2f} pp", f"- Macro-F1: {100 * (ensemble['macro_f1'] - triplet['macro_f1']):+.2f} pp", f"- C recall: {100 * (ensemble['c_recall'] - triplet['c_recall']):+.2f} pp", f"- C specificity: {100 * (ensemble['c_specificity'] - triplet['c_specificity']):+.2f} pp", f"- C→A: {ensemble['c_to_a'] - triplet['c_to_a']:+d}장", f"- C→B: {ensemble['c_to_b'] - triplet['c_to_b']:+d}장", "", "## 대응 통계", "", f"- Macro-F1 delta bootstrap 95% CI: [{100 * bootstrap['ci95_lower']:+.2f}, {100 * bootstrap['ci95_upper']:+.2f}] pp", f"- Bootstrap P(delta>0): {bootstrap['probability_positive']:.3f}", f"- McNemar exact p: {mcnemar['exact_two_sided_p']:.4f}"])
    report = "\n".join(lines) + "\n"
    (args.output / "FINAL_ENSEMBLE_TEST_RESULTS_KO.md").write_text(report, encoding="utf-8")
    completed.write_text(f"completed_utc={payload['protocol']['evaluated_at_utc']}\nlock_sha256={payload['protocol']['ensemble_lock_sha256']}\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
