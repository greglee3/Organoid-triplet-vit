from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asano_data import AsanoDataset  # noqa: E402
from paper_reproduction import apply_grid_mask  # noqa: E402


CLASS_NAMES = ("A", "B", "C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-only EfficientNetV2-S OOF reproduction.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=35)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="LR for unfrozen EfficientNet features; defaults to 0.1 * --lr.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--trainable-mode",
        choices=("head", "last_block", "all"),
        default="head",
        help="Train only the classifier, the last feature stage plus classifier, or all parameters.",
    )
    parser.add_argument("--gridmask-probability", type=float, default=0.7)
    parser.add_argument(
        "--gridmask-implementation",
        choices=("paper", "notebook"),
        default="paper",
        help=(
            "paper keeps the existing paper-protocol mask; notebook reproduces "
            "the public notebook's num_grid=1/2, mode=0, rotate=0 behavior."
        ),
    )
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--vertical-flip-probability", type=float, default=0.5)
    parser.add_argument(
        "--scheduler",
        choices=("none", "step"),
        default="none",
        help="Learning-rate scheduler; the public notebook uses StepLR.",
    )
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_notebook_grid_mask(image: np.ndarray, num_grid: int) -> np.ndarray:
    """Reproduce the effective GridMask behavior in eff/effv2.ipynb.

    The notebook passes floating-point ``mode`` values 0.2 and 0.3.  Its
    implementation only has special branches for exactly mode 1 or 2, so both
    choices effectively use mode 0.  It also fixes rotation to zero.
    """
    height, width = image.shape[:2]
    grid_height = height / num_grid
    grid_width = width / num_grid
    mask = np.ones(
        (int((num_grid + 1) * grid_height), int((num_grid + 1) * grid_width)),
        dtype=np.uint8,
    )
    for row in range(num_grid + 1):
        for column in range(num_grid + 1):
            y0 = int(row * grid_height)
            x0 = int(column * grid_width)
            mask[
                y0 : int(y0 + grid_height / 2),
                x0 : int(x0 + grid_width / 2),
            ] = 0
    offset_y = int(np.random.randint(max(1, int(grid_height))))
    offset_x = int(np.random.randint(max(1, int(grid_width))))
    cropped = mask[offset_y : offset_y + height, offset_x : offset_x + width]
    return image * cropped[:, :, None]


class EfficientPaperTransform:
    def __init__(
        self,
        image_size: int = 224,
        gridmask_probability: float = 0.7,
        gridmask_implementation: str = "paper",
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
    ) -> None:
        self.resize = transforms.Resize((image_size, image_size), antialias=True)
        self.to_tensor = transforms.ToTensor()
        self.gridmask_probability = gridmask_probability
        self.gridmask_implementation = gridmask_implementation
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self.gridmask_implementation == "notebook":
            # Albumentations in the notebook resizes before applying GridMask.
            array = np.asarray(self.resize(image), dtype=np.uint8).copy()
            if np.random.random() < self.gridmask_probability:
                array = apply_notebook_grid_mask(
                    array,
                    num_grid=int(np.random.choice([1, 2])),
                )
            transformed = Image.fromarray(array)
        else:
            array = np.asarray(image, dtype=np.uint8).copy()
            if np.random.random() < self.gridmask_probability:
                array = apply_grid_mask(array, mode=int(np.random.choice([0, 2])))
            transformed = self.resize(Image.fromarray(array))
        if np.random.random() < self.horizontal_flip_probability:
            transformed = ImageOps.mirror(transformed)
        if np.random.random() < self.vertical_flip_probability:
            transformed = ImageOps.flip(transformed)
        return self.to_tensor(transformed)


def create_model(pretrained: bool, trainable_mode: str = "head") -> nn.Module:
    model = timm.create_model(
        "tf_efficientnetv2_s",
        pretrained=pretrained,
        num_classes=len(CLASS_NAMES),
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    if not hasattr(model, "classifier") or not isinstance(model.classifier, nn.Linear):
        raise RuntimeError("Unexpected tf_efficientnetv2_s classifier structure")
    input_features = int(model.classifier.in_features)
    model.classifier = nn.Sequential(
        nn.Linear(input_features, 512),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(CLASS_NAMES)),
    )
    if trainable_mode == "last_block":
        if not hasattr(model, "blocks") or len(model.blocks) == 0:
            raise RuntimeError("Unexpected tf_efficientnetv2_s feature-block structure")
        for module in (model.blocks[-1], model.conv_head, model.bn2):
            for parameter in module.parameters():
                parameter.requires_grad = True
    elif trainable_mode == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif trainable_mode != "head":
        raise ValueError(f"Unsupported trainable mode: {trainable_mode}")
    return model


def create_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    head_parameters = list(model.classifier.parameters())
    head_parameter_ids = {id(parameter) for parameter in head_parameters}
    feature_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_parameter_ids
    ]
    if not feature_parameters:
        return torch.optim.AdamW(
            head_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.lr * 0.1
    return torch.optim.AdamW(
        [
            {"params": feature_parameters, "lr": backbone_lr},
            {"params": head_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    seen = 0
    true: list[int] = []
    predicted: list[int] = []
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        count = labels.size(0)
        seen += count
        loss_sum += float(loss.item()) * count
        true.extend(labels.detach().cpu().tolist())
        predicted.extend(logits.argmax(dim=1).detach().cpu().tolist())
    if seen == 0:
        raise RuntimeError("Loader produced no samples")
    return {
        "loss": loss_sum / seen,
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", labels=[0, 1, 2], zero_division=0)),
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        logits = model(images.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels_all.append(labels.numpy())
    if not probabilities:
        raise RuntimeError("Prediction loader produced no samples")
    return np.concatenate(probabilities), np.concatenate(labels_all)


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    c_tp = int(matrix[2, 2])
    c_fp = int(matrix[:2, 2].sum())
    c_tn = int(matrix[:2, :2].sum())
    c_precision = c_tp / (c_tp + c_fp) if c_tp + c_fp else 0.0
    c_total = int(matrix[2, :].sum())
    c_recall = c_tp / c_total if c_total else 0.0
    try:
        auc_values = roc_auc_score(
            labels,
            probabilities,
            multi_class="ovr",
            average=None,
            labels=[0, 1, 2],
        ).tolist()
    except ValueError:
        # Smoke tests may intentionally evaluate only the first few samples.
        auc_values = [float("nan")] * len(CLASS_NAMES)
    return {
        "evaluated_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", labels=[0, 1, 2], zero_division=0)),
        "per_class_f1": dict(zip(CLASS_NAMES, f1_score(labels, predictions, average=None, labels=[0, 1, 2], zero_division=0).tolist())),
        "c_recall": c_recall,
        "c_precision": c_precision,
        "c_specificity": c_tn / (c_tn + c_fp) if c_tn + c_fp else 0.0,
        "c_to_a": int(matrix[2, 0]),
        "c_to_b": int(matrix[2, 1]),
        "false_c": c_fp,
        "roc_auc_ovr": dict(zip(CLASS_NAMES, auc_values)),
        "confusion_matrix": matrix.tolist(),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    set_deterministic(args.seed)
    evaluation_transform = transforms.Compose(
        [transforms.Resize((224, 224), antialias=True), transforms.ToTensor()]
    )
    training_transform = EfficientPaperTransform(
        224,
        args.gridmask_probability,
        args.gridmask_implementation,
        args.horizontal_flip_probability,
        args.vertical_flip_probability,
    )
    # Deliberately never constructs split='test'.
    train_dataset = AsanoDataset(args.data, "train", transform=training_transform)
    validation_dataset = AsanoDataset(args.data, "train", transform=evaluation_transform)
    fold_manifest = json.loads(args.folds_json.read_text(encoding="utf-8"))
    if int(fold_manifest["seed"]) != args.seed:
        raise ValueError("fold manifest seed does not match --seed")
    manifest_samples = fold_manifest["ordered_samples"]
    if len(manifest_samples) != len(train_dataset):
        raise ValueError("fold manifest sample count mismatch")
    for index, ((sample, label), manifest) in enumerate(zip(train_dataset.samples, manifest_samples)):
        if index != int(manifest["index"]) or sample != manifest["sample"] or int(label) != int(manifest["label"]):
            raise ValueError(f"fold manifest sample mismatch at index {index}")
    splits = [
        (
            np.asarray(fold["train_indices"], dtype=int),
            np.asarray(fold["validation_indices"], dtype=int),
        )
        for fold in fold_manifest["folds"]
    ]
    if len(splits) < 2:
        raise ValueError("Expected at least two folds")

    device = resolve_device(args.device)
    model = create_model(
        pretrained=not args.no_pretrained,
        trainable_mode=args.trainable_mode,
    ).to(device)
    initial_state = copy.deepcopy(model.state_dict())
    oof_probabilities = np.full((len(train_dataset), 3), np.nan, dtype=np.float32)
    evaluated = np.zeros(len(train_dataset), dtype=bool)
    history: list[dict] = []
    fold_summaries: list[dict] = []
    folds_dir = args.output / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"device={device} model=tf_efficientnetv2_s seed={args.seed} folds={len(splits)} "
        f"epochs={args.epochs} head_lr={args.lr:g} "
        f"backbone_lr={args.backbone_lr if args.backbone_lr is not None else args.lr * 0.1:g} "
        f"wd={args.weight_decay:g} trainable={args.trainable_mode} "
        f"scheduler={args.scheduler} gridmask={args.gridmask_implementation} "
        f"train_only={len(train_dataset)}"
    )
    for fold_number, (train_indices, validation_indices) in enumerate(splits, start=1):
        if args.max_folds is not None and fold_number > args.max_folds:
            break
        set_deterministic(args.seed + fold_number * 10_000)
        model.load_state_dict(initial_state)
        generator = torch.Generator().manual_seed(args.seed + fold_number * 10_000)
        train_loader = DataLoader(
            Subset(train_dataset, train_indices.tolist()),
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.workers,
        )
        validation_loader = DataLoader(
            Subset(validation_dataset, validation_indices.tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        optimizer = create_optimizer(model, args)
        scheduler = (
            StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
            if args.scheduler == "step"
            else None
        )
        best_loss = math.inf
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
            train_metrics = run_epoch(model, train_loader, device, optimizer, args.max_batches)
            validation_metrics = run_epoch(model, validation_loader, device, None, args.max_batches)
            row = {
                "seed": args.seed,
                "fold": fold_number,
                "epoch": epoch,
                "seconds": time.perf_counter() - started,
                "learning_rates": ";".join(f"{value:.12g}" for value in learning_rates),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in validation_metrics.items()},
            }
            history.append(row)
            print(
                f"seed={args.seed} fold={fold_number}/{len(splits)} epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.4f} val_loss={validation_metrics['loss']:.4f} "
                f"val_acc={validation_metrics['accuracy']:.3f} val_f1={validation_metrics['macro_f1']:.3f} "
                f"time={row['seconds']:.1f}s",
                flush=True,
            )
            if scheduler is not None:
                scheduler.step()
            if float(validation_metrics["loss"]) < best_loss:
                best_loss = float(validation_metrics["loss"])
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_state is None:
            raise RuntimeError(f"fold {fold_number} produced no checkpoint")
        model.load_state_dict(best_state)
        probabilities, labels = predict(model, validation_loader, device, args.max_batches)
        assigned_indices = validation_indices[: len(labels)]
        expected_labels = np.asarray(train_dataset.targets, dtype=int)[assigned_indices]
        if not np.array_equal(expected_labels, labels):
            raise RuntimeError("Validation labels did not align with fold indices")
        oof_probabilities[assigned_indices] = probabilities
        evaluated[assigned_indices] = True
        fold_metrics = calculate_metrics(labels, probabilities)
        fold_summaries.append(
            {
                "fold": fold_number,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                **fold_metrics,
            }
        )
        torch.save(
            {
                "model": best_state,
                "fold": fold_number,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "train_indices": train_indices.tolist(),
                "validation_indices": validation_indices.tolist(),
                "args": vars(args),
            },
            folds_dir / f"fold_{fold_number}_best.pt",
        )
        del optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not evaluated.any():
        raise RuntimeError("No OOF samples evaluated")
    evaluated_indices = np.flatnonzero(evaluated)
    labels = np.asarray(train_dataset.targets, dtype=int)[evaluated_indices]
    metrics = calculate_metrics(labels, oof_probabilities[evaluated_indices])
    summary = {
        "model": "tf_efficientnetv2_s",
        "pretrained": not args.no_pretrained,
        "trainable": args.trainable_mode,
        "optimizer": "AdamW",
        "lr": args.lr,
        "head_lr": args.lr,
        "backbone_lr": (
            args.backbone_lr if args.backbone_lr is not None else args.lr * 0.1
        ),
        "weight_decay": args.weight_decay,
        "scheduler": {
            "name": args.scheduler,
            "step_size": args.step_size if args.scheduler == "step" else None,
            "gamma": args.gamma if args.scheduler == "step" else None,
        },
        "epochs": args.epochs,
        "folds": len(splits),
        "augmentation": {
            "gridmask_probability": args.gridmask_probability,
            "gridmask_implementation": args.gridmask_implementation,
            "horizontal_flip_probability": args.horizontal_flip_probability,
            "vertical_flip_probability": args.vertical_flip_probability,
        },
        "checkpoint_rule": "minimum validation cross-entropy",
        "fold_summaries": fold_summaries,
        "test_status": "NOT_ACCESSED",
        **metrics,
    }
    write_json(args.output / "oof_metrics.json", summary)
    np.save(args.output / "oof_probabilities.npy", oof_probabilities)
    with (args.output / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["train_index", "sample", "true_label", "predicted_label", "prob_A", "prob_B", "prob_C"])
        for index in evaluated_indices:
            probabilities = oof_probabilities[index]
            predicted = int(probabilities.argmax())
            sample, label = train_dataset.samples[index]
            writer.writerow([index, sample, CLASS_NAMES[int(label)], CLASS_NAMES[predicted], *probabilities.tolist()])
    if history:
        with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    print(
        f"efficientnet OOF seed={args.seed} evaluated={metrics['evaluated_samples']} "
        f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"c_recall={metrics['c_recall']:.4f} c_specificity={metrics['c_specificity']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
