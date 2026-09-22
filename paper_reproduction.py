from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from asano_data import AsanoDataset
from training_common import save_confusion_matrix, write_json


CLASS_NAMES = ["A", "B", "C"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-faithful 5-fold Asano ViT reproduction."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/paper_reproduction"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augmentation",
        choices=["paper", "none"],
        default="paper",
        help="paper activates the intended GridMask and AugMix pipeline",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights (smoke tests only)",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Limit folds for a smoke test",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit every loader for a smoke test",
    )
    return parser.parse_args()


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def int_parameter(level: float, max_value: float) -> int:
    return int(level * max_value / 10)


def float_parameter(level: float, max_value: float) -> float:
    return float(level) * max_value / 10.0


def sample_level(level: float) -> float:
    return float(np.random.uniform(low=0.1, high=level))


def autocontrast(image: Image.Image, _: float) -> Image.Image:
    return ImageOps.autocontrast(image)


def equalize(image: Image.Image, _: float) -> Image.Image:
    return ImageOps.equalize(image)


def posterize(image: Image.Image, level: float) -> Image.Image:
    bits = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(image, 4 - bits)


def rotate(image: Image.Image, level: float) -> Image.Image:
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return image.rotate(degrees, resample=Image.Resampling.BILINEAR)


def solarize(image: Image.Image, level: float) -> Image.Image:
    threshold = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(image, 256 - threshold)


def shear_x(image: Image.Image, level: float) -> Image.Image:
    amount = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        amount = -amount
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, amount, 0, 0, 1, 0),
        resample=Image.Resampling.BILINEAR,
    )


def shear_y(image: Image.Image, level: float) -> Image.Image:
    amount = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        amount = -amount
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, 0, 0, amount, 1, 0),
        resample=Image.Resampling.BILINEAR,
    )


def translate_x(image: Image.Image, level: float) -> Image.Image:
    amount = int_parameter(sample_level(level), image.size[0] / 3)
    if np.random.random() > 0.5:
        amount = -amount
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, 0, amount, 0, 1, 0),
        resample=Image.Resampling.BILINEAR,
    )


def translate_y(image: Image.Image, level: float) -> Image.Image:
    amount = int_parameter(sample_level(level), image.size[0] / 3)
    if np.random.random() > 0.5:
        amount = -amount
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, 0, 0, 0, 1, amount),
        resample=Image.Resampling.BILINEAR,
    )


AUGMIX_OPERATIONS = [
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
]


def augment_and_mix(
    image: np.ndarray,
    severity: int = 4,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
) -> np.ndarray:
    weights = np.float32(np.random.dirichlet([alpha] * width))
    mixture_weight = np.float32(np.random.beta(alpha, alpha))
    mixture = np.zeros_like(image, dtype=np.float32)

    sampled_depth = depth if depth > 0 else int(np.random.randint(1, 4))
    for branch_index in range(width):
        augmented = image.copy()
        for _ in range(sampled_depth):
            operation = np.random.choice(AUGMIX_OPERATIONS)
            augmented_pil = Image.fromarray(augmented.astype(np.uint8))
            augmented = np.asarray(operation(augmented_pil, severity))
        mixture += weights[branch_index] * augmented

    mixed = (1.0 - mixture_weight) * image + mixture_weight * mixture
    return np.clip(mixed, 0, 255).astype(np.uint8)


def apply_grid_mask(image: np.ndarray, mode: int) -> np.ndarray:
    height, width = image.shape[:2]
    num_grid = 3
    grid_height = height / num_grid
    grid_width = width / num_grid
    mask_height = int((num_grid + 1) * grid_height)
    mask_width = int((num_grid + 1) * grid_width)
    mask = np.ones((mask_height, mask_width), dtype=np.uint8)

    for row in range(num_grid + 1):
        for column in range(num_grid + 1):
            y0 = int(row * grid_height)
            x0 = int(column * grid_width)
            mask[y0 : int(y0 + grid_height / 2), x0 : int(x0 + grid_width / 2)] = 0
            if mode == 2:
                mask[
                    int(y0 + grid_height / 2) : int(y0 + grid_height),
                    int(x0 + grid_width / 2) : int(x0 + grid_width),
                ] = 0

    angle = int(np.random.randint(-15, 15))
    mask_image = Image.fromarray(mask * 255)
    mask = (
        np.asarray(
            mask_image.rotate(
                angle,
                resample=Image.Resampling.NEAREST,
                fillcolor=255,
            )
        )
        > 127
    ).astype(np.uint8)
    max_y = max(1, int(grid_height))
    max_x = max(1, int(grid_width))
    offset_y = int(np.random.randint(max_y))
    offset_x = int(np.random.randint(max_x))
    cropped = mask[offset_y : offset_y + height, offset_x : offset_x + width]
    return image * cropped[:, :, None]


class PaperTrainTransform:
    """Activate the GridMask/AugMix pipeline described in the paper."""

    def __init__(self, image_size: int = 224) -> None:
        self.resize = transforms.Resize(
            (image_size, image_size),
            antialias=True,
        )
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.uint8).copy()
        if np.random.random() < 0.7:
            array = apply_grid_mask(array, mode=int(np.random.choice([0, 2])))
        if np.random.random() < 0.7:
            array = augment_and_mix(
                array,
                severity=4,
                width=3,
                depth=-1,
                alpha=1.0,
            )
        return self.to_tensor(self.resize(Image.fromarray(array)))


def build_transforms(augmentation: str):
    evaluation = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
        ]
    )
    training = PaperTrainTransform(224) if augmentation == "paper" else evaluation
    return training, evaluation


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    seen = 0
    labels_all: list[int] = []
    predictions_all: list[int] = []

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        seen += batch_size
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())

    if seen == 0:
        raise RuntimeError("The loader produced no batches")
    return {
        "loss": total_loss / seen,
        "accuracy": float(accuracy_score(labels_all, predictions_all)),
        "macro_f1": float(
            f1_score(
                labels_all,
                predictions_all,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
    }


@torch.no_grad()
def predict_probabilities(
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
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(probabilities), np.concatenate(labels_all)


def specificity_per_class(matrix: np.ndarray) -> list[float]:
    total = int(matrix.sum())
    values: list[float] = []
    for class_index in range(matrix.shape[0]):
        true_positive = int(matrix[class_index, class_index])
        false_positive = int(matrix[:, class_index].sum() - true_positive)
        false_negative = int(matrix[class_index, :].sum() - true_positive)
        true_negative = total - true_positive - false_positive - false_negative
        values.append(true_negative / max(1, true_negative + false_positive))
    return values


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "folds").mkdir(exist_ok=True)
    write_json(
        args.output / "config.json",
        {
            **vars(args),
            "method_interpretation": (
                "Official notebook architecture/optimizer/scheduler/folds. "
                + (
                    "The paper-intended GridMask and AugMix pipeline is activated. "
                    if args.augmentation == "paper"
                    else "The public notebook's executed Resize+ToTensor transform "
                    "is used without augmentation. "
                )
                + "No ImageNet mean/std normalization."
            ),
            "known_source_ambiguity": (
                "The public notebook overwrites train_transformer with "
                "Resize+ToTensor after defining GridMask/AugMix. This run "
                "activates augmentation because the paper states it was used."
            ),
        },
    )

    train_transform, evaluation_transform = build_transforms(args.augmentation)
    train_dataset = AsanoDataset(args.data, "train", transform=train_transform)
    validation_dataset = AsanoDataset(
        args.data,
        "train",
        transform=evaluation_transform,
    )
    test_dataset = AsanoDataset(
        args.data,
        "test",
        transform=evaluation_transform,
    )

    split_generator = KFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    all_indices = np.arange(len(train_dataset))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_options = {
        "num_workers": args.workers,
    }
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )

    print(
        f"device={device} model=vit_small_patch16_224 "
        f"train={len(train_dataset)} test={len(test_dataset)} "
        f"folds={args.folds} batch={args.batch_size} "
        f"augmentation={args.augmentation}"
    )

    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=not args.no_pretrained,
        num_classes=3,
    )
    model.to(device)
    initial_state = copy.deepcopy(model.state_dict())

    criterion = nn.CrossEntropyLoss()
    histories: list[dict] = []
    fold_probabilities: list[np.ndarray] = []
    test_labels_reference: np.ndarray | None = None

    for fold_index, (train_indices, validation_indices) in enumerate(
        split_generator.split(all_indices),
        start=1,
    ):
        if args.max_folds is not None and fold_index > args.max_folds:
            break
        fold_started = time.perf_counter()
        model.load_state_dict(initial_state)
        train_loader = DataLoader(
            Subset(train_dataset, train_indices.tolist()),
            batch_size=args.batch_size,
            shuffle=True,
            **loader_options,
        )
        validation_loader = DataLoader(
            Subset(validation_dataset, validation_indices.tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            **loader_options,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        best_validation_loss = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                args.max_batches,
            )
            scheduler.step()
            validation_metrics = run_epoch(
                model,
                validation_loader,
                criterion,
                device,
                None,
                args.max_batches,
            )
            row = {
                "fold": fold_index,
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - epoch_started,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": validation_metrics["loss"],
                "val_accuracy": validation_metrics["accuracy"],
                "val_macro_f1": validation_metrics["macro_f1"],
            }
            histories.append(row)
            print(
                f"fold={fold_index}/{args.folds} epoch={epoch:02d} "
                f"lr={row['lr']:.3e} train_loss={row['train_loss']:.4f} "
                f"train_acc={row['train_accuracy']:.3f} "
                f"val_loss={row['val_loss']:.4f} "
                f"val_acc={row['val_accuracy']:.3f} "
                f"time={row['seconds']:.1f}s"
            )

            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        if best_state is None:
            raise RuntimeError(f"Fold {fold_index} never produced a checkpoint")
        model.load_state_dict(best_state)
        checkpoint_path = args.output / "folds" / f"fold_{fold_index}_best.pt"
        torch.save(
            {
                "model": best_state,
                "fold": fold_index,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "train_indices": train_indices.tolist(),
                "validation_indices": validation_indices.tolist(),
                "class_names": CLASS_NAMES,
                "args": vars(args),
            },
            checkpoint_path,
        )

        fold_probs, fold_test_labels = predict_probabilities(
            model,
            test_loader,
            device,
            args.max_batches,
        )
        fold_probabilities.append(fold_probs)
        if test_labels_reference is None:
            test_labels_reference = fold_test_labels
        elif not np.array_equal(test_labels_reference, fold_test_labels):
            raise RuntimeError("Test label order changed between folds")

        print(
            f"fold={fold_index} best_epoch={best_epoch} "
            f"best_val_loss={best_validation_loss:.4f} "
            f"fold_time={time.perf_counter() - fold_started:.1f}s"
        )
        del optimizer, scheduler
        torch.cuda.empty_cache()

    if not fold_probabilities or test_labels_reference is None:
        raise RuntimeError("No folds were trained")

    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=histories[0].keys())
        writer.writeheader()
        writer.writerows(histories)

    stacked_probabilities = np.stack(fold_probabilities, axis=0)
    ensemble_probabilities = stacked_probabilities.mean(axis=0)
    predictions = ensemble_probabilities.argmax(axis=1)
    matrix = confusion_matrix(
        test_labels_reference,
        predictions,
        labels=[0, 1, 2],
    )
    accuracy = float(accuracy_score(test_labels_reference, predictions))
    macro_f1 = float(
        f1_score(
            test_labels_reference,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )
    per_class_f1 = f1_score(
        test_labels_reference,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    ).tolist()
    sensitivity = (
        np.diag(matrix) / np.maximum(1, matrix.sum(axis=1))
    ).tolist()
    specificity = specificity_per_class(matrix)
    try:
        auc = roc_auc_score(
            test_labels_reference,
            ensemble_probabilities,
            multi_class="ovr",
            average=None,
            labels=[0, 1, 2],
        ).tolist()
    except ValueError:
        auc = [float("nan")] * 3

    summary = {
        "trained_folds": len(fold_probabilities),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(CLASS_NAMES, per_class_f1)),
        "sensitivity": dict(zip(CLASS_NAMES, sensitivity)),
        "specificity": dict(zip(CLASS_NAMES, specificity)),
        "roc_auc_ovr": dict(zip(CLASS_NAMES, auc)),
        "confusion_matrix": matrix.tolist(),
    }
    write_json(args.output / "test_metrics.json", summary)
    np.save(args.output / "fold_probabilities.npy", stacked_probabilities)
    np.save(args.output / "ensemble_probabilities.npy", ensemble_probabilities)
    save_confusion_matrix(
        matrix.tolist(),
        CLASS_NAMES,
        args.output / "test_confusion_matrix.png",
    )

    with (args.output / "test_predictions.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["sample", "true_label", "predicted_label", "prob_A", "prob_B", "prob_C"]
        )
        for sample, true_label, predicted_label, probabilities in zip(
            test_dataset.samples,
            test_labels_reference,
            predictions,
            ensemble_probabilities,
        ):
            writer.writerow(
                [
                    sample[0],
                    CLASS_NAMES[int(true_label)],
                    CLASS_NAMES[int(predicted_label)],
                    *[float(value) for value in probabilities],
                ]
            )

    print(
        f"paper_reproduction test_accuracy={accuracy:.3f} "
        f"macro_f1={macro_f1:.3f} confusion={matrix.tolist()}"
    )


if __name__ == "__main__":
    main()
