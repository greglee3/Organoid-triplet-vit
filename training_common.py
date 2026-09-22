from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GridMask:
    """Simple GridMask implementation applied to a C x H x W float tensor."""

    def __init__(self, probability: float = 0.7) -> None:
        self.probability = probability

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.probability:
            return image

        _, height, width = image.shape
        spacing = random.randint(max(8, min(height, width) // 12), max(9, min(height, width) // 5))
        hole = max(1, round(spacing * random.uniform(0.35, 0.55)))
        offset_y = random.randint(0, spacing - 1)
        offset_x = random.randint(0, spacing - 1)
        mask = torch.ones((height, width), dtype=image.dtype, device=image.device)
        for y in range(offset_y - spacing, height, spacing):
            for x in range(offset_x - spacing, width, spacing):
                y0, y1 = max(0, y), min(height, y + hole)
                x0, x1 = max(0, x), min(width, x + hole)
                if y0 < y1 and x0 < x1:
                    mask[y0:y1, x0:x1] = 0
        return image * mask.unsqueeze(0)


def build_transforms(image_size: int = 224):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomApply([transforms.AugMix(severity=3)], p=0.7),
            transforms.ToTensor(),
            GridMask(probability=0.7),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_vit_classifier(num_classes: int, weights: str) -> nn.Module:
    selected_weights = (
        ViT_B_16_Weights.IMAGENET1K_V1 if weights == "imagenet" else None
    )
    model = vit_b_16(weights=selected_weights)
    input_features = model.heads.head.in_features
    model.heads.head = nn.Linear(input_features, num_classes)
    return model


class ViTMetricModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        weights: str,
    ) -> None:
        super().__init__()
        selected_weights = (
            ViT_B_16_Weights.IMAGENET1K_V1 if weights == "imagenet" else None
        )
        self.backbone = vit_b_16(weights=selected_weights)
        feature_dim = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images: torch.Tensor):
        features = self.backbone(images)
        embeddings = torch.nn.functional.normalize(
            self.projector(features), dim=1
        )
        logits = self.classifier(embeddings)
        return embeddings, logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classification_metrics(
    labels: list[int],
    predictions: list[int],
    num_classes: int,
) -> dict[str, Any]:
    all_classes = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=all_classes,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=all_classes,
        ).tolist(),
    }


def save_confusion_matrix(
    matrix: list[list[int]],
    class_names: list[str],
    output_path: Path,
) -> None:
    array = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(array, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Test confusion matrix",
    )
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            axis.text(
                column,
                row,
                str(array[row, column]),
                ha="center",
                va="center",
                color="white" if array[row, column] > array.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
