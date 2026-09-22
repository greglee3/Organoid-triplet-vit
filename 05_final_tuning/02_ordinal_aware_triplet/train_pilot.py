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
import torch.nn.functional as functional
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asano_data import AsanoDataset  # noqa: E402
from paper_reproduction import PaperTrainTransform  # noqa: E402
from projection_head import (  # noqa: E402
    LinearMetricProjection,
    MLPMetricProjection,
)
from training_common import write_json  # noqa: E402
from triplet_schedule import scheduled_triplet_lambda  # noqa: E402


CLASS_NAMES = ["A", "B", "C"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only pilot for CE + configurable Triplet mining on the "
            "shared ViT pre-logits representation. Equal adjacent/far "
            "margins run a class-agnostic uniform-margin ablation. "
            "This script never opens the test split."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0,0.005",
        help="Comma-separated Triplet weights; 0 is the internal CE baseline.",
    )
    parser.add_argument(
        "--ordinal-lambda",
        type=float,
        default=0.0,
        help=(
            "Weight for the Stage-2 expected ordinal-cost term. "
            "Use 0 for CE + Triplet and 0.30 for the fixed Stage-2 choice."
        ),
    )
    parser.add_argument(
        "--reference-baseline-metrics",
        type=Path,
        default=None,
        help=(
            "Optional lambda=0 oof_metrics.json from an identical protocol. "
            "This permits incremental positive-lambda runs without retraining "
            "the already verified CE baseline."
        ),
    )
    parser.add_argument("--adjacent-margin", type=float, default=0.05)
    parser.add_argument("--far-margin", type=float, default=0.15)
    parser.add_argument(
        "--mining-mode",
        choices=["semi-hard", "batch-all", "distance-weighted"],
        default="semi-hard",
        help="Triplet relation selection strategy.",
    )
    parser.add_argument(
        "--distance-weighted-cutoff",
        type=float,
        default=0.5,
        help=(
            "Minimum Euclidean distance used in density correction for "
            "distance-weighted mining."
        ),
    )
    parser.add_argument(
        "--distance-weighted-nonzero-cutoff",
        type=float,
        default=1.4,
        help=(
            "Maximum Euclidean negative distance considered by "
            "distance-weighted mining."
        ),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--classes-per-batch", type=int, default=3)
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head-lr-multiplier", type=float, default=10.0)
    parser.add_argument(
        "--embedding-head",
        choices=["shared", "linear-projection", "mlp-projection"],
        default="shared",
        help=(
            "shared applies Triplet directly to ViT pre-logits and exactly "
            "preserves completed runs. linear-projection gives Triplet a "
            "separate learned linear subspace. mlp-projection uses a two-layer "
            "GELU projection. CE always keeps the ViT pre-logits."
        ),
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=128,
        help="Metric dimension for a linear or MLP projection head.",
    )
    parser.add_argument(
        "--distance-metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Pairwise distance used by semi-hard Triplet mining.",
    )
    parser.add_argument(
        "--embedding-normalization",
        choices=["l2", "none"],
        default="l2",
        help=(
            "Normalize metric embeddings before mining. cosine requires l2; "
            "euclidean supports both l2 and raw embeddings."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw"],
        default="adam",
        help=(
            "Optimizer family. The default preserves all completed runs; "
            "Student Trial weight-decay searches use AdamW."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "constant", "multistep"],
        default="cosine",
        help=(
            "Learning-rate schedule after optional linear warm-up. "
            "The default cosine exactly preserves completed runs."
        ),
    )
    parser.add_argument(
        "--scheduler-milestones",
        type=str,
        default="10,15",
        help="Comma-separated decay epochs used by the multistep scheduler.",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.1,
        help="Multiplicative LR decay at every multistep milestone.",
    )
    parser.add_argument(
        "--triplet-ce-only-epochs",
        type=int,
        default=0,
        help=(
            "Number of initial epochs trained with CE/ordinal losses only. "
            "The default 0 exactly preserves completed Triplet runs."
        ),
    )
    parser.add_argument(
        "--triplet-ramp-epochs",
        type=int,
        default=0,
        help=(
            "Number of epochs after the CE-only phase used to linearly "
            "increase the effective Triplet lambda to its target value. "
            "Zero applies the target lambda immediately after CE-only epochs."
        ),
    )
    parser.add_argument(
        "--augmentation",
        choices=["none", "paper"],
        default="none",
        help=(
            "paper activates the intended GridMask+AugMix training pipeline; "
            "validation is always Resize+ToTensor only"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-c-specificity", type=float, default=0.90)
    parser.add_argument("--max-c-recall-drop", type=float, default=0.02)
    parser.add_argument("--max-c-precision-drop", type=float, default=0.03)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser.parse_args()


def parse_lambdas(raw: str) -> list[float]:
    values = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("At least one lambda is required")
    if any(value < 0 for value in values):
        raise ValueError("Triplet lambda must be non-negative")
    return values


def lambda_key(value: float) -> str:
    return f"lambda_{value:g}".replace(".", "p")


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
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PKBatchSampler(Sampler[list[int]]):
    """Deterministic P x K sampler over positions in a training Subset."""

    def __init__(
        self,
        labels: np.ndarray,
        classes_per_batch: int,
        samples_per_class: int,
        seed: int,
    ) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.classes = np.unique(self.labels)
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.epoch = 0
        if classes_per_batch != len(self.classes):
            raise ValueError(
                "This pilot requires every class in every P x K batch"
            )
        if any(
            int((self.labels == class_index).sum()) < samples_per_class
            for class_index in self.classes
        ):
            raise ValueError("Not enough samples to construct a P x K batch")
        self.batch_size = classes_per_batch * samples_per_class
        self.batch_count = math.ceil(len(self.labels) / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        class_positions = {
            class_index: np.flatnonzero(self.labels == class_index)
            for class_index in self.classes
        }
        for _ in range(self.batch_count):
            batch: list[int] = []
            for class_index in self.classes:
                selected = rng.choice(
                    class_positions[class_index],
                    size=self.samples_per_class,
                    replace=False,
                )
                batch.extend(int(value) for value in selected)
            rng.shuffle(batch)
            yield batch


def cosine_lr_factor(epoch: int, epochs: int, warmup_epochs: int) -> float:
    if epoch <= warmup_epochs:
        return epoch / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def parse_scheduler_milestones(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("At least one scheduler milestone is required")
    if any(value < 1 for value in values):
        raise ValueError("Scheduler milestones must be positive epochs")
    return values


def scheduler_lr_factor(
    epoch: int,
    epochs: int,
    warmup_epochs: int,
    scheduler: str,
    milestones: list[int],
    gamma: float,
) -> float:
    """Return a deterministic epoch-level LR multiplier.

    Linear warm-up is shared by every scheduler. The cosine branch calls the
    legacy helper so all completed default runs remain exactly reproducible.
    """

    if scheduler == "cosine":
        return cosine_lr_factor(epoch, epochs, warmup_epochs)
    if epoch <= warmup_epochs:
        return epoch / max(1, warmup_epochs)
    if scheduler == "constant":
        return 1.0
    if scheduler == "multistep":
        decays = sum(epoch >= milestone for milestone in milestones)
        return gamma**decays
    raise ValueError(f"Unsupported scheduler: {scheduler}")


def ordinal_cost_matrix(
    device: torch.device | None = None,
) -> torch.Tensor:
    """Squared ordinal distance normalized to [0, 1].

    A<->B and B<->C cost 0.25, while A<->C costs 1.0.
    """

    indices = torch.arange(3, dtype=torch.float32, device=device)
    return (indices[:, None] - indices[None, :]).abs().pow(2) / 4.0


def expected_ordinal_cost(
    logits: torch.Tensor,
    labels: torch.Tensor,
    cost_matrix: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    sample_costs = cost_matrix.index_select(0, labels)
    return (probabilities * sample_costs).sum(dim=1).mean()


class ViTSharedMetricModel(nn.Module):
    """Classify on ViT pre-logits with a shared or projected metric space."""

    def __init__(
        self,
        pretrained: bool,
        embedding_head: str = "shared",
        projection_dim: int = 128,
        embedding_normalization: str = "l2",
    ) -> None:
        super().__init__()
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=len(CLASS_NAMES),
        )
        feature_dim = int(self.vit.num_features)
        self.embedding_head = embedding_head
        self.embedding_normalization = embedding_normalization
        if embedding_head == "shared":
            self.metric_projection: nn.Module | None = None
            self.embedding_dim = feature_dim
            self.embedding_source = "shared_vit_pre_logits"
        elif embedding_head == "linear-projection":
            self.metric_projection = LinearMetricProjection(
                feature_dim,
                projection_dim,
            )
            self.embedding_dim = projection_dim
            self.embedding_source = f"linear_projection_{projection_dim}"
        elif embedding_head == "mlp-projection":
            self.metric_projection = MLPMetricProjection(
                feature_dim,
                projection_dim,
            )
            self.embedding_dim = projection_dim
            self.embedding_source = f"mlp_projection_{projection_dim}"
        else:
            raise ValueError(f"Unsupported embedding head: {embedding_head}")

    def metric_head_parameters(self) -> list[nn.Parameter]:
        if self.metric_projection is None:
            return []
        return list(self.metric_projection.parameters())

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_features = self.vit.forward_features(images)
        classifier_features = self.vit.forward_head(
            token_features,
            pre_logits=True,
        )
        logits = self.vit.head(classifier_features)
        metric_features = (
            classifier_features
            if self.metric_projection is None
            else self.metric_projection(classifier_features)
        )
        if self.embedding_normalization == "l2":
            embeddings = functional.normalize(
                metric_features,
                p=2,
                dim=1,
            )
        elif self.embedding_normalization == "none":
            embeddings = metric_features
        else:
            raise ValueError(
                "Unsupported embedding normalization: "
                f"{self.embedding_normalization}"
            )
        return logits, embeddings


def pairwise_embedding_distances(
    embeddings: torch.Tensor,
    distance_metric: str,
) -> torch.Tensor:
    """Compute the mining distance while preserving legacy cosine behavior."""

    if distance_metric == "cosine":
        return torch.clamp(1.0 - embeddings @ embeddings.T, 0.0, 2.0)
    if distance_metric == "euclidean":
        return torch.cdist(embeddings, embeddings, p=2).clamp_min(0.0)
    raise ValueError(f"Unsupported distance metric: {distance_metric}")


def ordinal_semi_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    adjacent_margin: float,
    far_margin: float,
    distance_metric: str = "cosine",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Nearest semi-hard negative per anchor-positive pair.

    A-B and B-C use the adjacent margin. A-C uses the far margin. When the
    two values are equal, every different class is treated identically and
    this becomes a standard class-agnostic semi-hard Triplet loss. Positive
    pairs for which no negative is farther than the positive are skipped,
    preventing extreme same-class outliers from dominating every anchor.
    """

    distances = pairwise_embedding_distances(embeddings, distance_metric)
    equal_labels = labels[:, None].eq(labels[None, :])
    positive_mask = equal_labels.clone()
    positive_mask.fill_diagonal_(False)
    negative_mask = ~equal_labels
    class_distance = (labels[:, None] - labels[None, :]).abs()
    margin_matrix = torch.where(
        class_distance == 2,
        torch.as_tensor(
            far_margin,
            dtype=embeddings.dtype,
            device=embeddings.device,
        ),
        torch.as_tensor(
            adjacent_margin,
            dtype=embeddings.dtype,
            device=embeddings.device,
        ),
    )

    positive_distances = distances[:, :, None]
    negative_distances = distances[:, None, :]
    margins = margin_matrix[:, None, :]
    valid_candidates = (
        positive_mask[:, :, None]
        & negative_mask[:, None, :]
        & (negative_distances > positive_distances)
        & (negative_distances < positive_distances + margins)
    )
    expanded_negative_distances = negative_distances.expand(
        -1,
        embeddings.size(0),
        -1,
    )
    candidate_distances = expanded_negative_distances.masked_fill(
        ~valid_candidates,
        float("inf"),
    )
    selected_negative, selected_indices = candidate_distances.min(dim=2)
    valid_pairs = torch.isfinite(selected_negative)
    if not valid_pairs.any():
        zero = embeddings.sum() * 0.0
        return zero, zero, zero, 0

    expanded_margins = margins.expand(
        -1,
        embeddings.size(0),
        -1,
    )
    selected_margins = expanded_margins.gather(
        2,
        selected_indices.unsqueeze(2),
    ).squeeze(2)
    selected_positive = distances[valid_pairs]
    selected_negative_values = selected_negative[valid_pairs]
    losses = (
        selected_positive
        - selected_negative_values
        + selected_margins[valid_pairs]
    )
    return (
        losses.mean(),
        selected_positive.mean(),
        selected_negative_values.mean(),
        int(valid_pairs.sum().item()),
    )


def _triplet_geometry(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    adjacent_margin: float,
    far_margin: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return cosine distances, positive/negative masks, and margins."""

    distances = torch.clamp(1.0 - embeddings @ embeddings.T, 0.0, 2.0)
    equal_labels = labels[:, None].eq(labels[None, :])
    positive_mask = equal_labels.clone()
    positive_mask.fill_diagonal_(False)
    negative_mask = ~equal_labels
    class_distance = (labels[:, None] - labels[None, :]).abs()
    margin_matrix = torch.where(
        class_distance == 2,
        torch.as_tensor(
            far_margin,
            dtype=embeddings.dtype,
            device=embeddings.device,
        ),
        torch.as_tensor(
            adjacent_margin,
            dtype=embeddings.dtype,
            device=embeddings.device,
        ),
    )
    return distances, positive_mask, negative_mask, margin_matrix


def batch_all_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    adjacent_margin: float,
    far_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Mean over every margin-violating anchor-positive-negative relation."""

    (
        distances,
        positive_mask,
        negative_mask,
        margin_matrix,
    ) = _triplet_geometry(
        embeddings,
        labels,
        adjacent_margin,
        far_margin,
    )
    positive_distances = distances[:, :, None]
    negative_distances = distances[:, None, :]
    losses = (
        positive_distances
        - negative_distances
        + margin_matrix[:, None, :]
    )
    valid_relations = (
        positive_mask[:, :, None] & negative_mask[:, None, :]
    )
    active_relations = valid_relations & (losses > 0)
    if not active_relations.any():
        zero = embeddings.sum() * 0.0
        return zero, zero, zero, 0

    expanded_positive = positive_distances.expand_as(losses)
    expanded_negative = negative_distances.expand_as(losses)
    return (
        losses[active_relations].mean(),
        expanded_positive[active_relations].mean(),
        expanded_negative[active_relations].mean(),
        int(active_relations.sum().item()),
    )


def distance_weighted_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    adjacent_margin: float,
    far_margin: float,
    distance_cutoff: float,
    nonzero_loss_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Density-corrected soft mining over active negative relations.

    The normalized-embedding Euclidean distance density is corrected using
    the hypersphere expression from distance-weighted sampling. Instead of
    drawing a stochastic negative, this implementation takes the normalized
    weighted mean over active negatives for each anchor-positive pair. The
    detached weights make mining a selector rather than an optimizable
    shortcut.
    """

    (
        cosine_distances,
        positive_mask,
        negative_mask,
        margin_matrix,
    ) = _triplet_geometry(
        embeddings,
        labels,
        adjacent_margin,
        far_margin,
    )
    euclidean_distances = torch.sqrt(
        torch.clamp(2.0 * cosine_distances, min=1e-12)
    )
    density_distances = euclidean_distances.clamp(
        min=distance_cutoff,
        max=2.0 - 1e-4,
    )
    embedding_dim = float(embeddings.size(1))
    inverse_log_density = (
        (2.0 - embedding_dim) * torch.log(density_distances)
        - ((embedding_dim - 3.0) / 2.0)
        * torch.log(
            torch.clamp(
                1.0 - 0.25 * density_distances.square(),
                min=1e-12,
            )
        )
    )

    allowed_negatives = (
        negative_mask & (euclidean_distances < nonzero_loss_cutoff)
    )
    # A fallback prevents a row from becoming all -inf in unusual batches.
    has_allowed_negative = allowed_negatives.any(dim=1, keepdim=True)
    allowed_negatives = torch.where(
        has_allowed_negative,
        allowed_negatives,
        negative_mask,
    )
    masked_log_weights = inverse_log_density.masked_fill(
        ~allowed_negatives,
        float("-inf"),
    )
    negative_weights = torch.softmax(
        masked_log_weights.detach(),
        dim=1,
    )

    positive_distances = cosine_distances[:, :, None]
    negative_distances = cosine_distances[:, None, :]
    losses = torch.relu(
        positive_distances
        - negative_distances
        + margin_matrix[:, None, :]
    )
    candidate_relations = (
        positive_mask[:, :, None] & allowed_negatives[:, None, :]
    )
    active_relations = candidate_relations & (losses > 0)
    if not active_relations.any():
        zero = embeddings.sum() * 0.0
        return zero, zero, zero, 0

    relation_weights = (
        negative_weights[:, None, :]
        * active_relations.to(embeddings.dtype)
    )
    weight_per_positive_pair = relation_weights.sum(dim=2)
    valid_positive_pairs = (
        positive_mask & (weight_per_positive_pair > 0)
    )
    weighted_loss_per_pair = (
        (losses * relation_weights).sum(dim=2)
        / weight_per_positive_pair.clamp_min(1e-12)
    )
    expanded_negative = negative_distances.expand_as(losses)
    weighted_negative_per_pair = (
        (expanded_negative * relation_weights).sum(dim=2)
        / weight_per_positive_pair.clamp_min(1e-12)
    )
    return (
        weighted_loss_per_pair[valid_positive_pairs].mean(),
        cosine_distances[valid_positive_pairs].mean(),
        weighted_negative_per_pair[valid_positive_pairs].mean(),
        int(active_relations.sum().item()),
    )


def triplet_mining_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    adjacent_margin: float,
    far_margin: float,
    mining_mode: str,
    distance_weighted_cutoff: float,
    distance_weighted_nonzero_cutoff: float,
    distance_metric: str = "cosine",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if mining_mode == "semi-hard":
        return ordinal_semi_hard_triplet_loss(
            embeddings,
            labels,
            adjacent_margin,
            far_margin,
            distance_metric,
        )
    if mining_mode == "batch-all":
        return batch_all_triplet_loss(
            embeddings,
            labels,
            adjacent_margin,
            far_margin,
        )
    if mining_mode == "distance-weighted":
        return distance_weighted_triplet_loss(
            embeddings,
            labels,
            adjacent_margin,
            far_margin,
            distance_weighted_cutoff,
            distance_weighted_nonzero_cutoff,
        )
    raise ValueError(f"Unsupported mining mode: {mining_mode}")


def run_epoch(
    model: ViTSharedMetricModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    ordinal_lambda: float,
    triplet_lambda: float,
    cost_matrix: torch.Tensor,
    adjacent_margin: float,
    far_margin: float,
    mining_mode: str,
    distance_weighted_cutoff: float,
    distance_weighted_nonzero_cutoff: float,
    distance_metric: str,
    max_batches: int | None,
) -> dict:
    is_training = optimizer is not None
    model.train(is_training)
    cross_entropy = nn.CrossEntropyLoss()
    total_sum = 0.0
    ce_sum = 0.0
    ordinal_sum = 0.0
    triplet_sum = 0.0
    positive_distance_sum = 0.0
    negative_distance_sum = 0.0
    valid_triplet_pairs = 0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    validation_embeddings: list[torch.Tensor] = []
    validation_labels: list[torch.Tensor] = []
    seen = 0

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits, embeddings = model(images)
            ce_loss = cross_entropy(logits, labels)
            ordinal_loss = expected_ordinal_cost(
                logits,
                labels,
                cost_matrix,
            )
            triplet_loss, positive_mean, negative_mean, valid_count = (
                triplet_mining_loss(
                    embeddings,
                    labels,
                    adjacent_margin,
                    far_margin,
                    mining_mode,
                    distance_weighted_cutoff,
                    distance_weighted_nonzero_cutoff,
                    distance_metric,
                )
            )
            total_loss = (
                ce_loss
                + ordinal_lambda * ordinal_loss
                + triplet_lambda * triplet_loss
            )
            if optimizer is not None:
                total_loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        seen += batch_size
        total_sum += float(total_loss.item()) * batch_size
        ce_sum += float(ce_loss.item()) * batch_size
        ordinal_sum += float(ordinal_loss.item()) * batch_size
        triplet_sum += float(triplet_loss.item()) * batch_size
        if valid_count:
            positive_distance_sum += float(positive_mean.item()) * valid_count
            negative_distance_sum += float(negative_mean.item()) * valid_count
            valid_triplet_pairs += valid_count
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
        if not is_training:
            validation_embeddings.append(embeddings.detach())
            validation_labels.append(labels.detach())

    if seen == 0:
        raise RuntimeError("The loader produced no batches")

    if not is_training:
        all_embeddings = torch.cat(validation_embeddings)
        all_labels = torch.cat(validation_labels)
        (
            global_triplet,
            global_positive,
            global_negative,
            global_valid,
        ) = triplet_mining_loss(
            all_embeddings,
            all_labels,
            adjacent_margin,
            far_margin,
            mining_mode,
            distance_weighted_cutoff,
            distance_weighted_nonzero_cutoff,
            distance_metric,
        )
        triplet_average = float(global_triplet.item())
        total_average = (
            ce_sum / seen
            + ordinal_lambda * ordinal_sum / seen
            + triplet_lambda * triplet_average
        )
        positive_average = (
            float(global_positive.item()) if global_valid else float("nan")
        )
        negative_average = (
            float(global_negative.item()) if global_valid else float("nan")
        )
        valid_triplet_pairs = global_valid
    else:
        triplet_average = triplet_sum / seen
        total_average = total_sum / seen
        positive_average = (
            positive_distance_sum / valid_triplet_pairs
            if valid_triplet_pairs
            else float("nan")
        )
        negative_average = (
            negative_distance_sum / valid_triplet_pairs
            if valid_triplet_pairs
            else float("nan")
        )

    return {
        "loss": total_average,
        "ce_loss": ce_sum / seen,
        "ordinal_loss": ordinal_sum / seen,
        "triplet_loss": triplet_average,
        "mined_positive_distance": positive_average,
        "mined_negative_distance": negative_average,
        "active_triplet_relations": valid_triplet_pairs,
        # Legacy aliases keep prior history-processing code compatible.
        "semi_hard_positive_distance": positive_average,
        "semi_hard_negative_distance": negative_average,
        "valid_triplet_pairs": valid_triplet_pairs,
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
    model: ViTSharedMetricModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    embeddings_all: list[np.ndarray] = []
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        logits, embeddings = model(images.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels_all.append(labels.numpy())
        embeddings_all.append(embeddings.cpu().numpy())
    if not probabilities:
        raise RuntimeError("The prediction loader produced no batches")
    return (
        np.concatenate(probabilities),
        np.concatenate(labels_all),
        np.concatenate(embeddings_all),
    )


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    c_true_positive = int(matrix[2, 2])
    c_false_negative = int(matrix[2, :].sum() - c_true_positive)
    c_false_positive = int(matrix[:, 2].sum() - c_true_positive)
    c_true_negative = int(
        matrix.sum()
        - c_true_positive
        - c_false_negative
        - c_false_positive
    )
    c_recall = c_true_positive / max(1, c_true_positive + c_false_negative)
    c_precision = c_true_positive / max(1, c_true_positive + c_false_positive)
    c_specificity = c_true_negative / max(1, c_true_negative + c_false_positive)
    try:
        auc = roc_auc_score(
            labels,
            probabilities,
            multi_class="ovr",
            average=None,
            labels=[0, 1, 2],
        ).tolist()
    except ValueError:
        auc = [float("nan")] * 3
    return {
        "evaluated_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "c_recall": float(c_recall),
        "c_precision": float(c_precision),
        "c_specificity": float(c_specificity),
        "c_to_a": int(matrix[2, 0]),
        "c_to_b": int(matrix[2, 1]),
        "a_to_c": int(matrix[0, 2]),
        "severe_a_c_errors": int(matrix[0, 2] + matrix[2, 0]),
        "ordinal_mae": float(np.abs(labels - predictions).mean()),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(
                labels,
                predictions,
                labels=[0, 1, 2],
                weights="quadratic",
            )
        ),
        "roc_auc_ovr": dict(zip(CLASS_NAMES, auc)),
        "confusion_matrix": matrix.tolist(),
    }


def calculate_embedding_metrics(
    labels: np.ndarray,
    embeddings: np.ndarray,
) -> dict:
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2 or np.any(counts < 2):
        return {
            "silhouette_cosine": float("nan"),
            "mean_within_class_cosine_distance": float("nan"),
            "mean_between_class_cosine_distance": float("nan"),
            "between_minus_within_distance": float("nan"),
        }

    cosine_distances = 1.0 - embeddings @ embeddings.T
    equal_labels = labels[:, None] == labels[None, :]
    diagonal = np.eye(len(labels), dtype=bool)
    within_mask = equal_labels & ~diagonal
    between_mask = ~equal_labels
    within_distance = float(cosine_distances[within_mask].mean())
    between_distance = float(cosine_distances[between_mask].mean())
    return {
        "silhouette_cosine": float(
            silhouette_score(embeddings, labels, metric="cosine")
        ),
        "mean_within_class_cosine_distance": within_distance,
        "mean_between_class_cosine_distance": between_distance,
        "between_minus_within_distance": between_distance - within_distance,
    }


def aggregate_fold_embedding_metrics(fold_rows: list[dict]) -> dict:
    metric_names = [
        "silhouette_cosine",
        "mean_within_class_cosine_distance",
        "mean_between_class_cosine_distance",
        "between_minus_within_distance",
    ]
    aggregated: dict[str, float | str | list[dict]] = {
        "method": "validation-fold-wise weighted mean",
        "per_fold": fold_rows,
    }
    for metric_name in metric_names:
        valid_rows = [
            row
            for row in fold_rows
            if np.isfinite(float(row[metric_name]))
        ]
        total_weight = sum(int(row["evaluated_samples"]) for row in valid_rows)
        aggregated[metric_name] = (
            sum(
                float(row[metric_name]) * int(row["evaluated_samples"])
                for row in valid_rows
            )
            / total_weight
            if total_weight
            else float("nan")
        )
    return aggregated


def save_oof_predictions(
    path: Path,
    samples: list[tuple[str, int]],
    probabilities: np.ndarray,
    evaluated: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "train_index",
                "sample",
                "true_label",
                "predicted_label",
                "prob_A",
                "prob_B",
                "prob_C",
            ]
        )
        for index in np.flatnonzero(evaluated):
            predicted = int(probabilities[index].argmax())
            writer.writerow(
                [
                    int(index),
                    samples[index][0],
                    CLASS_NAMES[int(samples[index][1])],
                    CLASS_NAMES[predicted],
                    *[float(value) for value in probabilities[index]],
                ]
            )


def main() -> None:
    args = parse_args()
    lambdas = parse_lambdas(args.lambdas)
    scheduler_milestones = parse_scheduler_milestones(
        args.scheduler_milestones
    )
    if args.adjacent_margin <= 0:
        raise ValueError("--adjacent-margin must be positive")
    if args.far_margin < args.adjacent_margin:
        raise ValueError(
            "--far-margin must be greater than or equal to adjacent margin"
        )
    if not (
        0.0
        < args.distance_weighted_cutoff
        < args.distance_weighted_nonzero_cutoff
        < 2.0
    ):
        raise ValueError(
            "Distance-weighted cutoffs must satisfy "
            "0 < cutoff < nonzero cutoff < 2"
        )
    if args.ordinal_lambda < 0:
        raise ValueError("--ordinal-lambda must be non-negative")
    if (
        0.0 not in lambdas
        and args.reference_baseline_metrics is None
    ):
        raise ValueError(
            "Lambda 0 or --reference-baseline-metrics is required"
        )
    if (
        args.reference_baseline_metrics is not None
        and not args.reference_baseline_metrics.is_file()
    ):
        raise FileNotFoundError(
            "Reference baseline metrics not found: "
            f"{args.reference_baseline_metrics}"
        )
    if args.classes_per_batch != len(CLASS_NAMES):
        raise ValueError("--classes-per-batch must equal 3")
    if args.samples_per_class < 2:
        raise ValueError("--samples-per-class must be at least 2")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")
    if args.projection_dim <= 0:
        raise ValueError("--projection-dim must be positive")
    if (
        args.distance_metric == "cosine"
        and args.embedding_normalization != "l2"
    ):
        raise ValueError("cosine distance requires --embedding-normalization l2")
    if args.mining_mode != "semi-hard" and args.distance_metric != "cosine":
        raise ValueError(
            "Non-cosine distance is currently supported only for semi-hard mining"
        )
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("--warmup-epochs must be in [0, epochs)")
    if not 0 < args.scheduler_gamma <= 1:
        raise ValueError("--scheduler-gamma must be in (0, 1]")
    if args.scheduler == "multistep" and any(
        value <= args.warmup_epochs or value > args.epochs
        for value in scheduler_milestones
    ):
        raise ValueError(
            "Multistep milestones must be after warm-up and within epochs"
        )
    if args.triplet_ce_only_epochs < 0:
        raise ValueError("--triplet-ce-only-epochs must be non-negative")
    if args.triplet_ramp_epochs < 0:
        raise ValueError("--triplet-ramp-epochs must be non-negative")
    if any(value > 0 for value in lambdas):
        if args.triplet_ce_only_epochs >= args.epochs:
            raise ValueError(
                "--triplet-ce-only-epochs must be smaller than --epochs"
            )
        if (
            args.triplet_ce_only_epochs + args.triplet_ramp_epochs
            > args.epochs
        ):
            raise ValueError(
                "Triplet CE-only and ramp epochs must fit within --epochs"
            )
    margin_mode = (
        "uniform"
        if math.isclose(
            args.far_margin,
            args.adjacent_margin,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        else "ordinal_aware"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "search_config.json",
        {
            **vars(args),
            "lambdas": lambdas,
            "model": "timm/vit_small_patch16_224",
            "loss": (
                "cross_entropy + lambda_ordinal * expected_ordinal_cost + "
                f"lambda_triplet * {args.mining_mode}_triplet"
            ),
            "ordinal": {
                "lambda": args.ordinal_lambda,
                "cost_matrix": ordinal_cost_matrix().tolist(),
                "cost_definition": "squared ordinal distance divided by 4",
            },
            "triplet": {
                "mining_mode": args.mining_mode,
                "margin_mode": margin_mode,
                "embedding_head": args.embedding_head,
                "projection_dim": (
                    args.projection_dim
                    if args.embedding_head != "shared"
                    else None
                ),
                "embedding": (
                    f"{args.embedding_normalization} shared ViT pre-logits"
                    if args.embedding_head == "shared"
                    else (
                        f"{args.embedding_normalization} learned "
                        f"{args.embedding_head} of ViT pre-logits"
                    )
                ),
                "distance": args.distance_metric,
                "embedding_normalization": args.embedding_normalization,
                "adjacent_margin_A_B_and_B_C": args.adjacent_margin,
                "far_margin_A_C": args.far_margin,
                "positive": "all same-class anchor-positive pairs",
                "negative_policy": {
                    "semi-hard": (
                        "nearest negative farther than the positive but "
                        "inside the margin"
                    ),
                    "batch-all": (
                        "all margin-violating negatives; average active "
                        "anchor-positive-negative relations"
                    ),
                    "distance-weighted": (
                        "hypersphere-density-corrected detached weights "
                        "over active negatives for each anchor-positive pair"
                    ),
                }[args.mining_mode],
                "distance_weighted_cutoff": (
                    args.distance_weighted_cutoff
                ),
                "distance_weighted_nonzero_cutoff": (
                    args.distance_weighted_nonzero_cutoff
                ),
                "training_mining": "within deterministic P x K mini-batch",
                "validation_mining": "within entire validation fold",
                "outlier_policy": (
                    "skip anchor-positive pairs with no active relation; "
                    "do not delete any image"
                ),
                "sampler": {
                    "P": args.classes_per_batch,
                    "K": args.samples_per_class,
                },
                "lambda_schedule": {
                    "target_lambdas": lambdas,
                    "ce_only_epochs": args.triplet_ce_only_epochs,
                    "linear_ramp_epochs": args.triplet_ramp_epochs,
                    "validation_loss_weight": (
                        "target lambda at every epoch so checkpoint tie-break "
                        "losses remain comparable across epochs"
                    ),
                },
            },
            "optimization": {
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "backbone_lr": args.lr,
                "classifier_head_lr": (
                    args.lr * args.head_lr_multiplier
                ),
                "metric_projection_lr": (
                    args.lr * args.head_lr_multiplier
                    if args.embedding_head != "shared"
                    else None
                ),
                "schedule": args.scheduler,
                "warmup_epochs": args.warmup_epochs,
                "scheduler_milestones": scheduler_milestones,
                "scheduler_gamma": args.scheduler_gamma,
                "checkpoint": "maximize validation macro-F1; tie: lower total loss",
            },
            "augmentation": {
                "training": args.augmentation,
                "paper_definition": (
                    "GridMask probability 0.7 plus AugMix probability 0.7"
                ),
                "validation": "Resize(224)+ToTensor only",
            },
            "test_split_access": "PROHIBITED: this script constructs only split='train'",
            "selection_rule": [
                "keep positive lambdas that preserve C recall/precision and C specificity",
                "maximize OOF macro-F1",
                "tie: maximize OOF embedding silhouette score",
                "tie: maximize OOF C recall",
                "final tie: choose smaller lambda",
            ],
        },
    )

    set_deterministic(args.seed)
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
        ]
    )
    training_transform = (
        PaperTrainTransform(224)
        if args.augmentation == "paper"
        else evaluation_transform
    )
    # Deliberately no AsanoDataset(..., "test", ...) in this program.
    train_dataset = AsanoDataset(
        args.data,
        "train",
        transform=training_transform,
    )
    validation_dataset = AsanoDataset(
        args.data,
        "train",
        transform=evaluation_transform,
    )
    targets = np.asarray(train_dataset.targets, dtype=np.int64)
    all_indices = np.arange(len(train_dataset))
    splitter = KFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    splits = [
        (train_indices, validation_indices)
        for train_indices, validation_indices in splitter.split(all_indices)
    ]
    write_json(
        args.output / "folds.json",
        {
            "seed": args.seed,
            "ordered_samples": [
                {"index": index, "sample": sample, "label": int(label)}
                for index, (sample, label) in enumerate(train_dataset.samples)
            ],
            "folds": [
                {
                    "fold": fold_index,
                    "train_indices": train_indices.tolist(),
                    "validation_indices": validation_indices.tolist(),
                }
                for fold_index, (train_indices, validation_indices) in enumerate(
                    splits,
                    start=1,
                )
            ],
        },
    )

    device = resolve_device(args.device)
    cost_matrix = ordinal_cost_matrix(device)
    model = ViTSharedMetricModel(
        pretrained=not args.no_pretrained,
        embedding_head=args.embedding_head,
        projection_dim=args.projection_dim,
        embedding_normalization=args.embedding_normalization,
    ).to(device)
    embedding_dim = model.embedding_dim
    embedding_source = model.embedding_source
    initial_state = copy.deepcopy(model.state_dict())
    loader_options = {"num_workers": args.workers}
    summaries: list[dict] = []

    print(
        f"device={device} train_only={len(train_dataset)} folds={args.folds} "
        f"ordinal_lambda={args.ordinal_lambda:g} "
        f"triplet_lambdas={lambdas} embedding_dim={embedding_dim} "
        f"embedding_source={embedding_source} "
        f"mining_mode={args.mining_mode} "
        f"margin_mode={margin_mode} "
        f"margins=({args.adjacent_margin:g},{args.far_margin:g}) "
        f"P={args.classes_per_batch} K={args.samples_per_class} "
        f"optimizer={args.optimizer} weight_decay={args.weight_decay:g}"
    )

    for triplet_lambda in lambdas:
        # Every lambda sees the same initialization, folds, and batch-order
        # random stream. This isolates the additional triplet-loss term.
        set_deterministic(args.seed)
        condition_dir = args.output / lambda_key(triplet_lambda)
        folds_dir = condition_dir / "folds"
        folds_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict] = []
        oof_probabilities = np.full(
            (len(train_dataset), len(CLASS_NAMES)),
            np.nan,
            dtype=np.float32,
        )
        oof_labels = np.full(len(train_dataset), -1, dtype=np.int64)
        oof_embeddings = np.full(
            (len(train_dataset), embedding_dim),
            np.nan,
            dtype=np.float32,
        )
        fold_embedding_metrics: list[dict] = []
        evaluated = np.zeros(len(train_dataset), dtype=bool)

        for fold_index, (train_indices, validation_indices) in enumerate(
            splits,
            start=1,
        ):
            if args.max_folds is not None and fold_index > args.max_folds:
                break
            model.load_state_dict(initial_state)
            train_sampler = PKBatchSampler(
                targets[train_indices],
                classes_per_batch=args.classes_per_batch,
                samples_per_class=args.samples_per_class,
                seed=args.seed + fold_index * 10_000,
            )
            train_loader = DataLoader(
                Subset(train_dataset, train_indices.tolist()),
                batch_sampler=train_sampler,
                **loader_options,
            )
            validation_loader = DataLoader(
                Subset(validation_dataset, validation_indices.tolist()),
                batch_size=(
                    args.classes_per_batch * args.samples_per_class
                ),
                shuffle=False,
                **loader_options,
            )
            head_parameters = (
                list(model.vit.head.parameters())
                + model.metric_head_parameters()
            )
            head_parameter_ids = {
                id(parameter) for parameter in head_parameters
            }
            backbone_parameters = [
                parameter
                for parameter in model.parameters()
                if id(parameter) not in head_parameter_ids
            ]
            base_lrs = [
                args.lr,
                args.lr * args.head_lr_multiplier,
            ]
            optimizer_class = (
                torch.optim.AdamW
                if args.optimizer == "adamw"
                else torch.optim.Adam
            )
            optimizer = optimizer_class(
                [
                    {"params": backbone_parameters, "lr": base_lrs[0]},
                    {"params": head_parameters, "lr": base_lrs[1]},
                ],
                weight_decay=args.weight_decay,
            )
            best_validation_loss = math.inf
            best_validation_macro_f1 = -math.inf
            best_state: dict[str, torch.Tensor] | None = None
            best_epoch = 0

            for epoch in range(1, args.epochs + 1):
                effective_triplet_lambda = scheduled_triplet_lambda(
                    triplet_lambda,
                    epoch,
                    args.triplet_ce_only_epochs,
                    args.triplet_ramp_epochs,
                )
                lr_factor = scheduler_lr_factor(
                    epoch,
                    args.epochs,
                    args.warmup_epochs,
                    args.scheduler,
                    scheduler_milestones,
                    args.scheduler_gamma,
                )
                for group, base_lr in zip(
                    optimizer.param_groups,
                    base_lrs,
                ):
                    group["lr"] = base_lr * lr_factor
                train_sampler.set_epoch(epoch)
                started = time.perf_counter()
                train_metrics = run_epoch(
                    model,
                    train_loader,
                    device,
                    optimizer,
                    args.ordinal_lambda,
                    effective_triplet_lambda,
                    cost_matrix,
                    args.adjacent_margin,
                    args.far_margin,
                    args.mining_mode,
                    args.distance_weighted_cutoff,
                    args.distance_weighted_nonzero_cutoff,
                    args.distance_metric,
                    args.max_batches,
                )
                validation_metrics = run_epoch(
                    model,
                    validation_loader,
                    device,
                    None,
                    args.ordinal_lambda,
                    triplet_lambda,
                    cost_matrix,
                    args.adjacent_margin,
                    args.far_margin,
                    args.mining_mode,
                    args.distance_weighted_cutoff,
                    args.distance_weighted_nonzero_cutoff,
                    args.distance_metric,
                    args.max_batches,
                )
                row = {
                    "lambda_ordinal": args.ordinal_lambda,
                    "lambda_triplet": triplet_lambda,
                    "effective_lambda_triplet": effective_triplet_lambda,
                    "mining_mode": args.mining_mode,
                    "margin_mode": margin_mode,
                    "adjacent_margin": args.adjacent_margin,
                    "far_margin": args.far_margin,
                    "fold": fold_index,
                    "epoch": epoch,
                    "backbone_lr": optimizer.param_groups[0]["lr"],
                    "head_lr": optimizer.param_groups[1]["lr"],
                    "seconds": time.perf_counter() - started,
                    **{
                        f"train_{key}": value
                        for key, value in train_metrics.items()
                    },
                    **{
                        f"val_{key}": value
                        for key, value in validation_metrics.items()
                    },
                }
                history.append(row)
                print(
                    f"ordinal={args.ordinal_lambda:g} "
                    f"triplet={triplet_lambda:g} "
                    f"effective_triplet={effective_triplet_lambda:g} "
                    f"fold={fold_index}/{args.folds} "
                    f"epoch={epoch:02d} train_total={train_metrics['loss']:.4f} "
                    f"val_total={validation_metrics['loss']:.4f} "
                    f"val_acc={validation_metrics['accuracy']:.3f} "
                    f"val_f1={validation_metrics['macro_f1']:.3f} "
                    f"time={row['seconds']:.1f}s"
                )
                candidate_macro_f1 = float(
                    validation_metrics["macro_f1"]
                )
                candidate_loss = float(validation_metrics["loss"])
                if (
                    candidate_macro_f1 > best_validation_macro_f1 + 1e-12
                    or (
                        abs(
                            candidate_macro_f1
                            - best_validation_macro_f1
                        )
                        <= 1e-12
                        and candidate_loss < best_validation_loss
                    )
                ):
                    best_validation_loss = validation_metrics["loss"]
                    best_validation_macro_f1 = candidate_macro_f1
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }

            if best_state is None:
                raise RuntimeError(
                    f"lambda={triplet_lambda:g} fold={fold_index} "
                    "produced no checkpoint"
                )
            model.load_state_dict(best_state)
            probabilities, fold_labels, fold_embeddings = predict_probabilities(
                model,
                validation_loader,
                device,
                args.max_batches,
            )
            assigned_indices = validation_indices[: len(fold_labels)]
            if not np.array_equal(targets[assigned_indices], fold_labels):
                raise RuntimeError("Validation label order did not match fold indices")
            oof_probabilities[assigned_indices] = probabilities
            oof_labels[assigned_indices] = fold_labels
            oof_embeddings[assigned_indices] = fold_embeddings
            fold_embedding_metrics.append(
                {
                    "fold": fold_index,
                    "evaluated_samples": int(len(fold_labels)),
                    **calculate_embedding_metrics(
                        fold_labels,
                        fold_embeddings,
                    ),
                }
            )
            evaluated[assigned_indices] = True
            torch.save(
                {
                    "model": best_state,
                    "fold": fold_index,
                    "lambda_ordinal": args.ordinal_lambda,
                    "lambda_triplet": triplet_lambda,
                    "triplet_ce_only_epochs": args.triplet_ce_only_epochs,
                    "triplet_ramp_epochs": args.triplet_ramp_epochs,
                    "mining_mode": args.mining_mode,
                    "margin_mode": margin_mode,
                    "ordinal_cost_matrix": cost_matrix.cpu(),
                    "embedding_dim": embedding_dim,
                    "embedding_head": args.embedding_head,
                    "projection_dim": (
                        args.projection_dim
                        if args.embedding_head != "shared"
                        else None
                    ),
                    "embedding_source": embedding_source,
                    "adjacent_margin": args.adjacent_margin,
                    "far_margin": args.far_margin,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "best_validation_macro_f1": (
                        best_validation_macro_f1
                    ),
                    "train_indices": train_indices.tolist(),
                    "validation_indices": validation_indices.tolist(),
                    "class_names": CLASS_NAMES,
                    "args": vars(args),
                },
                folds_dir / f"fold_{fold_index}_best.pt",
            )
            del optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not evaluated.any():
            raise RuntimeError(
                f"lambda={triplet_lambda:g} produced no OOF predictions"
            )
        evaluated_labels = oof_labels[evaluated]
        evaluated_probabilities = oof_probabilities[evaluated]
        metrics = calculate_metrics(evaluated_labels, evaluated_probabilities)
        embedding_metrics = aggregate_fold_embedding_metrics(
            fold_embedding_metrics
        )
        summary = {
            "lambda_ordinal": args.ordinal_lambda,
            "lambda_triplet": triplet_lambda,
            "mining_mode": args.mining_mode,
            "margin_mode": margin_mode,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            "scheduler_milestones": scheduler_milestones,
            "scheduler_gamma": args.scheduler_gamma,
            "augmentation": args.augmentation,
            "backbone_lr": args.lr,
            "head_lr_multiplier": args.head_lr_multiplier,
            "triplet_ce_only_epochs": args.triplet_ce_only_epochs,
            "triplet_ramp_epochs": args.triplet_ramp_epochs,
            "embedding_head": args.embedding_head,
            "distance_metric": args.distance_metric,
            "embedding_normalization": args.embedding_normalization,
            "projection_dim": (
                args.projection_dim
                if args.embedding_head != "shared"
                else None
            ),
            "embedding_dim": embedding_dim,
            "embedding_source": embedding_source,
            "adjacent_margin": args.adjacent_margin,
            "far_margin": args.far_margin,
            "embedding_metrics": embedding_metrics,
            **metrics,
        }
        summaries.append(summary)
        write_json(condition_dir / "oof_metrics.json", summary)
        np.save(condition_dir / "oof_probabilities.npy", oof_probabilities)
        np.save(condition_dir / "oof_embeddings.npy", oof_embeddings)
        save_oof_predictions(
            condition_dir / "oof_predictions.csv",
            train_dataset.samples,
            oof_probabilities,
            evaluated,
        )
        with (condition_dir / "history.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        print(
            f"lambda={triplet_lambda:g} OOF "
            f"c_recall={metrics['c_recall']:.3f} "
            f"c_specificity={metrics['c_specificity']:.3f} "
            f"c_to_a={metrics['c_to_a']} "
            f"macro_f1={metrics['macro_f1']:.3f}"
        )

    baseline = next(
        (
            item
            for item in summaries
            if float(item["lambda_triplet"]) == 0.0
        ),
        None,
    )
    baseline_source = "computed_in_this_run"
    if baseline is None:
        if args.reference_baseline_metrics is None:
            raise RuntimeError("No CE baseline is available for selection")
        with args.reference_baseline_metrics.open(
            "r",
            encoding="utf-8",
        ) as stream:
            baseline = json.load(stream)
        if float(baseline["lambda_triplet"]) != 0.0:
            raise ValueError(
                "--reference-baseline-metrics must describe lambda=0"
            )
        baseline_source = str(args.reference_baseline_metrics)
    eligible = [
        item
        for item in summaries
        if float(item["lambda_triplet"]) > 0.0
        and float(item["c_specificity"]) >= args.min_c_specificity
        and float(item["c_recall"])
        >= float(baseline["c_recall"]) - args.max_c_recall_drop
        and float(item["c_precision"])
        >= float(baseline["c_precision"]) - args.max_c_precision_drop
    ]
    ranked = sorted(
        eligible,
        key=lambda item: (
            -float(item["macro_f1"]),
            -float(item["embedding_metrics"]["silhouette_cosine"]),
            -float(item["c_recall"]),
            float(item["lambda_triplet"]),
        ),
    )

    positive_summaries = [
        item
        for item in summaries
        if float(item["lambda_triplet"]) > 0.0
    ]

    if not positive_summaries:
        selection = {
            "status": "baseline_only",
            "selected_lambda": 0.0,
            "selected_adjacent_margin": args.adjacent_margin,
            "selected_far_margin": args.far_margin,
            "selected_embedding_dim": embedding_dim,
            "embedding_source": embedding_source,
            "baseline_lambda": 0.0,
            "baseline_source": baseline_source,
            "baseline_oof_metrics": baseline,
            "eligible_positive_lambdas": [],
            "test_status": "NOT_ACCESSED",
        }
    elif ranked:
        selected = ranked[0]
        checks = {
            "macro_f1_improved": selected["macro_f1"] > baseline["macro_f1"],
            "embedding_silhouette_improved": (
                selected["embedding_metrics"]["silhouette_cosine"]
                > baseline["embedding_metrics"]["silhouette_cosine"]
            ),
            "c_recall_preserved": (
                selected["c_recall"]
                >= baseline["c_recall"] - args.max_c_recall_drop
            ),
            "c_specificity_at_least_minimum": (
                selected["c_specificity"] >= args.min_c_specificity
            ),
            "c_precision_not_excessively_lower": (
                selected["c_precision"]
                >= baseline["c_precision"] - args.max_c_precision_drop
            ),
        }
        selection = {
            "status": "selected",
            "selected_lambda": selected["lambda_triplet"],
            "selected_adjacent_margin": args.adjacent_margin,
            "selected_far_margin": args.far_margin,
            "selected_embedding_dim": embedding_dim,
            "embedding_source": embedding_source,
            "baseline_lambda": 0.0,
            "baseline_source": baseline_source,
            "baseline_oof_metrics": baseline,
            "selected_oof_metrics": selected,
            "eligible_positive_lambdas": [
                item["lambda_triplet"] for item in ranked
            ],
            "selection_rule": {
                "minimum_c_specificity": args.min_c_specificity,
                "maximum_c_recall_drop": args.max_c_recall_drop,
                "maximum_c_precision_drop": args.max_c_precision_drop,
                "primary": "maximize macro-F1",
                "tie_1": "maximize embedding silhouette score",
                "tie_2": "maximize C recall",
                "tie_3": "choose smaller lambda",
            },
            "validation_success_checks": checks,
            "validation_success_all": bool(all(checks.values())),
            "test_status": "NOT_ACCESSED",
        }
    else:
        selection = {
            "status": "no_eligible_positive_lambda",
            "selected_lambda": None,
            "selected_adjacent_margin": args.adjacent_margin,
            "selected_far_margin": args.far_margin,
            "selected_embedding_dim": embedding_dim,
            "embedding_source": embedding_source,
            "baseline_lambda": 0.0,
            "baseline_source": baseline_source,
            "baseline_oof_metrics": baseline,
            "eligible_positive_lambdas": [],
            "minimum_c_specificity": args.min_c_specificity,
            "maximum_c_recall_drop": args.max_c_recall_drop,
            "maximum_c_precision_drop": args.max_c_precision_drop,
            "test_status": "NOT_ACCESSED",
            "required_action": (
                "Review train-only OOF results. Do not evaluate the test split."
            ),
        }

    write_json(args.output / "lambda_summary.json", summaries)
    write_json(args.output / "selection.json", selection)
    print(f"selection={selection}")


if __name__ == "__main__":
    main()
