from __future__ import annotations

import torch
from torch import nn


class LinearMetricProjection(nn.Module):
    """Learn a metric subspace without changing the CE classifier features."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Projection dimensions must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class MLPMetricProjection(nn.Module):
    """Two-layer metric head used only by Triplet loss.

    The hidden width stays equal to the ViT feature dimension so the only
    screened architectural choice is the final 128/256-D metric space.
    Classification continues to use the unmodified ViT pre-logits.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Projection dimensions must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim, bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)
