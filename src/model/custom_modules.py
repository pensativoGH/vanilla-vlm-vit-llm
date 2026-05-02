"""Shared custom modules and helper functions for the custom models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class LayerNormalization(nn.Module):
    """Custom layer normalization over the last dimension."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.gamma = nn.Parameter(torch.ones(self.model_dim))
        self.beta = nn.Parameter(torch.zeros(self.model_dim))
        self.eps = 1e-5

    def forward(self, x: Tensor) -> Tensor:
        """Normalize each token representation."""
        x_centered = x - torch.mean(x, dim=-1, keepdim=True)
        x_var = torch.mean(x_centered**2, dim=-1, keepdim=True)
        output = x_centered / torch.sqrt(x_var + self.eps)
        return self.gamma * output + self.beta


class Linear(nn.Module):
    """Linear projection with explicit weight initialization."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = False) -> None:
        super().__init__()
        del bias
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) / math.sqrt(in_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Project the last dimension."""
        return torch.matmul(x, self.W)


class GLU(nn.Module):
    """Gated linear unit used in the feed forward block."""

    def forward(self, x: Tensor) -> Tensor:
        """Split channels in half and gate one half with the other."""
        last_dim = x.shape[-1]
        split = last_dim // 2
        left = x[:, :, :split]
        right = x[:, :, split:]
        return left * torch.sigmoid(right)


def softmax_fn(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax."""
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_scaled = x - x_max
    num = torch.exp(x_scaled)
    den = torch.sum(num, dim=dim, keepdim=True)
    return num / den


def cross_entropy_loss(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """Cross entropy with optional ignored targets."""
    mask = targets != ignore_index
    logits = logits[mask]
    targets = targets[mask]
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    predictions = log_probs[torch.arange(len(targets), device=logits.device), targets]
    return -predictions.mean()
