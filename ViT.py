"""Standalone ViT training components extracted from VLM.ipynb."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from GPT import OptimParameters, TransformerBlock, Linear, cross_entropy_loss


MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


@dataclass
class ConfigParametersViT:
    """ViT hyperparameters used in the notebook."""

    patch_size: int
    patch_dim: int
    num_patches: int
    num_classes: int
    device: str | torch.device
    model_dim: int = 256
    num_heads: int = 4
    batch_size: int = 64
    num_blocks: int = 2


class CIFAR10(Dataset[tuple[Tensor, Tensor]]):
    """Numpy based CIFAR10 dataset loader."""

    def __init__(
        self,
        split: str = "train",
        normalize: bool = True,
        channels_first: bool = True,
        root: str = "../cifar10",
    ) -> None:
        images = np.load(os.path.join(root, f"{split}_images.npy"))
        labels = np.load(os.path.join(root, f"{split}_labels.npy"))

        x = images.astype(np.float32) / 255.0
        if normalize:
            x = (x - MEAN) / STD
        if channels_first:
            x = x.transpose(0, 3, 1, 2)

        self.images = torch.from_numpy(np.ascontiguousarray(x))
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return one image and label."""
        return self.images[idx], self.labels[idx]


def get_loaders(
    batch_size: int = 128,
    num_workers: int = 0,
    **kwargs: object,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Build CIFAR10 train and validation loaders."""
    train_dataset = CIFAR10(split="train", **kwargs)
    test_dataset = CIFAR10(split="test", **kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


class ViT(nn.Module):
    """Vision transformer classifier."""

    def __init__(self, cfg: ConfigParametersViT) -> None:
        super().__init__()
        self.model_dim = cfg.model_dim
        self.num_heads = cfg.num_heads
        self.num_blocks = cfg.num_blocks
        self.num_patches = cfg.num_patches
        self.device = cfg.device
        self.patch_size = cfg.patch_size
        self.patch_dim = cfg.patch_dim
        self.num_classes = cfg.num_classes

        self.blocks = nn.ModuleList(
            [TransformerBlock(self.model_dim, self.num_heads, self.device) for _ in range(self.num_blocks)]
        )
        self.logit_proj = Linear(self.model_dim, cfg.num_classes, bias=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.model_dim))
        self.x_proj = Linear(self.patch_dim, self.model_dim)
        self.pos_emb = nn.Embedding(self.num_patches + 1, self.model_dim)

    def img_to_patch(self, x: Tensor) -> Tensor:
        """Split images into flattened non overlapping patches."""
        batch_size, channels, height, width = x.shape
        patch_size = self.patch_size

        x = x.unfold(dimension=2, size=patch_size, step=patch_size)
        x = x.unfold(dimension=3, size=patch_size, step=patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5)

        num_patches = height // patch_size * width // patch_size
        patch_dim = channels * patch_size * patch_size
        x = x.reshape(batch_size, num_patches, channels, patch_size, patch_size)
        x = x.reshape(batch_size, num_patches, patch_dim)
        return self.x_proj(x)

    def get_contextualized_embeddings(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        """Create patch tokens, add class token and pass through transformer blocks."""
        batch_size = x.shape[0]
        x = self.img_to_patch(x)
        cls_token = self.cls_token.expand(batch_size, 1, self.model_dim)
        x = torch.cat([cls_token, x], dim=1)
        pos_emb = self.pos_emb(torch.arange(self.num_patches + 1, device=x.device))
        x = x + pos_emb

        for block in self.blocks:
            x = block(x, attention_mask, causal)

        return x

    def encode(self, x: Tensor) -> Tensor:
        """Return contextualized patch embeddings without the class token."""
        x = self.get_contextualized_embeddings(x)
        return x[:, 1:, :]

    def forward(
        self,
        x: Tensor,
        targets: Tensor | None = None,
        attention_mask: Tensor | None = None,
        causal: bool = False,
    ) -> tuple[Tensor, Tensor | float]:
        """Run image classification and optional loss computation."""
        hidden = self.get_contextualized_embeddings(x, attention_mask, causal)
        cls_logit = hidden[:, 0, :]
        logits = self.logit_proj(cls_logit)

        loss: Tensor | float = 0.0
        if targets is not None:
            loss = cross_entropy_loss(logits, targets)

        return logits, loss


def evaluate(
    model: ViT,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device | str,
) -> tuple[float, float]:
    """Compute mean loss and accuracy over a dataloader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits, loss = model(x, y)
            total_loss += float(loss) * x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == y).sum().item())
            total_samples += x.size(0)

    model.train()
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


__all__ = [
    "CIFAR10",
    "ConfigParametersViT",
    "MEAN",
    "OptimParameters",
    "STD",
    "ViT",
    "evaluate",
    "get_loaders",
]
