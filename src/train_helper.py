"""Training and evaluation helpers for GPT and ViT."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)
from torch.utils.data import DataLoader

from configs import OptimParameters
from GPT import GPT
from ViT import ViT
from VLM import VLM


def get_lr_scheduler(optimizer: Optimizer, cfg: OptimParameters) -> LRScheduler:
    """Linear warmup followed by cosine decay to ``cfg.min_lr``."""
    warmup = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=cfg.warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_steps - cfg.warmup_steps,
        eta_min=cfg.min_lr,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[cfg.warmup_steps],
    )


def evaluate_gpt_loss(
    model: GPT,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device | str,
    max_batches: int | None = None,
) -> float:
    """Compute mean loss over a dataloader."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            total_loss += float(loss)
            total_batches += 1

            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    model.train()
    if total_batches == 0:
        return 0.0
    return total_loss / total_batches


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


@torch.no_grad()
def validate_vlm(
    model: VLM,
    val_loader: DataLoader,
    device: torch.device | str,
    max_batches: int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> float:
    """Compute mean validation loss over the val dataloader.

    Args:
        model: VLM instance.
        val_loader: yields (images, text_tokens, attention_mask, targets).
        device: cuda device or string.
        max_batches: cap on batches (handy when val is large).
        autocast_dtype: if set, run forward inside `torch.autocast` with this
            dtype (match training to keep loss numbers comparable).

    Returns:
        Mean loss across processed batches.
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    n_batches = 0

    for batch_idx, (images, text_tokens, attention_mask, targets) in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        text_tokens = text_tokens.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        if autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                _, loss = model(
                    x_img=images, x_text=text_tokens,
                    targets=targets, attention_mask=attention_mask,
                )
        else:
            _, loss = model(
                x_img=images, x_text=text_tokens,
                targets=targets, attention_mask=attention_mask,
            )

        total_loss += loss.item()
        n_batches += 1

    if was_training:
        model.train()

    return total_loss / max(n_batches, 1)
