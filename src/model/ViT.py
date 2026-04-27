"""Custom ViT model definition (vision transformer classifier)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from GPT import TransformerBlock, Linear, cross_entropy_loss
from configs import ConfigParametersViT


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
