"""Custom GPT model definition (decoder-only transformer)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor
from typing import List

from src.configs import ConfigParametersLLM, TransformerBlockConfig, NORM_REGISTRY, MLP_REGISTRY, PROJECTION_REGISTRY
from src.model.custom_modules import GLU, LayerNormalization, Linear, cross_entropy_loss
from src.model.attention import MultiHeadAttention, GroupQueryAttention, ATTENTION_REGISTRY
from typing import Literal
from dataclasses import dataclass


class TransformerBlock(nn.Module):
    """Pre norm transformer block."""

    def __init__(self, cfg_transformer_block: TransformerBlockConfig):
        super().__init__()
        
        assert cfg_transformer_block.attention_type in ATTENTION_REGISTRY, f"Attention type {cfg_transformer_block.attention_type} not found in ATTENTION_REGISTRY"
        assert cfg_transformer_block.norm_type in NORM_REGISTRY, f"Norm type {cfg_transformer_block.norm_type} not found in NORM_REGISTRY"
        assert cfg_transformer_block.mlp_type in MLP_REGISTRY, f"MLP type {cfg_transformer_block.mlp_type} not found in MLP_REGISTRY"
       

        self.attention = ATTENTION_REGISTRY[cfg_transformer_block.attention_type](cfg_transformer_block)
        self.layernorm1 = NORM_REGISTRY[cfg_transformer_block.norm_type](cfg_transformer_block)
        self.layernorm2 = NORM_REGISTRY[cfg_transformer_block.norm_type](cfg_transformer_block)
        self.mlp = MLP_REGISTRY[cfg_transformer_block.mlp_type](cfg_transformer_block)


    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        causal: bool = True,
    ) -> Tensor:
        """Apply attention then feed forward layers with residuals."""
        x1 = x + self.attention(self.layernorm1(x), attention_mask=attention_mask, causal=causal)
        x2 = x1 + self.mlp(self.layernorm2(x1))
        return x2


class GPT(nn.Module):
    """Decoder only transformer language model."""

    def __init__(self, cfg: ConfigParametersLLM, cfg_transformer_blocks: List[TransformerBlockConfig]) -> None:
        super().__init__()
        
        assert len(cfg_transformer_blocks) == cfg.num_blocks, "Number of transformer blocks must match number of blocks in configuration"
        self.blocks = nn.ModuleList(
            [TransformerBlock( cfg_transformer_block=cfg_transformer_blocks[i]) for i in range(cfg.num_blocks)]
        )

        if not hasattr(cfg, "logit_projection_type"):
            cfg.logit_projection_type = "custom_linear"

        assert PROJECTION_REGISTRY[cfg.logit_projection_type] is not None, f"Projection type {cfg.logit_projection_type} not found in PROJECTION_REGISTRY"
        self.logit_proj = PROJECTION_REGISTRY[cfg.logit_projection_type](cfg.model_dim, cfg.vocab_size)

        #learnable token embeddings
        self.token_embeddings = nn.Embedding(cfg.vocab_size, cfg.model_dim)

        #position embedding type
        self.pos_emb_type = cfg.pos_emb_type
        #if no position embedding type is provided, use a learned embedding
        if cfg.pos_emb_type is None or cfg.pos_emb_type == "absolute":
            self.pos_emb = nn.Embedding(cfg.max_seq_length, cfg.model_dim)
            

    def input_embeddings(self, x: Tensor, pos_emb_type: str | None = None) -> Tensor:
        """Add token and position embeddings for integer token ids."""
        
        _, seq_len = x.shape
        token_embeddings = self.token_embeddings(x)

        if pos_emb_type is None or pos_emb_type != "rope":
            pos_embeddings = self.pos_emb(torch.arange(seq_len, device=x.device))
            return token_embeddings + pos_embeddings
        
        return token_embeddings

    def forward(
        self,
        x: Tensor | None = None,
        targets: Tensor | None = None,
        input_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        causal: bool = True,
    ) -> tuple[Tensor, Tensor | float]:
        """Run the model and optionally compute training loss."""
        if x is None and input_embeds is None:
            raise ValueError("Either x or input_embeds must be provided")

        if input_embeds is not None:
            hidden = input_embeds
        else:
            hidden = self.input_embeddings(x, self.pos_emb_type)

        for block in self.blocks:
            hidden = block(hidden, attention_mask, causal)

        logits = self.logit_proj(hidden)
        batch_size, seq_len, vocab_size = logits.shape
        flat_logits = logits.view(batch_size * seq_len, vocab_size)

        loss: Tensor | float = 0.0
        if targets is not None:
            flat_targets = targets.reshape(batch_size * seq_len)
            loss = cross_entropy_loss(flat_logits, flat_targets)

        return flat_logits, loss
