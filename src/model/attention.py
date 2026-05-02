"""Attention modules shared by the custom GPT and ViT models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor
from torchtune.modules import RotaryPositionalEmbeddings as RotaryEmbedding

from src.model.gpt import Linear, softmax_fn


class GroupQueryAttention(nn.Module):
    def __init__(self, model_dim: int, q_heads: int, kv_heads: int, device: torch.device | str, pos_emb_type: str | None = None) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.q_heads = q_heads
        self.kv_heads = kv_heads

        assert self.q_heads % self.kv_heads == 0, "q_heads must be divisible by kv_heads"
        assert model_dim % q_heads == 0, "model_dim must be divisible by q_heads"

        self.head_dim = model_dim // q_heads
        self.device = device

        self.q_proj = Linear(self.model_dim, self.q_heads * self.head_dim, bias=False)
        self.k_proj = Linear(self.model_dim, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(self.model_dim, self.kv_heads * self.head_dim, bias=False)

        self.o_proj = Linear(self.model_dim, self.model_dim, bias=False)

        self.pos_emb = None
        if pos_emb_type is not None and pos_emb_type == "rope":
            self.pos_emb = RotaryEmbedding(dim=self.head_dim, max_seq_len=2048)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None, causal: bool = False) -> Tensor:
        """Run group query attention with optional causal and key masking."""

        batch_size, seq_len, model_dim = x.shape

        assert model_dim == self.model_dim, "model_dim must match the input dimension"

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.q_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.kv_heads, self.head_dim)

        repeat_factor = self.q_heads // self.kv_heads
        assert repeat_factor >= 1, "q_heads must be at least as large as kv_heads"

        k = k.repeat_interleave(repeat_factor, dim=2)
        v = v.repeat_interleave(repeat_factor, dim=2)

        if self.pos_emb is not None:
            pos = torch.arange(seq_len, device=x.device)
            q = self.pos_emb(q, input_pos=pos)
            k = self.pos_emb(k, input_pos=pos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        score = torch.matmul(q, k.transpose(-2, -1))
        score = score / math.sqrt(self.head_dim)

        if causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            score = score.masked_fill(causal_mask == 0, float("-inf"))

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].bool()
            score = score.masked_fill(~key_mask, float("-inf"))

        attn_score = softmax_fn(score, dim=-1)
        attention = torch.matmul(attn_score, v)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
        return self.o_proj(attention)


class MultiHeadAttention(nn.Module):
    """Multi head self attention used by GPT and ViT."""

    def __init__(self, model_dim: int, num_heads: int, device: torch.device | str, pos_emb_type: str | None = None) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.device = device

        self.q_proj = Linear(self.model_dim, self.model_dim, bias=False)
        self.k_proj = Linear(self.model_dim, self.model_dim, bias=False)
        self.v_proj = Linear(self.model_dim, self.model_dim, bias=False)
        self.o_proj = Linear(self.model_dim, self.model_dim, bias=False)

        self.pos_emb = None
        if pos_emb_type is not None and pos_emb_type == "rope":
            self.pos_emb = RotaryEmbedding(dim=self.head_dim, max_seq_len=2048)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        """Run self attention with optional causal and key masking."""
        batch_size, seq_len, model_dim = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        if self.pos_emb is not None:
            pos = torch.arange(seq_len, device=self.device)
            q = self.pos_emb(q, input_pos=pos)
            k = self.pos_emb(k, input_pos=pos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        score = torch.matmul(q, k.transpose(-2, -1))
        score = score / math.sqrt(self.head_dim)

        if causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            score = score.masked_fill(causal_mask == 0, float("-inf"))

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].bool()
            score = score.masked_fill(~key_mask, float("-inf"))

        attn_score = softmax_fn(score, dim=-1)
        attention = torch.matmul(attn_score, v)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
        return self.o_proj(attention)
