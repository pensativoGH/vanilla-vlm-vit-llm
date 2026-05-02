"""Custom GPT model definition (decoder-only transformer)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from src.configs import ConfigParametersLLM


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

from src.model.attention import GroupQueryAttention, MultiHeadAttention


class TransformerBlock(nn.Module):
    """Pre norm transformer block."""

    def __init__(self, model_dim: int, num_heads: int, device: torch.device | str, pos_emb_type: str | None = None) -> None:
        super().__init__()
        self.pos_emb_type = pos_emb_type
        self.MHSA = MultiHeadAttention(model_dim, num_heads, device, pos_emb_type)
        self.layernorm1 = LayerNormalization(model_dim)
        self.layernorm2 = LayerNormalization(model_dim)
        self.FFN = nn.Sequential(
            Linear(model_dim, 4 * model_dim, bias=False),
            GLU(),
            Linear(2 * model_dim, model_dim, bias=False),
        )

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        causal: bool = True,
    ) -> Tensor:
        """Apply attention then feed forward layers with residuals."""
        x1 = x + self.MHSA(self.layernorm1(x), attention_mask=attention_mask, causal=causal)
        x2 = x1 + self.FFN(self.layernorm2(x1))
        return x2


class GPT(nn.Module):
    """Decoder only transformer language model."""

    def __init__(self, cfg: ConfigParametersLLM) -> None:
        super().__init__()
        self.model_dim = cfg.model_dim
        self.num_heads = cfg.num_heads
        self.num_blocks = cfg.num_blocks
        self.vocab_size = cfg.vocab_size
        self.max_seq_len = cfg.max_seq_length
        self.device = cfg.device
        self.pos_emb_type = cfg.pos_emb_type


        self.blocks = nn.ModuleList(
            [TransformerBlock(self.model_dim, self.num_heads, self.device, self.pos_emb_type) for _ in range(self.num_blocks)]
        )
        self.logit_proj = Linear(self.model_dim, cfg.vocab_size, bias=False)
        self.token_embeddings = nn.Embedding(self.vocab_size, self.model_dim)

        #if no position embedding type is provided, use a learned embedding
        if cfg.pos_emb_type is None or cfg.pos_emb_type == "absolute":
            self.pos_emb = nn.Embedding(self.max_seq_len, self.model_dim)

    def input_embeddings(self, x: Tensor, pos_emb_type: str | None = None) -> Tensor:
        """Add token and position embeddings for integer token ids."""
        
        _, seq_len = x.shape
        token_embeddings = self.token_embeddings(x)

        if pos_emb_type is None or pos_emb_type != "rope":
            pos_embeddings = self.pos_emb(torch.arange(seq_len, device=self.device))
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
