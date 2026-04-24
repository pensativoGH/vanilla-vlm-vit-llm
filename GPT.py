"""Standalone GPT training components"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


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


class MultiHeadAttention(nn.Module):
    """Multi head self attention used by GPT and ViT."""

    def __init__(self, model_dim: int, num_heads: int, device: torch.device | str) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.device = device

        self.q_proj = Linear(self.model_dim, self.model_dim, bias=False)
        self.k_proj = Linear(self.model_dim, self.model_dim, bias=False)
        self.v_proj = Linear(self.model_dim, self.model_dim, bias=False)
        self.o_proj = Linear(self.model_dim, self.model_dim, bias=False)

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

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

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


class TransformerBlock(nn.Module):
    """Pre norm transformer block."""

    def __init__(self, model_dim: int, num_heads: int, device: torch.device | str) -> None:
        super().__init__()
        self.MHSA = MultiHeadAttention(model_dim, num_heads, device)
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

    def __init__(self, cfg: "ConfigParametersLLM") -> None:
        super().__init__()
        self.model_dim = cfg.model_dim
        self.num_heads = cfg.num_heads
        self.num_blocks = cfg.num_blocks
        self.vocab_size = cfg.vocab_size
        self.max_seq_len = cfg.max_seq_length
        self.device = cfg.device

        self.blocks = nn.ModuleList(
            [TransformerBlock(self.model_dim, self.num_heads, self.device) for _ in range(self.num_blocks)]
        )
        self.logit_proj = Linear(self.model_dim, cfg.vocab_size, bias=False)
        self.token_embeddings = nn.Embedding(self.vocab_size, self.model_dim)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.model_dim)

    def input_embeddings(self, x: Tensor) -> Tensor:
        """Add token and position embeddings for integer token ids."""
        _, seq_len = x.shape
        token_embeddings = self.token_embeddings(x)
        pos_embeddings = self.pos_emb(torch.arange(seq_len, device=self.device))
        return token_embeddings + pos_embeddings

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
            hidden = self.input_embeddings(x)

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


@dataclass
class ConfigParametersLLM:
    """GPT hyperparameters used in the notebook."""

    vocab_size: int
    device: str | torch.device
    max_seq_length: int = 512
    model_dim: int = 768
    num_heads: int = 4
    chunk_size: int = 256
    batch_size: int = 16
    num_blocks: int = 2


@dataclass
class OptimParameters:
    """AdamW configuration."""

    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8


class PreTrainTextDataset(Dataset[tuple[Tensor, Tensor]]):
    """Sliding window dataset for autoregressive language modeling."""

    def __init__(self, data: np.ndarray, chunk_size: int) -> None:
        super().__init__()
        self.data = data
        self.chunk_size = chunk_size

    def __len__(self) -> int:
        """Return the number of valid windows."""
        return len(self.data) - self.chunk_size

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return input and next token targets for one window."""
        x = torch.from_numpy(self.data[idx : idx + self.chunk_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + 1 + self.chunk_size].astype(np.int64))
        return x, y


def load_tinyshakespeare_tokens(
    data_path: str | Path = "./char-rnn/data/tinyshakespeare/input.txt",
    tokenizer_name: str = "gpt2",
) -> tuple[AutoTokenizer, np.ndarray]:
    """Tokenize the Tiny Shakespeare corpus with a GPT tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    eos_id = tokenizer.eos_token_id
    text = Path(data_path).read_text()
    ids = tokenizer(text, add_special_tokens=False)["input_ids"] + [eos_id]
    return tokenizer, np.array(ids, dtype=np.uint16)


def build_gpt_dataloader(
    data: np.ndarray,
    cfg: ConfigParametersLLM,
    shuffle: bool = True,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build the GPT pretraining dataloader."""
    dataset = PreTrainTextDataset(data, cfg.chunk_size)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle)


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
