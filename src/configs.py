"""Hyperparameter dataclasses for GPT and ViT training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class ConfigParametersLLM:
    """GPT hyperparameters used in the notebook."""

    vocab_size: int
    device: str | torch.device
    output_dir: str | None = None
    data_path: str | None = None
    max_seq_length: int = 512
    model_dim: int = 768
    num_heads: int = 4
    chunk_size: int = 256
    batch_size: int = 16
    num_blocks: int = 2

    @classmethod
    def from_dict(cls, data: dict) -> "ConfigParametersLLM":
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "ConfigParametersLLM":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


@dataclass
class OptimParameters:
    """AdamW and LR scheduler configuration."""

    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    min_lr: float = 3e-5
    warmup_steps: int = 500
    max_steps: int = 10000
    scheduler: bool | None = None
    compile: bool | None = None
    autocast: bool | None = None
    autocast_dtype: torch.dtype | None = None

    def __post_init__(self):
        if self.autocast is not None:
            if self.autocast_dtype is None:
                self.autocast_dtype = torch.bfloat16

    @classmethod
    def from_dict(cls, data: dict) -> "OptimParameters":
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "OptimParameters":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


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
    num_blocks: int = 4

    @classmethod
    def from_dict(cls, d: dict) -> "ConfigParametersViT":
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "ConfigParametersViT":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
