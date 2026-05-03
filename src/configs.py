"""Hyperparameter dataclasses for GPT and ViT training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn

from src.model.custom_modules import LayerNormalization, Linear, GLU



@dataclass
class ConfigParametersLLM:
    """GPT hyperparameters used in the notebook."""

    vocab_size: int
    device: str | torch.device
    output_dir: str | None = None
    data_path: str | None = None
    max_seq_length: int = 512
    chunk_size: int = 256
    batch_size: int = 16
    num_blocks: int = 2
    pos_emb_type: str | None = "rope"
    logit_projection_type: str | None = "linear"
    model_dim: int = 768
    
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
    pos_emb_type: str | None = "rope"

    @classmethod
    def from_dict(cls, d: dict) -> "ConfigParametersViT":
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "ConfigParametersViT":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


@dataclass
class ConfigParametersVLM:
    """VLM configuration. Vision encoder + LLM are passed in already built."""

    model_dim: int
    vision_model_dim: int
    device: str | torch.device
    vision_encoder: Optional[object] = None
    LLM: Optional[object] = None
    batch_size: int = 32
    pos_emb_type: str | None = "rope"


    @classmethod
    def from_dict(cls, d: dict) -> ConfigParametersVLM:
        return cls(**d)



@dataclass
class OptimParametersVLM:
    """AdamW configuration for VLM training."""

    lr: float = 2e-5
    betas: Tuple[float, float] = (0.9, 0.99)
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
    def from_dict(cls, d: dict) -> OptimParametersVLM:
        return cls(**d)


NORM_REGISTRY = {
    "custom_layernorm": lambda cfg: LayerNormalization(cfg.model_dim),
    "layernorm": lambda cfg: nn.LayerNorm(cfg.model_dim),
}

PROJECTION_REGISTRY = {
    "custom_linear": lambda dim_in, dim_out: Linear(dim_in, dim_out, bias=False),
    "linear": lambda dim_in, dim_out: nn.Linear(dim_in, dim_out, bias=False),
}


MLP_REGISTRY = {
    "custom_ffn_glu": lambda cfg: nn.Sequential(
        Linear(cfg.model_dim, 4 * cfg.model_dim, bias=False),
        GLU(),
        Linear((4 * cfg.model_dim) // 2, cfg.model_dim, bias=False),
    ),
    "ffn_glu": lambda cfg: nn.Sequential(
        nn.Linear(cfg.model_dim, 4 * cfg.model_dim, bias=False),
        nn.GLU(),
        nn.Linear((4* cfg.model_dim) // 2, cfg.model_dim, bias=False),
    ),
}


@dataclass
class TransformerBlockConfig:
    attention_type: Literal["mha", "gqa"] = "mha"
    norm_type: Literal["layernorm", "custom_layernorm"] = "custom_layernorm"
    projection_type: Literal["linear", "custom_linear"] = "custom_linear"
    mlp_type: Literal["ffn_glu", "custom_ffn_glu"] = "custom_ffn_glu"
    num_heads: int = 4
    q_heads: int = 16
    kv_heads: int = 4
    model_dim: int = 768
    pos_emb_type: str | None = "rope"

    @classmethod
    def from_dict(cls, data: dict) -> "TransformerBlockConfig":
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "TransformerBlockConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
