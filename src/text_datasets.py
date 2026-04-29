"""Text dataset and dataloader utilities for GPT pretraining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.configs import ConfigParametersLLM


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


def load_text_tokens(
    data_path: str | Path = "./char-rnn/data/tinyshakespeare/input.txt",
    tokenizer_name: str = "gpt2",
) -> tuple[AutoTokenizer, np.ndarray]:
    """Tokenize the text corpus with a GPT tokenizer.

    Args:
        data_path: path to the text file to tokenize.
        tokenizer_name: HuggingFace tokenizer name.

    Returns:
        the tokenizer and the tokenized data as a uint16 numpy array.
    """
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
