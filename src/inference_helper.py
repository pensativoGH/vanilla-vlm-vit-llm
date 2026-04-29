"""Inference helpers for the VLM notebook."""

from __future__ import annotations

import os

import torch
from PIL import Image


def prepare_inference_input(
    messages: list[dict],
    tokenizer,
    max_len: int,
    transform,
    root: str | None = None,
) -> dict:
    """Convert a chat-style message into VLM model inputs (single sample)."""
    content = messages[0]["content"]
    img = None
    text = None

    for item in content:
        if item["type"] == "image":
            image_path = item.get("path") or item.get("url")
            if root is not None and not os.path.isabs(image_path):
                image_path = os.path.join(root, image_path)
            img = Image.open(image_path).convert("RGB")
            img = transform(img)
        elif item["type"] == "text":
            text = item["text"]

    text_ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_len]
    text_tokens = torch.tensor(text_ids, dtype=torch.long).unsqueeze(0)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    attention_mask = torch.ones_like(text_tokens)

    return {
        "image": img.unsqueeze(0),
        "text_tokens": text_tokens,
        "attention_mask": attention_mask,
    }
