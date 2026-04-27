"""Vision and vision-language dataset / dataloader utilities."""

from __future__ import annotations

import json
import os
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


# ---------------------------------------------------------------------------
# CIFAR10 (used by the ViT classifier)
# ---------------------------------------------------------------------------

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


class CIFAR10(Dataset[tuple[Tensor, Tensor]]):
    """Numpy-based CIFAR10 dataset loader."""

    def __init__(
        self,
        split: str = "train",
        normalize: bool = True,
        channels_first: bool = True,
        root: str = "../cifar10",
    ) -> None:
        images = np.load(os.path.join(root, f"{split}_images.npy"))
        labels = np.load(os.path.join(root, f"{split}_labels.npy"))

        x = images.astype(np.float32) / 255.0
        if normalize:
            x = (x - CIFAR_MEAN) / CIFAR_STD
        if channels_first:
            x = x.transpose(0, 3, 1, 2)

        self.images = torch.from_numpy(np.ascontiguousarray(x))
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return one image and label."""
        return self.images[idx], self.labels[idx]


def get_loaders(
    batch_size: int = 128,
    num_workers: int = 0,
    **kwargs: object,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Build CIFAR10 train and validation loaders."""
    train_dataset = CIFAR10(split="train", **kwargs)
    test_dataset = CIFAR10(split="test", **kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# ImageNet-style transforms (used for VLM image preprocessing)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tfm = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tfm = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Imagenette QA dataset (image + question + answer)
# ---------------------------------------------------------------------------

class ImagenetteQADataset(Dataset):
    """Imagenette-based VQA pairs for VLM pretraining."""

    def __init__(
        self,
        split: str,
        tokenizer,
        transform: Callable,
        max_len: int = 256,
        data_dir: str = "./data",
        image_root: str = "../../datasets/imagenette/imagenette2",
        limit: int | None = None,
    ) -> None:
        self.transform = transform
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token_id

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.pad_id = tokenizer.pad_token_id

        self.max_len = max_len
        self.root = image_root

        fname = "imagenette_qa_train.json" if split == "train" else "imagenette_qa_val.json"
        with open(os.path.join(data_dir, fname)) as f:
            self.data = json.load(f)

        if limit is not None:
            self.data = self.data[:limit]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        img = Image.open(os.path.join(self.root, item["image"])).convert("RGB")
        img = self.transform(img)

        q_ids = self.tokenizer(item["question"], add_special_tokens=False)["input_ids"]
        a_ids = self.tokenizer(item["answer"], add_special_tokens=False)["input_ids"] + [self.eos]

        text_ids = (q_ids + a_ids)[:self.max_len]
        text_tokens = torch.tensor(text_ids, dtype=torch.long)

        # next-token targets
        targets = text_tokens.clone()
        targets[:-1] = text_tokens[1:]
        targets[-1] = self.eos

        # mask out question positions in the loss
        q_len = min(len(q_ids), len(text_tokens))
        targets[:q_len] = -100

        return {
            "image": img,
            "text_tokens": text_tokens,
            "targets": targets,
        }


def imagenette_qa_collate_fn(batch: list[dict], pad_id: int):
    """Right-pad text tokens to the longest in batch."""
    images = torch.stack([item["image"] for item in batch], dim=0)

    lengths = [item["text_tokens"].size(0) for item in batch]
    max_len = max(lengths)
    B = len(batch)

    text_tokens = torch.full((B, max_len), pad_id, dtype=torch.long)
    targets = torch.full((B, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["text_tokens"].size(0)
        text_tokens[i, :n] = item["text_tokens"]
        targets[i, :n] = item["targets"]
        attention_mask[i, :n] = 1

    return images, text_tokens, attention_mask, targets


# ---------------------------------------------------------------------------
# COCO Captions dataset (image + caption)
# ---------------------------------------------------------------------------

class CocoCaptionsDataset(Dataset):
    """COCO 2017 image-caption pairs for VLM pretraining.

    Each item is one (image, caption) pair. Each image has ~5 captions, so
    every image appears ~5 times across the dataset.
    """

    def __init__(
        self,
        split: str,
        tokenizer,
        transform: Callable,
        max_len: int = 64,
        data_dir: str = "./data/coco",
        limit: int | None = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.transform = transform
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token_id

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.pad_id = tokenizer.pad_token_id

        self.max_len = max_len
        self.image_dir = os.path.join(data_dir, f"{split}2017")

        ann_path = os.path.join(data_dir, "annotations", f"captions_{split}2017.json")
        with open(ann_path) as f:
            data = json.load(f)

        # image_id -> file_name lookup
        id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

        # one (file_name, caption) pair per annotation
        self.samples = [
            (id_to_file[a["image_id"]], a["caption"])
            for a in data["annotations"]
        ]

        if limit is not None:
            self.samples = self.samples[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        file_name, caption = self.samples[idx]

        img = Image.open(os.path.join(self.image_dir, file_name)).convert("RGB")
        img = self.transform(img)

        text_ids = self.tokenizer(caption, add_special_tokens=False)["input_ids"] + [self.eos]
        text_ids = text_ids[:self.max_len]
        text_tokens = torch.tensor(text_ids, dtype=torch.long)

        # next-token targets (shift by 1; last position predicts eos)
        targets = text_tokens.clone()
        targets[:-1] = text_tokens[1:]
        targets[-1] = self.eos

        return {
            "image": img,
            "text_tokens": text_tokens,
            "targets": targets,
        }


# ---------------------------------------------------------------------------
# VLM dataloader builders
# ---------------------------------------------------------------------------

def build_vlm_dataloaders(
    tokenizer,
    transform: Callable = val_tfm,
    batch_size: int = 32,
    max_len: int = 256,
    data_dir: str = "./data",
    image_root: str = "../../datasets/imagenette/imagenette2",
    num_workers: int = 2,
    train_limit: int | None = None,
    val_limit: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train + val dataloaders for the Imagenette QA dataset."""
    train_ds = ImagenetteQADataset(
        "train", tokenizer, transform, max_len=max_len,
        data_dir=data_dir, image_root=image_root, limit=train_limit,
    )
    val_ds = ImagenetteQADataset(
        "val", tokenizer, transform, max_len=max_len,
        data_dir=data_dir, image_root=image_root, limit=val_limit,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: imagenette_qa_collate_fn(batch, train_ds.pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: imagenette_qa_collate_fn(batch, val_ds.pad_id),
    )
    return train_loader, val_loader


def build_coco_dataloaders(
    tokenizer,
    transform: Callable = val_tfm,
    batch_size: int = 32,
    max_len: int = 64,
    data_dir: str = "./data/coco",
    num_workers: int = 4,
    train_limit: int | None = None,
    val_limit: int | None = 500,
) -> tuple[DataLoader, DataLoader]:
    """Build train + val dataloaders for COCO 2017 captions.

    val_limit defaults to 500 caption pairs (~100 images) for fast eval; pass
    val_limit=None to use the full 25K val set.
    """
    train_ds = CocoCaptionsDataset(
        "train", tokenizer, transform, max_len=max_len,
        data_dir=data_dir, limit=train_limit,
    )
    val_ds = CocoCaptionsDataset(
        "val", tokenizer, transform, max_len=max_len,
        data_dir=data_dir, limit=val_limit,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: imagenette_qa_collate_fn(batch, train_ds.pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: imagenette_qa_collate_fn(batch, val_ds.pad_id),
    )
    return train_loader, val_loader
