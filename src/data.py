from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


@dataclass
class CIFAR10Loaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def _maybe_subset(dataset: Dataset, max_items: Optional[int], seed: int) -> Dataset:
    if max_items is None or max_items <= 0 or max_items >= len(dataset):
        return dataset
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=gen)[:max_items].tolist()
    return Subset(dataset, indices)


def build_cifar10_loaders(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 2,
    image_size: int = 160,
    val_size: int = 5000,
    seed: int = 42,
    max_train_items: Optional[int] = None,
    max_val_items: Optional[int] = None,
    max_test_items: Optional[int] = None,
) -> CIFAR10Loaders:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    full_train = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_tf)
    full_train_eval = datasets.CIFAR10(
        root=str(data_dir), train=True, download=False, transform=eval_tf
    )
    test_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=eval_tf)

    if val_size <= 0 or val_size >= len(full_train):
        raise ValueError(f"val_size must be in [1, {len(full_train)-1}]")

    train_size = len(full_train) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset_idx = random_split(full_train, [train_size, val_size], generator=gen)

    # Rebuild val subset on eval transform (no augmentation).
    val_indices = val_subset_idx.indices  # type: ignore[attr-defined]
    val_subset = Subset(full_train_eval, val_indices)

    train_subset = _maybe_subset(train_subset, max_train_items, seed)
    val_subset = _maybe_subset(val_subset, max_val_items, seed + 1)
    test_ds = _maybe_subset(test_ds, max_test_items, seed + 2)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return CIFAR10Loaders(train=train_loader, val=val_loader, test=test_loader)
