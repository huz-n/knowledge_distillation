from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from tqdm.auto import tqdm


TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


@dataclass
class DatasetLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    num_classes: int
    dataset_name: str


def _maybe_subset(dataset: Dataset, max_items: Optional[int], seed: int) -> Dataset:
    if max_items is None or max_items <= 0 or max_items >= len(dataset):
        return dataset
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=gen)[:max_items].tolist()
    return Subset(dataset, indices)


def _build_loader_kwargs(
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
) -> dict:
    pin_memory = torch.cuda.is_available()
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def _build_transforms(
    dataset_name: str,
    image_size: int,
    augment: str,
) -> tuple[transforms.Compose, transforms.Compose]:
    if dataset_name.lower() in {"cifar10", "cifar100"}:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        base_size = 32
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        base_size = 64

    resize_step = []
    if image_size != base_size:
        resize_step = [transforms.Resize((image_size, image_size))]

    if augment == "none":
        train_tf = transforms.Compose(
            resize_step
            + [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    elif augment == "strong":
        train_tf = transforms.Compose(
            resize_step
            + [
                transforms.RandomCrop(image_size, padding=max(4, image_size // 8)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.0)),
            ]
        )
    else:
        train_tf = transforms.Compose(
            resize_step
            + [
                transforms.RandomCrop(image_size, padding=max(4, image_size // 8)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    eval_tf = transforms.Compose(
        resize_step
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, eval_tf


def _download_tiny_imagenet_if_needed(data_dir: Path) -> Path:
    tiny_root = data_dir / "tiny-imagenet-200"
    train_dir = tiny_root / "train"
    complete_flag = tiny_root / ".extract_complete"
    if train_dir.exists() and complete_flag.exists():
        return tiny_root

    zip_path = data_dir / "tiny-imagenet-200.zip"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"Downloading Tiny ImageNet from {TINY_IMAGENET_URL} ...")
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)

    print("Extracting Tiny ImageNet archive (first run can take a while on Drive) ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="Extracting Tiny ImageNet", unit="file"):
            target = data_dir / member.filename
            # Resume-friendly: skip already extracted paths.
            if target.exists():
                continue
            zf.extract(member, path=data_dir)
    complete_flag.write_text("ok\n", encoding="utf-8")
    print("Extraction finished.")
    return tiny_root


class TinyImageNetValDataset(Dataset):
    def __init__(
        self,
        tiny_root: Path,
        class_to_idx: dict[str, int],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        val_dir = tiny_root / "val"
        images_dir = val_dir / "images"
        ann_file = val_dir / "val_annotations.txt"
        if not ann_file.exists():
            raise FileNotFoundError(f"Missing Tiny ImageNet val annotations: {ann_file}")
        with ann_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_name, wnid = parts[0], parts[1]
                if wnid not in class_to_idx:
                    continue
                img_path = images_dir / img_name
                if img_path.exists():
                    self.samples.append((img_path, class_to_idx[wnid]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        img = default_loader(str(img_path))
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _build_cifar_loaders(
    dataset_name: str,
    data_dir: Path,
    train_tf: transforms.Compose,
    eval_tf: transforms.Compose,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    val_size: int,
    seed: int,
    max_train_items: Optional[int],
    max_val_items: Optional[int],
    max_test_items: Optional[int],
) -> DatasetLoaders:
    name = dataset_name.lower()
    if name == "cifar10":
        ds_cls = datasets.CIFAR10
        num_classes = 10
    elif name == "cifar100":
        ds_cls = datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError(f"Unsupported CIFAR dataset: {dataset_name}")

    full_train = ds_cls(root=str(data_dir), train=True, download=True, transform=train_tf)
    full_train_eval = ds_cls(root=str(data_dir), train=True, download=False, transform=eval_tf)
    test_ds = ds_cls(root=str(data_dir), train=False, download=True, transform=eval_tf)

    if val_size <= 0 or val_size >= len(full_train):
        raise ValueError(f"val_size must be in [1, {len(full_train)-1}]")

    train_size = len(full_train) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset_idx = random_split(full_train, [train_size, val_size], generator=gen)
    val_indices = val_subset_idx.indices  # type: ignore[attr-defined]
    val_subset = Subset(full_train_eval, val_indices)

    train_subset = _maybe_subset(train_subset, max_train_items, seed)
    val_subset = _maybe_subset(val_subset, max_val_items, seed + 1)
    test_ds = _maybe_subset(test_ds, max_test_items, seed + 2)

    loader_kwargs = _build_loader_kwargs(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return DatasetLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        num_classes=num_classes,
        dataset_name=dataset_name,
    )


def _build_tiny_imagenet_loaders(
    data_dir: Path,
    train_tf: transforms.Compose,
    eval_tf: transforms.Compose,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    val_size: int,
    seed: int,
    max_train_items: Optional[int],
    max_val_items: Optional[int],
    max_test_items: Optional[int],
) -> DatasetLoaders:
    tiny_root = _download_tiny_imagenet_if_needed(data_dir)
    train_dir = tiny_root / "train"
    full_train = datasets.ImageFolder(root=str(train_dir), transform=train_tf)
    full_train_eval = datasets.ImageFolder(root=str(train_dir), transform=eval_tf)
    # Avoid expensive file-by-file copy to class folders on Drive.
    test_ds = TinyImageNetValDataset(
        tiny_root=tiny_root,
        class_to_idx=full_train.class_to_idx,
        transform=eval_tf,
    )
    num_classes = len(full_train.classes)

    if val_size <= 0 or val_size >= len(full_train):
        raise ValueError(f"val_size must be in [1, {len(full_train)-1}] for tiny_imagenet")

    train_size = len(full_train) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset_idx = random_split(full_train, [train_size, val_size], generator=gen)
    val_indices = val_subset_idx.indices  # type: ignore[attr-defined]
    val_subset = Subset(full_train_eval, val_indices)

    train_subset = _maybe_subset(train_subset, max_train_items, seed)
    val_subset = _maybe_subset(val_subset, max_val_items, seed + 1)
    test_ds = _maybe_subset(test_ds, max_test_items, seed + 2)

    loader_kwargs = _build_loader_kwargs(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return DatasetLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        num_classes=num_classes,
        dataset_name="tiny_imagenet",
    )


def build_image_classification_loaders(
    dataset_name: str,
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 160,
    augment: str = "basic",
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    val_size: int = 5000,
    seed: int = 42,
    max_train_items: Optional[int] = None,
    max_val_items: Optional[int] = None,
    max_test_items: Optional[int] = None,
) -> DatasetLoaders:
    name = dataset_name.lower()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_tf, eval_tf = _build_transforms(dataset_name=name, image_size=image_size, augment=augment)

    if name in {"cifar10", "cifar100"}:
        return _build_cifar_loaders(
            dataset_name=name,
            data_dir=data_dir,
            train_tf=train_tf,
            eval_tf=eval_tf,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            val_size=val_size,
            seed=seed,
            max_train_items=max_train_items,
            max_val_items=max_val_items,
            max_test_items=max_test_items,
        )
    if name in {"tiny_imagenet", "tiny-imagenet", "tinyimagenet"}:
        return _build_tiny_imagenet_loaders(
            data_dir=data_dir,
            train_tf=train_tf,
            eval_tf=eval_tf,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            val_size=val_size,
            seed=seed,
            max_train_items=max_train_items,
            max_val_items=max_val_items,
            max_test_items=max_test_items,
        )
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")
