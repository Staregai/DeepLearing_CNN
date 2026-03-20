from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from src.data.augmentations import build_eval_transforms, build_train_transforms


def _find_split(root: Path, preferred: str, alternatives: list[str]) -> Path:
    candidates = [preferred] + alternatives
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find split among: {candidates} in {root}")


def load_cinic10_datasets(data_root: Path, train_aug_profile: str = "baseline") -> Tuple[ImageFolder, ImageFolder, ImageFolder]:
    train_dir = _find_split(data_root, "train", ["training"])
    val_dir = _find_split(data_root, "valid", ["val", "validation"])
    test_dir = _find_split(data_root, "test", ["testing"])

    train_ds = ImageFolder(root=str(train_dir), transform=build_train_transforms(train_aug_profile))
    val_ds = ImageFolder(root=str(val_dir), transform=build_eval_transforms())
    test_ds = ImageFolder(root=str(test_dir), transform=build_eval_transforms())
    return train_ds, val_ds, test_ds


def make_dataloaders(
    train_ds: ImageFolder,
    val_ds: ImageFolder,
    test_ds: ImageFolder,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def make_reduced_subset(dataset: ImageFolder, ratio: float = 0.3, seed: int = 42) -> Subset:
    if not (0 < ratio <= 1.0):
        raise ValueError("ratio must be in (0, 1]")

    total = len(dataset)
    keep = int(total * ratio)

    import torch

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(total, generator=g)[:keep].tolist()
    return Subset(dataset, idx)


def subset_training_dataset(
    dataset: ImageFolder,
    seed: int = 42,
    subset_count: int | None = None,
    subset_ratio: float | None = None,
) -> ImageFolder | Subset:
    if subset_count is not None and subset_ratio is not None:
        raise ValueError("Use either subset_count or subset_ratio, not both")

    if subset_count is None and subset_ratio is None:
        return dataset

    total = len(dataset)

    if subset_ratio is not None:
        if not (0 < subset_ratio <= 1.0):
            raise ValueError("subset_ratio must be in (0, 1]")
        keep = max(1, int(total * subset_ratio))
    else:
        if subset_count is None or subset_count <= 0:
            raise ValueError("subset_count must be a positive integer")
        keep = min(subset_count, total)

    import torch

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(total, generator=g)[:keep].tolist()
    return Subset(dataset, idx)
