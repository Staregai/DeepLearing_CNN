from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import TrainConfig
from src.data.cinic10 import load_cinic10_datasets, make_dataloaders, make_reduced_subset, subset_training_dataset
from src.models.cnn_baseline import BaselineCNN
from src.training.supervised import train_supervised
from src.utils.device import get_device
from src.utils.io import save_json
from src.utils.reproducibility import set_seed


def main() -> None:
    cfg_defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Compare full vs reduced data for CNN")
    parser.add_argument("--data-dir", type=Path, default=Path("src/dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/reduced_data"))
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=cfg_defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg_defaults.batch_size)
    parser.add_argument("--seed", type=int, default=cfg_defaults.seed)
    parser.add_argument("--resume", action="store_true", help="Resume from full/reduced train_state.pt files")
    subset_group = parser.add_mutually_exclusive_group()
    subset_group.add_argument("--train-subset-ratio", type=float, default=None)
    subset_group.add_argument("--train-subset-size", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    train_ds, val_ds, test_ds = load_cinic10_datasets(args.data_dir, train_aug_profile="combo")
    train_ds = subset_training_dataset(
        train_ds,
        seed=args.seed,
        subset_count=args.train_subset_size,
        subset_ratio=args.train_subset_ratio,
    )
    reduced_train = make_reduced_subset(train_ds, ratio=args.ratio, seed=args.seed)

    full_train_loader, val_loader, test_loader = make_dataloaders(train_ds, val_ds, test_ds, args.batch_size, 4)

    from torch.utils.data import DataLoader

    reduced_loader = DataLoader(
        reduced_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    cfg = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=1e-3,
        optimizer="adam",
        label_smoothing=0.1,
        dropout=0.3,
        seed=args.seed,
    )

    full_model = BaselineCNN(num_classes=10, dropout=cfg.dropout)
    reduced_model = BaselineCNN(num_classes=10, dropout=cfg.dropout)

    full_resume_state = (args.out_dir / "full" / "train_state.pt") if args.resume else None
    reduced_resume_state = (args.out_dir / "reduced" / "train_state.pt") if args.resume else None

    full_result = train_supervised(
        full_model,
        full_train_loader,
        val_loader,
        test_loader,
        cfg,
        args.out_dir / "full",
        device,
        resume_state=full_resume_state,
    )
    reduced_result = train_supervised(
        reduced_model,
        reduced_loader,
        val_loader,
        test_loader,
        cfg,
        args.out_dir / "reduced",
        device,
        resume_state=reduced_resume_state,
    )

    summary = {
        "ratio": args.ratio,
        "full_test_accuracy": full_result["test_metrics"]["accuracy"],
        "reduced_test_accuracy": reduced_result["test_metrics"]["accuracy"],
        "accuracy_drop": full_result["test_metrics"]["accuracy"] - reduced_result["test_metrics"]["accuracy"],
    }
    save_json(summary, args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
