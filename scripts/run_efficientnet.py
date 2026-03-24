from __future__ import annotations

import argparse
from pathlib import Path

from src.config import TrainConfig
from src.data.cinic10 import load_cinic10_datasets, make_dataloaders, subset_training_dataset
from src.models.efficientnet import build_efficientnet
from src.training.supervised import train_supervised
from src.utils.device import get_device
from src.utils.reproducibility import set_seed


def main() -> None:
    cfg_defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train EfficientNet on CINIC-10")
    parser.add_argument("--data-dir", type=Path, default=Path("src/dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/efficientnet"))
    parser.add_argument("--epochs", type=int, default=cfg_defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg_defaults.batch_size)
    parser.add_argument("--seed", type=int, default=cfg_defaults.seed)
    parser.add_argument("--resume", action="store_true", help="Resume from out-dir/train_state.pt if available")
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
    train_loader, val_loader, test_loader = make_dataloaders(train_ds, val_ds, test_ds, args.batch_size, num_workers=4)

    cfg = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=1e-4,
        optimizer="adam",
        label_smoothing=0.1,
        dropout=0.2,
        seed=args.seed,
    )

    model = build_efficientnet(num_classes=10, dropout=0.2)
    resume_state = args.out_dir / "train_state.pt" if args.resume else None
    train_supervised(
        model,
        train_loader,
        val_loader,
        test_loader,
        cfg,
        args.out_dir,
        device,
        resume_state=resume_state,
    )


if __name__ == "__main__":
    main()
