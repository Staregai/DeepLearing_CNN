from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from src.config import TrainConfig
from src.data.cinic10 import load_cinic10_datasets, make_dataloaders, subset_training_dataset
from src.models.cnn_baseline import BaselineCNN
from src.training.supervised import train_supervised
from src.utils.device import get_device
from src.utils.io import save_json
from src.utils.reproducibility import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CNN hyperparameter grid on CINIC-10")
    parser.add_argument("--data-dir", type=Path, default=Path("src/dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cnn_grid"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    subset_group = parser.add_mutually_exclusive_group()
    subset_group.add_argument("--train-subset-ratio", type=float, default=None)
    subset_group.add_argument("--train-subset-size", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    aug_profiles = ["baseline", "color_jitter", "autoaugment", "cutout", "compression", "combo"]
    optimizers = ["sgd", "adam"]
    momentums = [0.7,0.8, 0.9]
    label_smoothing_vals = [0.0, 0.1, 0.2]
    dropouts = [0.0, 0.15, 0.3]

    runs = []

    for aug, opt, mom, ls, do in product(aug_profiles, optimizers, momentums, label_smoothing_vals, dropouts):
        if opt == "adam" and mom != momentums[0]:
            continue

        train_ds, val_ds, test_ds = load_cinic10_datasets(args.data_dir, train_aug_profile=aug)
        train_ds = subset_training_dataset(
            train_ds,
            seed=args.seed,
            subset_count=args.train_subset_size,
            subset_ratio=args.train_subset_ratio,
        )
        train_loader, val_loader, test_loader = make_dataloaders(
            train_ds, val_ds, test_ds, args.batch_size, num_workers=4
        )

        cfg = TrainConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=1e-3 if opt == "adam" else 1e-2,
            optimizer=opt,
            momentum=mom,
            label_smoothing=ls,
            dropout=do,
            seed=args.seed,
        )

        run_name = f"aug={aug}_opt={opt}_mom={mom}_ls={ls}_drop={do}"
        out = args.out_dir / run_name.replace("/", "-")

        model = BaselineCNN(num_classes=10, dropout=do)
        result = train_supervised(model, train_loader, val_loader, test_loader, cfg, out, device)
        result["run_name"] = run_name
        runs.append(result)

    best = max(runs, key=lambda x: x["best_val_accuracy"])
    summary = {"best": best, "all_runs": runs}
    save_json(summary, args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
