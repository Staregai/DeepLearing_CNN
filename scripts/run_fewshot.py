from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from src.config import FewShotConfig
from src.data.cinic10 import load_cinic10_datasets, subset_training_dataset
from src.models.protonet import ProtoEncoder
from src.models.prototypical_classifier import PrototypicalClassifier, build_class_prototypes
from src.training.fewshot import train_fewshot
from src.training.supervised import run_epoch
from src.utils.device import get_device
from src.utils.io import save_json
from src.utils.reproducibility import set_seed


def main() -> None:
    cfg_defaults = FewShotConfig()
    parser = argparse.ArgumentParser(description="Train Prototypical Network on CINIC-10")
    parser.add_argument("--data-dir", type=Path, default=Path("src/dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/fewshot"))
    parser.add_argument("--epochs", type=int, default=cfg_defaults.epochs)
    parser.add_argument("--seed", type=int, default=cfg_defaults.seed)
    parser.add_argument("--resume", action="store_true", help="Resume from out-dir/train_state.pt if available")
    subset_group = parser.add_mutually_exclusive_group()
    subset_group.add_argument("--train-subset-ratio", type=float, default=None)
    subset_group.add_argument("--train-subset-size", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Training device: {'GPU' if device.type == 'cuda' else 'CPU'} ({device})")

    train_ds, val_ds, test_ds = load_cinic10_datasets(args.data_dir, train_aug_profile="combo")
    train_ds = subset_training_dataset(
        train_ds,
        seed=args.seed,
        subset_count=args.train_subset_size,
        subset_ratio=args.train_subset_ratio,
    )

    cfg = FewShotConfig(epochs=args.epochs, seed=args.seed)

    encoder = ProtoEncoder()
    resume_state = args.out_dir / "train_state.pt" if args.resume else None
    train_result = train_fewshot(
        encoder,
        train_ds,
        val_ds,
        cfg,
        args.out_dir,
        device,
        checkpoint_every=5,
        resume_state=resume_state,
    )

    import torch

    encoder.load_state_dict(torch.load(args.out_dir / "best.pt", map_location=device))
    encoder.to(device)

    support_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    prototypes = build_class_prototypes(encoder, support_loader, num_classes=10, device=device, max_batches=200)
    proto_clf = PrototypicalClassifier(encoder, prototypes.to(device)).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    test_metrics = run_epoch(proto_clf, test_loader, criterion, optimizer=None, device=device, train=False)

    result = {"train": train_result, "test_metrics": test_metrics}
    save_json(result, args.out_dir / "eval_result.json")


if __name__ == "__main__":
    main()
