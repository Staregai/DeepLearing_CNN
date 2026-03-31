from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data.cinic10 import load_cinic10_datasets, make_dataloaders, subset_training_dataset
from src.models.cnn_baseline import BaselineCNN
from src.training.early_stopping import EarlyStopping
from src.training.supervised import run_epoch
from src.utils.device import get_device
from src.utils.io import ensure_dir, save_json
from src.utils.reproducibility import set_seed
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def _as_cpu_byte_tensor(state) -> torch.Tensor:
    if isinstance(state, torch.Tensor):
        return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu").contiguous()


def main() -> None:
    cfg_defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train single CNN with early stopping")
    parser.add_argument("--data-dir", type=Path, default=Path("src/dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cnn_single"))
    parser.add_argument("--epochs", type=int, default=cfg_defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg_defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=cfg_defaults.learning_rate)
    parser.add_argument("--optimizer", type=str, default=cfg_defaults.optimizer, choices=["adam", "sgd"])
    parser.add_argument("--momentum", type=float, default=cfg_defaults.momentum)
    parser.add_argument("--label-smoothing", type=float, default=cfg_defaults.label_smoothing)
    parser.add_argument("--dropout", type=float, default=cfg_defaults.dropout)
    parser.add_argument("--seed", type=int, default=cfg_defaults.seed)
    parser.add_argument("--patience", type=int, default=cfg_defaults.early_stopping_patience, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=cfg_defaults.early_stopping_min_delta, help="Minimum loss delta")
    parser.add_argument("--checkpoint-every", type=int, default=cfg_defaults.checkpoint_every, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from out-dir/train_state.pt if available")
    parser.add_argument("--aug-profile", type=str, default="autoaugment")
    subset_group = parser.add_mutually_exclusive_group()
    subset_group.add_argument("--train-subset-ratio", type=float, default=None)
    subset_group.add_argument("--train-subset-size", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Training device: {'GPU' if device.type == 'cuda' else 'CPU'} ({device})")

    train_ds, val_ds, test_ds = load_cinic10_datasets(args.data_dir, train_aug_profile=args.aug_profile)
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
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        momentum=args.momentum,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        seed=args.seed,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        checkpoint_every=args.checkpoint_every,
    )

    ensure_dir(args.out_dir)
    writer = SummaryWriter(log_dir=str(args.out_dir / "tb"))

    model = BaselineCNN(num_classes=10, dropout=cfg.dropout)
    model.to(device)

    from torch.optim import Adam, SGD

    if cfg.optimizer.lower() == "sgd":
        optimizer = SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    early_stop = EarlyStopping(
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
        checkpoint_path=args.out_dir / "best.pt",
    )

    best_val_acc = -1.0
    best_ckpt = args.out_dir / "best.pt"
    history = {"train": [], "val": [], "early_stopped_epoch": None}
    state_path = args.out_dir / "train_state.pt"
    start_epoch = 0
    last_epoch = -1

    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        best_val_acc = float(state.get("best_val_accuracy", -1.0))
        history = state.get("history", history)
        saved_next_epoch = int(state.get("epoch", -1)) + 1
        history_next_epoch = len(history.get("train", []))
        start_epoch = history_next_epoch if history_next_epoch > 0 else saved_next_epoch

        es_state = state.get("early_stopping", {})
        early_stop.best_loss = float(es_state.get("best_loss", early_stop.best_loss))
        early_stop.counter = int(es_state.get("counter", early_stop.counter))
        early_stop.stopped_epoch = int(es_state.get("stopped_epoch", early_stop.stopped_epoch))

        if "torch_rng_state" in state and state["torch_rng_state"] is not None:
            torch.set_rng_state(_as_cpu_byte_tensor(state["torch_rng_state"]))
        if torch.cuda.is_available() and "cuda_rng_state_all" in state and state["cuda_rng_state_all"] is not None:
            cuda_states = state["cuda_rng_state_all"]
            if isinstance(cuda_states, (list, tuple)):
                torch.cuda.set_rng_state_all([_as_cpu_byte_tensor(s) for s in cuda_states])
            else:
                torch.cuda.set_rng_state_all([_as_cpu_byte_tensor(cuda_states)])

    for epoch in range(start_epoch, cfg.epochs):
        last_epoch = epoch
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_ckpt)

        if cfg.checkpoint_every > 0 and (epoch + 1) % cfg.checkpoint_every == 0:
            torch.save(model.state_dict(), args.out_dir / f"epoch_{epoch + 1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_val_acc,
                    "history": history,
                    "early_stopping": {
                        "best_loss": early_stop.best_loss,
                        "counter": early_stop.counter,
                        "stopped_epoch": early_stop.stopped_epoch,
                    },
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                state_path,
            )

        stopped = early_stop(val_metrics["loss"])
        improvement = early_stop.get_improvement(val_metrics["loss"])
        print(
            f"Epoch {epoch+1}/{cfg.epochs} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Improvement: {improvement:.6f} | Patience: {early_stop.counter}/{cfg.early_stopping_patience}"
        )

        if stopped:
            print(f"Early stopping at epoch {epoch+1}")
            history["early_stopped_epoch"] = epoch
            break

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_metrics = run_epoch(model, test_loader, criterion, optimizer=None, device=device, train=False)

    torch.save(
        {
            "epoch": last_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_acc,
            "history": history,
            "early_stopping": {
                "best_loss": early_stop.best_loss,
                "counter": early_stop.counter,
                "stopped_epoch": early_stop.stopped_epoch,
            },
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "completed": True,
        },
        state_path,
    )

    writer.close()

    result = {
        "config": {
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "optimizer": cfg.optimizer,
            "momentum": cfg.momentum,
            "label_smoothing": cfg.label_smoothing,
            "dropout": cfg.dropout,
            "seed": cfg.seed,
            "aug_profile": args.aug_profile,
            "early_stopping_patience": cfg.early_stopping_patience,
            "early_stopping_min_delta": cfg.early_stopping_min_delta,
            "checkpoint_every": cfg.checkpoint_every,
            "resume": args.resume,
        },
        "best_val_accuracy": best_val_acc,
        "test_metrics": test_metrics,
        "checkpoint": str(best_ckpt),
        "early_stopped": history["early_stopped_epoch"] is not None,
        "early_stopped_epoch": history["early_stopped_epoch"],
    }
    save_json(result, args.out_dir / "result.json")
    save_json(history, args.out_dir / "history.json")


if __name__ == "__main__":
    main()
