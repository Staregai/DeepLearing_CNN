from __future__ import annotations

import gc
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import TrainConfig
from src.training.early_stopping import EarlyStopping
from src.training.metrics import macro_precision_recall
from src.utils.io import ensure_dir, save_json


def _as_cpu_byte_tensor(state) -> torch.Tensor:
    # torch.set_rng_state requires a CPU uint8 tensor.
    if isinstance(state, torch.Tensor):
        return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu").contiguous()


def _build_optimizer(model: nn.Module, cfg: TrainConfig):
    if cfg.optimizer.lower() == "sgd":
        return SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "adam":
        return Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer, device: torch.device, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    all_preds = []
    all_targets = []

    desc = "Training" if train else "Validating"
    for images, targets in tqdm(loader, desc=desc, leave=False):
        images, targets = images.to(device), targets.to(device)
        
        if train:
            logits = model(images)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, targets)

        total_loss += loss.item() * images.size(0)
        all_preds.append(logits.argmax(dim=1).detach().cpu())
        all_targets.append(targets.detach().cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    avg_loss = total_loss / len(loader.dataset)
    acc = (preds == targets).float().mean().item()
    precision, recall = macro_precision_recall(preds, targets, num_classes=int(targets.max().item() + 1))

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
    }


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    output_dir: Path,
    device: torch.device,
    patience: int | None = None,
    min_delta: float | None = None,
    checkpoint_every: int | None = None,
    resume_state: Path | None = None,
) -> dict:
    ensure_dir(output_dir)
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = _build_optimizer(model, cfg)
    effective_patience = cfg.early_stopping_patience if patience is None else patience
    effective_min_delta = cfg.early_stopping_min_delta if min_delta is None else min_delta
    effective_checkpoint_every = cfg.checkpoint_every if checkpoint_every is None else checkpoint_every
    early_stopping = EarlyStopping(
        patience=effective_patience,
        min_delta=effective_min_delta,
        checkpoint_path=output_dir / "best.pt",
    )

    model.to(device)
    best_val_acc = -1.0
    best_ckpt = output_dir / "best.pt"
    history = {"train": [], "val": []}
    start_epoch = 0
    last_epoch = -1
    state_path = output_dir / "train_state.pt"

    if resume_state is not None and resume_state.exists():
        state = torch.load(resume_state, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        best_val_acc = float(state.get("best_val_accuracy", -1.0))
        history = state.get("history", history)
        saved_next_epoch = int(state.get("epoch", -1)) + 1
        history_next_epoch = len(history.get("train", []))
        start_epoch = history_next_epoch if history_next_epoch > 0 else saved_next_epoch

        es_state = state.get("early_stopping", {})
        early_stopping.best_loss = float(es_state.get("best_loss", early_stopping.best_loss))
        early_stopping.counter = int(es_state.get("counter", early_stopping.counter))
        early_stopping.stopped_epoch = int(es_state.get("stopped_epoch", early_stopping.stopped_epoch))

        if "torch_rng_state" in state and state["torch_rng_state"] is not None:
            torch.set_rng_state(_as_cpu_byte_tensor(state["torch_rng_state"]))
        if torch.cuda.is_available() and "cuda_rng_state_all" in state and state["cuda_rng_state_all"] is not None:
            cuda_states = state["cuda_rng_state_all"]
            if isinstance(cuda_states, (list, tuple)):
                torch.cuda.set_rng_state_all([_as_cpu_byte_tensor(s) for s in cuda_states])
            else:
                torch.cuda.set_rng_state_all([_as_cpu_byte_tensor(cuda_states)])

    for epoch in tqdm(range(start_epoch, cfg.epochs), desc="Epochs", leave=True):
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

        if effective_checkpoint_every > 0 and (epoch + 1) % effective_checkpoint_every == 0:
            torch.save(model.state_dict(), output_dir / f"epoch_{epoch + 1}.pt")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_val_acc,
                    "history": history,
                    "config": asdict(cfg),
                    "early_stopping": {
                        "best_loss": early_stopping.best_loss,
                        "counter": early_stopping.counter,
                        "stopped_epoch": early_stopping.stopped_epoch,
                    },
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                state_path,
            )

        # Early stopping
        if early_stopping(val_metrics["loss"]):
            tqdm.write(f"Early stopping at epoch {epoch + 1}")
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
            "config": asdict(cfg),
            "early_stopping": {
                "best_loss": early_stopping.best_loss,
                "counter": early_stopping.counter,
                "stopped_epoch": early_stopping.stopped_epoch,
            },
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "completed": True,
        },
        state_path,
    )

    writer.close()

    # Cleanup to prevent memory leaks
    model.cpu()
    del criterion, optimizer, writer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "best_val_accuracy": best_val_acc,
        "test_metrics": test_metrics,
        "history": history,
    }


