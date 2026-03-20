from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import TrainConfig
from src.training.metrics import macro_precision_recall
from src.utils.io import ensure_dir, save_json


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
        logits = model(images)
        loss = criterion(logits, targets)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
) -> dict:
    ensure_dir(output_dir)
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = _build_optimizer(model, cfg)

    model.to(device)
    best_val_acc = -1.0
    best_ckpt = output_dir / "best.pt"
    history = {"train": [], "val": []}

    for epoch in tqdm(range(cfg.epochs), desc="Epochs", leave=True):
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

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_metrics = run_epoch(model, test_loader, criterion, optimizer=None, device=device, train=False)

    writer.close()

    result = {
        "config": asdict(cfg),
        "best_val_accuracy": best_val_acc,
        "test_metrics": test_metrics,
        "checkpoint": str(best_ckpt),
    }
    save_json(result, output_dir / "result.json")
    save_json(history, output_dir / "history.json")
    return result
