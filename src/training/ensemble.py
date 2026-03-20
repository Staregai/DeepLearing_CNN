from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.metrics import macro_precision_recall
from src.utils.io import save_json


def _predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probs = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs.append(torch.softmax(logits, dim=1).cpu())
            targets.append(y)
    return torch.cat(probs), torch.cat(targets)


def evaluate_soft_voting(models: list[nn.Module], loader: DataLoader, device: torch.device, out_file: Path | None = None):
    all_probs = []
    targets = None

    for model in models:
        probs, y = _predict_proba(model, loader, device)
        all_probs.append(probs)
        if targets is None:
            targets = y

    mean_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    preds = mean_probs.argmax(dim=1)

    acc = (preds == targets).float().mean().item()
    precision, recall = macro_precision_recall(preds, targets, num_classes=mean_probs.size(1))

    result = {"accuracy": acc, "precision": precision, "recall": recall}
    if out_file is not None:
        save_json(result, out_file)
    return result
