from __future__ import annotations

import torch


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (preds == targets).float().mean().item()


def macro_precision_recall(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> tuple[float, float]:
    precision_scores = []
    recall_scores = []

    for cls in range(num_classes):
        tp = ((preds == cls) & (targets == cls)).sum().item()
        fp = ((preds == cls) & (targets != cls)).sum().item()
        fn = ((preds != cls) & (targets == cls)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_scores.append(precision)
        recall_scores.append(recall)

    return float(sum(precision_scores) / num_classes), float(sum(recall_scores) / num_classes)
