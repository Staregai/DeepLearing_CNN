from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def build_class_prototypes(
    encoder: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    max_batches: int | None = None,
) -> torch.Tensor:
    encoder.eval()
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.long)

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        emb = encoder(x).cpu()

        if sums is None:
            sums = torch.zeros((num_classes, emb.size(1)))

        for cls in range(num_classes):
            mask = y == cls
            if mask.any():
                sums[cls] += emb[mask].sum(dim=0)
                counts[cls] += int(mask.sum().item())

    counts = counts.clamp(min=1).unsqueeze(1)
    return sums / counts


class PrototypicalClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, prototypes: torch.Tensor):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("prototypes", prototypes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        distances = torch.cdist(emb, self.prototypes)
        return -distances
