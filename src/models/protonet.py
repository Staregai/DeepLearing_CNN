from __future__ import annotations

import torch
from torch import nn


class ProtoEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, embedding_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, embedding_dim, 3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def prototypical_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    n_way: int,
    n_support: int,
    n_query: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    classes = torch.unique(labels)
    if len(classes) != n_way:
        raise ValueError("Episode labels do not match n_way")

    support_embeddings = []
    query_embeddings = []
    query_labels = []

    for c_idx, c in enumerate(classes):
        class_mask = labels == c
        class_embeddings = embeddings[class_mask]
        support = class_embeddings[:n_support]
        query = class_embeddings[n_support : n_support + n_query]
        support_embeddings.append(support.mean(dim=0))
        query_embeddings.append(query)
        query_labels.append(torch.full((query.size(0),), c_idx, device=labels.device, dtype=torch.long))

    prototypes = torch.stack(support_embeddings, dim=0)
    queries = torch.cat(query_embeddings, dim=0)
    targets = torch.cat(query_labels, dim=0)

    distances = torch.cdist(queries, prototypes)
    logits = -distances
    loss = nn.CrossEntropyLoss()(logits, targets)
    preds = logits.argmax(dim=1)
    acc = (preds == targets).float().mean()
    return loss, acc
