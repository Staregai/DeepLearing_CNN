from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.config import FewShotConfig
from src.models.protonet import prototypical_loss
from src.utils.io import ensure_dir, save_json


def _sample_episode(dataset: ImageFolder, cfg: FewShotConfig):
    labels = [label for _, label in dataset.samples]
    class_to_indices = {}
    for idx, lab in enumerate(labels):
        class_to_indices.setdefault(lab, []).append(idx)

    selected_classes = random.sample(list(class_to_indices.keys()), cfg.n_way)

    images = []
    targets = []
    for episode_label, class_id in enumerate(selected_classes):
        indices = random.sample(class_to_indices[class_id], cfg.n_support + cfg.n_query)
        for idx in indices:
            img, _ = dataset[idx]
            images.append(img)
            targets.append(episode_label)

    batch_x = torch.stack(images, dim=0)
    batch_y = torch.tensor(targets, dtype=torch.long)
    return batch_x, batch_y


def train_fewshot(
    encoder: nn.Module,
    train_ds: ImageFolder,
    val_ds: ImageFolder,
    cfg: FewShotConfig,
    output_dir: Path,
    device: torch.device,
) -> dict:
    ensure_dir(output_dir)
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    encoder.to(device)
    optimizer = Adam(encoder.parameters(), lr=cfg.learning_rate)

    best_val_acc = -1.0
    best_ckpt = output_dir / "best.pt"

    for epoch in tqdm(range(cfg.epochs), desc="Epochs", leave=True):
        encoder.train()
        train_losses = []
        train_accs = []

        for _ in tqdm(range(cfg.episodes_per_epoch), desc=f"Epoch {epoch} Train", leave=False):
            x, y = _sample_episode(train_ds, cfg)
            x, y = x.to(device), y.to(device)
            embeddings = encoder(x)
            loss, acc = prototypical_loss(embeddings, y, cfg.n_way, cfg.n_support, cfg.n_query)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(acc.item())

        encoder.eval()
        with torch.no_grad():
            val_losses = []
            val_accs = []
            for _ in tqdm(range(max(10, cfg.episodes_per_epoch // 5)), desc=f"Epoch {epoch} Val", leave=False):
                x, y = _sample_episode(val_ds, cfg)
                x, y = x.to(device), y.to(device)
                embeddings = encoder(x)
                vloss, vacc = prototypical_loss(embeddings, y, cfg.n_way, cfg.n_support, cfg.n_query)
                val_losses.append(vloss.item())
                val_accs.append(vacc.item())

        train_loss = float(sum(train_losses) / len(train_losses))
        train_acc = float(sum(train_accs) / len(train_accs))
        val_loss = float(sum(val_losses) / len(val_losses))
        val_acc = float(sum(val_accs) / len(val_accs))

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(encoder.state_dict(), best_ckpt)

    writer.close()

    result = {
        "config": asdict(cfg),
        "best_val_accuracy": best_val_acc,
        "checkpoint": str(best_ckpt),
    }
    save_json(result, output_dir / "result.json")
    return result
