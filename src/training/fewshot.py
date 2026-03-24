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
from src.training.metrics import macro_precision_recall
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
    checkpoint_every: int = 5,
    resume_state: Path | None = None,
) -> dict:
    ensure_dir(output_dir)
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    encoder.to(device)
    optimizer = Adam(encoder.parameters(), lr=cfg.learning_rate)

    best_val_acc = -1.0
    best_ckpt = output_dir / "best.pt"
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_precision": [],
        "train_recall": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
    }
    start_epoch = 0
    state_path = output_dir / "train_state.pt"

    if resume_state is not None and resume_state.exists():
        state = torch.load(resume_state, map_location=device)
        encoder.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        best_val_acc = float(state.get("best_val_accuracy", -1.0))
        history = state.get("history", history)
        history.setdefault("train_precision", [])
        history.setdefault("train_recall", [])
        history.setdefault("val_precision", [])
        history.setdefault("val_recall", [])
        start_epoch = int(state.get("epoch", -1)) + 1

        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state_all" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

    for epoch in tqdm(range(start_epoch, cfg.epochs), desc="Epochs", leave=True):
        encoder.train()
        train_losses = []
        train_accs = []
        train_precisions = []
        train_recalls = []

        for _ in tqdm(range(cfg.episodes_per_epoch), desc=f"Epoch {epoch} Train", leave=False):
            x, y = _sample_episode(train_ds, cfg)
            x, y = x.to(device), y.to(device)
            embeddings = encoder(x)
            loss, acc, preds, targets = prototypical_loss(embeddings, y, cfg.n_way, cfg.n_support, cfg.n_query)
            precision, recall = macro_precision_recall(preds.detach().cpu(), targets.detach().cpu(), cfg.n_way)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(acc.item())
            train_precisions.append(precision)
            train_recalls.append(recall)

        encoder.eval()
        with torch.no_grad():
            val_losses = []
            val_accs = []
            val_precisions = []
            val_recalls = []
            for _ in tqdm(range(max(10, cfg.episodes_per_epoch // 5)), desc=f"Epoch {epoch} Val", leave=False):
                x, y = _sample_episode(val_ds, cfg)
                x, y = x.to(device), y.to(device)
                embeddings = encoder(x)
                vloss, vacc, vpreds, vtargets = prototypical_loss(embeddings, y, cfg.n_way, cfg.n_support, cfg.n_query)
                vprecision, vrecall = macro_precision_recall(vpreds.detach().cpu(), vtargets.detach().cpu(), cfg.n_way)
                val_losses.append(vloss.item())
                val_accs.append(vacc.item())
                val_precisions.append(vprecision)
                val_recalls.append(vrecall)

        train_loss = float(sum(train_losses) / len(train_losses))
        train_acc = float(sum(train_accs) / len(train_accs))
        train_precision = float(sum(train_precisions) / len(train_precisions))
        train_recall = float(sum(train_recalls) / len(train_recalls))
        val_loss = float(sum(val_losses) / len(val_losses))
        val_acc = float(sum(val_accs) / len(val_accs))
        val_precision = float(sum(val_precisions) / len(val_precisions))
        val_recall = float(sum(val_recalls) / len(val_recalls))

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("train/precision", train_precision, epoch)
        writer.add_scalar("train/recall", train_recall, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("val/precision", val_precision, epoch)
        writer.add_scalar("val/recall", val_recall, epoch)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_precision"].append(train_precision)
        history["train_recall"].append(train_recall)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(encoder.state_dict(), best_ckpt)

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            torch.save(encoder.state_dict(), output_dir / f"epoch_{epoch + 1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_val_acc,
                    "history": history,
                    "config": asdict(cfg),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                state_path,
            )

    writer.close()

    torch.save(
        {
            "epoch": cfg.epochs - 1,
            "model_state_dict": encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_acc,
            "history": history,
            "config": asdict(cfg),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "completed": True,
        },
        state_path,
    )

    result = {
        "config": asdict(cfg),
        "best_val_accuracy": best_val_acc,
        "checkpoint": str(best_ckpt),
        "state": str(state_path),
    }
    save_json(result, output_dir / "result.json")
    return result
