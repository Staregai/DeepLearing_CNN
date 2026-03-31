from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cinic10 import load_cinic10_datasets
from src.models.cnn_baseline import BaselineCNN
from src.models.efficientnet import build_efficientnet
from src.models.protonet import ProtoEncoder
from src.models.prototypical_classifier import PrototypicalClassifier, build_class_prototypes
from src.training.ensemble import evaluate_soft_voting
from src.utils.device import get_device
from src.utils.io import save_json


def _load_cnn(ckpt: Path, device: torch.device) -> BaselineCNN:
    model = BaselineCNN(num_classes=10, dropout=0.3)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    return model


def _load_effnet(ckpt: Path, device: torch.device):
    model = build_efficientnet(num_classes=10, dropout=0.2)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    return model


def _load_protonet_classifier(encoder_ckpt: Path, train_loader: DataLoader, device: torch.device):
    encoder = ProtoEncoder().to(device)
    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    prototypes = build_class_prototypes(encoder, train_loader, num_classes=10, device=device, max_batches=200)
    clf = PrototypicalClassifier(encoder, prototypes.to(device)).to(device)
    return clf


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate soft-voting ensemble")
    parser.add_argument("--data-dir", type=Path, default=Path("src/dataset"))
    parser.add_argument("--cnn-ckpt", type=Path, required=True)
    parser.add_argument("--effnet-ckpt", type=Path, required=True)
    parser.add_argument("--fewshot-ckpt", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, default=Path("outputs/ensemble/result.json"))
    args = parser.parse_args()

    device = get_device()

    train_ds, _, test_ds = load_cinic10_datasets(args.data_dir, train_aug_profile="baseline")
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    cnn = _load_cnn(args.cnn_ckpt, device)
    effnet = _load_effnet(args.effnet_ckpt, device)
    fewshot = _load_protonet_classifier(args.fewshot_ckpt, train_loader, device)

    result = evaluate_soft_voting([cnn, effnet, fewshot], test_loader, device, out_file=args.out_file)
    save_json(result, args.out_file)


if __name__ == "__main__":
    main()
