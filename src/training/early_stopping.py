from __future__ import annotations

from pathlib import Path


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        checkpoint_path: Path | None = None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0
        self.stopped_epoch = 0

    def __call__(self, val_loss: float) -> bool:
       
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = max(0, self.counter - self.patience)
                return True
            return False

    def get_improvement(self, val_loss: float) -> float:
        return self.best_loss - val_loss
