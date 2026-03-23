from __future__ import annotations

from pathlib import Path


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        checkpoint_path: Path | None = None,
    ):
        """
        Early stopping based on validation loss improvement.

        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum loss decrease to count as improvement
            checkpoint_path: Path to save best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0
        self.stopped_epoch = 0

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Returns:
            True if training should stop, False otherwise
        """
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
        """Get improvement score (negative means no improvement)."""
        return self.best_loss - val_loss
