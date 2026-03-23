from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    data_dir: Path
    outputs_dir: Path


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 5000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    momentum: float = 0.9
    label_smoothing: float = 0.0
    dropout: float = 0.0
    num_workers: int = 4
    seed: int = 42


@dataclass
class FewShotConfig:
    n_way: int = 5
    n_support: int = 5
    n_query: int = 15
    episodes_per_epoch: int = 100
    epochs: int = 5000
    learning_rate: float = 1e-3
    seed: int = 42
