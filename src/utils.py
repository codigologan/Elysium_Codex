from __future__ import annotations
import os, json, random
from dataclasses import dataclass, asdict
import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@dataclass
class TrainConfig:
    run_name: str = "lad_mlp"
    seed: int = 42
    epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    input_dim: int = 20
    hidden_dim: int = 128
    num_classes: int = 2
    train_size: int = 20000
    val_size: int = 5000
    log_every: int = 50
    save_best: bool = True
    early_stopping: bool = False
    patience: int = 2

    @staticmethod
    def from_json(path: str) -> "TrainConfig":
        data = load_json(path)
        return TrainConfig(**data)

    def to_dict(self) -> dict:
        return asdict(self)
