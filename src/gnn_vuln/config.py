"""
Central configuration for the gnn_vuln project.

Edit `configs/default.yaml` to override defaults without changing code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    raw_dir: Path = DATA_DIR / "raw"
    processed_dir: Path = DATA_DIR / "processed"
    splits_dir: Path = DATA_DIR / "splits"
    # Graph building
    max_nodes: int = 500        # drop graphs larger than this
    node_feat_dim: int = 100    # embedding dimension per node
    edge_types: list[str] = field(
        default_factory=lambda: ["AST", "CFG", "PDG", "CDG", "DDG"]
    )


@dataclass
class ModelConfig:
    architecture: str = "gcn"   # one of: gcn | gat | devign
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.3
    heads: int = 4              # only used by GAT
    num_classes: int = 2        # 2 = binary; set higher for multi-class


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10          # early stopping patience
    checkpoint_dir: Path = CHECKPOINT_DIR
    results_dir: Path = RESULTS_DIR
    log_dir: Path = LOG_DIR
    device: str = "cpu"         # set to "cuda" if GPU available


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from a YAML file, merging with defaults."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls()
        if "data" in raw:
            for k, v in raw["data"].items():
                setattr(cfg.data, k, v)
        if "model" in raw:
            for k, v in raw["model"].items():
                setattr(cfg.model, k, v)
        if "train" in raw:
            for k, v in raw["train"].items():
                setattr(cfg.train, k, v)
        return cfg


def load_default_config() -> Config:
    default_yaml = PROJECT_ROOT / "configs" / "default.yaml"
    if default_yaml.exists():
        return Config.from_yaml(default_yaml)
    return Config()
