"""
Central configuration for the gnn_vuln project.

Edit configs under configs/<model>/ to override defaults without changing code.
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
    edge_types: list[str] = field(
        default_factory=lambda: ["AST", "CFG", "PDG", "CDG", "DDG"]
    )
    # Dataset mode: "binary" (benign/vuln) or "multiclass" (per-CWE).
    # Controls which processed cache file is used so both can coexist.
    mode: str = "binary"


@dataclass
class ModelConfig:
    architecture: str = "lmgcn"  # lmgcn | lmgat | lmgat_ft | lmgat_mc
    pretrained_lm: str = "microsoft/codebert-base"  # HuggingFace model ID for node embeddings
    add_func_tokens: bool = False  # tokenize full function text → stored in Data for live CodeBERT
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.3
    heads: int = 4              # number of GAT attention heads (lmgat* only)
    edge_dim: int = 7          # edge feature dimension injected into GATv2 attention (lmgat* only)
    num_classes: int = 2        # 2 = binary; set higher for multi-class
    # Statement-level MIL head
    mil_weight: float = 0.5     # λ: weight of stmt MIL loss vs function loss
    mil_k: int = 3              # top-k statements used for pseudo-label assignment
    rank_loss_weight: float = 0.0  # pairwise ranking loss weight (0 = disabled)


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    lm_lr: float = 2e-5         # CodeBERT learning rate for lmgat_ft / lmgat_mc
    warmup_ratio: float = 0.0   # fraction of total steps for linear warmup (0 = disabled)
    grad_clip: float = 0.0      # gradient clipping max norm (0 = disabled)
    weight_decay: float = 1e-4
    patience: int = 10          # early stopping patience
    checkpoint_dir: Path = CHECKPOINT_DIR
    results_dir: Path = RESULTS_DIR
    log_dir: Path = LOG_DIR
    device: str = "cpu"         # set to "cuda" if GPU available
    use_class_weights: bool = True  # inverse-frequency weighting for imbalanced classes
    focal_loss_gamma: float = 0.0  # focal loss gamma; 0 = standard CE, 2.0 recommended for imbalanced


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from a YAML file, merging with defaults."""
        with open(path, encoding="utf-8") as f:
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
    default_yaml = PROJECT_ROOT / "configs" / "lmgcn" / "binary.yaml"
    if default_yaml.exists():
        return Config.from_yaml(default_yaml)
    return Config()
