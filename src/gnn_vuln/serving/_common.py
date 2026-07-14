"""Bagian bersama kontrak serving: data + train params, dan desain yang sama untuk ketiga
arsitektur (Bab III.2)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SHARED_DESIGN: dict[str, Any] = {   # Bab III — locked
    "pretrained_lm": "microsoft/unixcoder-base-nine",
    "func_lm": "microsoft/unixcoder-base-nine",
    "add_func_tokens": True,
    "edge_dim": 31,
    "add_self_loops": False,
    "use_skip": True,
    "gnn_block_style": "gnn_plus",
    "gnn_norm_type": "batch",
    "gnn_activation": "elu",
    "gnn_use_ffn": True,
    "gnn_ffn_expansion": 2,
    "func_head_type": "linear",
    "mil_weight": 0.3,
    "mil_k": 3,
    "rank_loss_weight": 0.2,
}


@dataclass
class DataParams:
    source: str
    ds_name: str = ""             # caller-owned .pt name; empty = derive from params
    mode: str = "multiclass"
    storage: str = "lazy"
    max_nodes: int = 2500
    # Build every class in the uploaded rows. Narrowing is explicit and auditable (cwe_list);
    # the research presets (filter_owasp, filter_top25_dangerous, top_cwe frequency cut,
    # max_per_class balancing) are benchmark-shaping knobs and are NOT part of this contract.
    cwe_list: list | None = None
    resample_seed: int = 42
    train_ratio: float = 0.9      # production split: no test holdout, val drives early stopping
    val_ratio: float = 0.1
    split_seed: int | None = None
    ds_name_suffix: str = ""
    target_vocab: dict[str, int] | None = None
    source_val: str = ""
    source_test: str = ""
    source_val_params: dict | None = None
    source_test_params: dict | None = None


@dataclass
class TrainParams:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    lm_lr: float = 1e-5           # only read by the hybrid's LM branch
    seed: int = 42
    device: str = "cuda"
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    weight_decay: float = 1e-3
    patience: int = 15
    early_stop_metric: str = "f1"
    use_class_weights: bool = True
    epoch_adaptive_weights: bool = True
    label_smoothing: float = 0.1
    lr_scheduler: str = "cosine"
    grad_accum_steps: int = 2
    save_last_every: int = 10
    num_workers: int = 0
    prefetch_factor: int = 2
    stmt_head_vectorized: bool = True
    use_amp: bool = False
    use_flash_attention: bool = False
    compile_model: bool = False


def to_raw(obj) -> dict:
    return asdict(obj) if obj is not None else {}
