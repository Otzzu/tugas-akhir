"""Serving-side config surface — for callers that own their own versioning (the SAST API).

The research path composes YAML freely: ~230 model knobs exist, most of them ablation-only
(supcon_*, jepa_*, window_*, stage2_*), and Config._from_raw setattr's ANY key without
validation — a typo silently trains a different architecture.

This module closes that door. The three production architectures are frozen in ARCH_PROFILES
exactly as Bab III specifies them (GATv2 + GNN+ block, 31-d edge features, JK-Net readout).
A caller may only override what is legitimately tunable — model sizing and training knobs.
Anything else raises, instead of quietly changing the model.

    from gnn_vuln.serving import build_config
    cfg = build_config("graph_based",
                       data={"source": "megavul_26", "storage": "lazy"},
                       train={"epochs": 50, "device": "cuda"})
"""
from __future__ import annotations

from dataclasses import fields
from typing import Any

from gnn_vuln.config import Config, EWCConfig, ReplayConfig

# Design decisions (Bab III.2). Locked: a caller cannot override these.
_SHARED_DESIGN: dict[str, Any] = {
    "pretrained_lm": "microsoft/unixcoder-base",
    "func_lm": "microsoft/unixcoder-base",
    "add_func_tokens": True,
    "edge_dim": 31,                 # CPG edge feature dim
    "add_self_loops": False,
    "use_skip": True,               # GNN+ residual
    "gnn_block_style": "gnn_plus",
    "gnn_norm_type": "batch",
    "gnn_activation": "elu",
    "gnn_use_ffn": True,
    "gnn_ffn_expansion": 2,
    "func_head_type": "linear",
    "mil_weight": 0.3,              # weakly-supervised localization
    "mil_k": 3,
    "rank_loss_weight": 0.2,
}

ARCH_PROFILES: dict[str, dict[str, Any]] = {
    "graph_based": {
        **_SHARED_DESIGN,
        "architecture": "lmgat_codebert",
        "live_lm": "none",
        "func_max_length": 1024,
        "localization_encoder": "gnn",
        "graph_pool": "jknet",
        "hidden_dim": 256,
        "num_layers": 4,
        "dropout": 0.3,
        "heads": 4,
        "num_classes": 26,
    },
    "hybrid_graph_lm": {
        **_SHARED_DESIGN,
        "architecture": "lmgat_codebert",
        "func_max_length": 5120,     # sliding window over the function
        "func_chunk_size": 1024,
        "func_chunk_stride": 1024,
        "window_attn_pool": False,
        "window_mixer": True,
        "window_mixer_max": 6,
        "cross_window_attn": False,
        "localization_encoder": "both",
        "stmt_both_mode": "concat",
        "cross_task_method": "none",
        "graph_pool": "jknet",
        "hidden_dim": 768,
        "num_layers": 4,
        "dropout": 0.3,
        "heads": 4,
        "num_classes": 26,
    },
    "sequential": {
        **_SHARED_DESIGN,
        "architecture": "lmgat_seqgnn",
        "live_lm": "none",
        "func_max_length": 1024,
        "localization_encoder": "gnn",
        "jknet_mode": "concat",
        "jknet_readout": "meanmax",
        "jknet_loc": False,
        "seq_stage2_input": "raw",
        "seq_susp_pool": False,
        "seq_susp_pool_k": 4.0,
        "seq_detach_susp": False,
        "hidden_dim": 256,
        "num_layers": 4,
        "dropout": 0.3,
        "heads": 4,
        "num_classes": 26,
    },
}

# Model sizing — may vary without changing the architecture described in Bab III.
TUNABLE_MODEL = frozenset({"hidden_dim", "num_layers", "dropout", "heads", "num_classes"})

TUNABLE_TRAIN = frozenset({
    "seed", "epochs", "batch_size", "lr", "lm_lr", "warmup_ratio", "grad_clip",
    "weight_decay", "patience", "early_stop_metric", "use_class_weights",
    "label_smoothing", "lr_scheduler", "epoch_adaptive_weights", "device",
    "grad_accum_steps", "num_workers", "prefetch_factor", "save_last_every",
    "use_amp", "use_flash_attention", "compile_model", "stmt_head_vectorized",
})

# Dataset identity + role datasets. The caller owns these (it owns the dataset versioning).
TUNABLE_DATA = frozenset({
    "source", "mode", "max_nodes", "top_cwe", "filter_top25_dangerous", "filter_owasp",
    "cwe_list", "cwe_groups", "max_per_class", "resample_seed", "train_ratio", "val_ratio",
    "split_seed", "storage", "ds_name_suffix", "target_vocab",
    "source_val", "source_test", "source_val_params", "source_test_params",
})


def _reject(section: str, keys, allowed: frozenset, locked: frozenset = frozenset()) -> None:
    bad_locked = sorted(k for k in keys if k in locked)
    if bad_locked:
        raise ValueError(
            f"{section}: {bad_locked} are design decisions of this architecture (Bab III) "
            f"and cannot be overridden. Change the profile in gnn_vuln.serving, not the caller."
        )
    unknown = sorted(k for k in keys if k not in allowed)
    if unknown:
        raise ValueError(f"{section}: unknown key(s) {unknown}. Allowed: {sorted(allowed)}")


def build_config(
    arch: str,
    *,
    data: dict | None = None,
    model: dict | None = None,
    train: dict | None = None,
    ewc: dict | None = None,
    replay: dict | None = None,
) -> Config:
    """Compose a Config from a frozen architecture profile plus tunable overrides."""
    if arch not in ARCH_PROFILES:
        raise ValueError(f"unknown arch {arch!r}. Known: {sorted(ARCH_PROFILES)}")

    profile = dict(ARCH_PROFILES[arch])
    locked = frozenset(profile) - TUNABLE_MODEL

    _reject("model", (model or {}), TUNABLE_MODEL, locked)
    _reject("data", (data or {}), TUNABLE_DATA)
    _reject("train", (train or {}), TUNABLE_TRAIN)
    _reject("ewc", (ewc or {}), frozenset(f.name for f in fields(EWCConfig)))
    _reject("replay", (replay or {}), frozenset(f.name for f in fields(ReplayConfig)))

    profile.update(model or {})
    return Config._from_raw({
        "data": data or {},
        "model": profile,
        "train": train or {},
        "ewc": ewc or {},
        "replay": replay or {},
    })


def profile_of(arch: str) -> dict[str, Any]:
    """The frozen design values for `arch` (read-only view)."""
    return dict(ARCH_PROFILES[arch])
