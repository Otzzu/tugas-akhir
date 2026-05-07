"""Optimizer and LR scheduler factory."""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from gnn_vuln.config import Config
from gnn_vuln.models.base import VulnDetectorBase

# Architectures that fine-tune an LM branch with a separate (lower) lr
_FT_ARCHS = frozenset({
    "lmgat_codebert", "lmgat_mcs", "lmgat_seq", "lmgat_waves_seq",
    "lmggnn", "lmgat_codebert_mtl", "lmgat_dualflow", "lmgat_hcdfgat",
    "lmrgcn",
})


def build_optimizer_and_scheduler(
    model: nn.Module,
    cfg: Config,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, object, bool]:
    """
    Build optimizer + LR scheduler for the given model and config.

    Returns
    -------
    (optimizer, scheduler, step_scheduler_per_batch)

    step_scheduler_per_batch=True  → linear warmup; call scheduler.step() after
                                     each batch inside train_epoch().
    step_scheduler_per_batch=False → ReduceLROnPlateau; call once per epoch.
    """
    arch = cfg.model.architecture.lower()
    is_ft = arch in _FT_ARCHS

    if is_ft:
        lm_lr        = getattr(cfg.train, "lm_lr", 2e-5)
        warmup_ratio = getattr(cfg.train, "warmup_ratio", 0.1)

        # Use base class method when available; fall back to attribute check
        if isinstance(model, VulnDetectorBase):
            lm_params    = model.lm_parameters()
            lm_param_ids = {id(p) for p in lm_params}
        else:
            lm_param_ids = {id(p) for p in model.codebert.parameters()}
            lm_params    = [p for p in model.parameters() if id(p) in lm_param_ids]

        other_params = [p for p in model.parameters() if id(p) not in {id(p) for p in lm_params}]

        optimizer = torch.optim.AdamW([
            {"params": lm_params,    "lr": lm_lr,        "weight_decay": 0.01},
            {"params": other_params, "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay},
        ])

        warmup_steps = max(1, int(total_steps * warmup_ratio))
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        step_per_batch = True

        logger.info(
            f"AdamW: LM lr={lm_lr:.1e}  GNN lr={cfg.train.lr:.1e} | "
            f"linear warmup {warmup_steps}/{total_steps} steps"
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        step_per_batch = False

    return optimizer, scheduler, step_per_batch
