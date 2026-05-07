"""
train.py — Training entry point.

Usage (via uv):
    uv run train --config configs/lmgcn/binary.yaml
    uv run train --config configs/lmgat/binary.yaml
    uv run train --config configs/lmgat_codebert/multiclass.yaml
    uv run train --config configs/lmgat_mcs/multiclass.yaml
"""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from loguru import logger
from tqdm import tqdm

from gnn_vuln.config import Config, load_default_config
from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset
from gnn_vuln.models.lmgcn import LMGCNVulnDetector
from gnn_vuln.models.lmgat import LMGATVulnDetector
from gnn_vuln.models.lmgat_codebert import LMGATCodeBERTVulnDetector
from gnn_vuln.models.lmgat_mcs import LMGATMCSVulnDetector
from gnn_vuln.models.lmgin import LMGINVulnDetector
from gnn_vuln.models.lmgat_interp import LMGATInterpVulnDetector
from gnn_vuln.models.lmgat_seq import LMGATSeqVulnDetector
from gnn_vuln.models.lmggnn import LMGNNVulnDetector
from gnn_vuln.models.lmgat_waves_seq import LMGATWavesSeqVulnDetector
from gnn_vuln.models.lmgat_dualflow import LMGATDualFlowVulnDetector
from gnn_vuln.models.lmgat_codebert_mtl import LMGATCodeBERTMTLVulnDetector
from gnn_vuln.models.lmgat_hcdfgat import LMGATHCDFGATVulnDetector
from gnn_vuln.losses import HierarchicalSupConLoss
from gnn_vuln.utils import (
    set_seed, setup_logging, get_device,
    save_checkpoint, load_resume_checkpoint, save_resume_checkpoint,
)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: Config, in_channels: int) -> nn.Module:
    arch = cfg.model.architecture.lower()
    pretrained_lm = getattr(cfg.model, "pretrained_lm", "microsoft/codebert-base")
    func_lm = getattr(cfg.model, "func_lm", "") or pretrained_lm

    _add_self_loops = getattr(cfg.model, "add_self_loops", False)
    _use_skip = getattr(cfg.model, "use_skip", False)
    _matryoshka_dim = getattr(cfg.model, "matryoshka_dim", None)

    if arch == "lmgcn":
        return LMGCNVulnDetector(
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            add_self_loops=_add_self_loops,
            use_skip=_use_skip,
        )
    if arch == "lmgat":
        return LMGATVulnDetector(
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            num_heads=cfg.model.heads,
            edge_dim=getattr(cfg.model, "edge_dim", 7),
            add_self_loops=_add_self_loops,
            use_skip=_use_skip,
        )
    if arch == "lmgat_codebert":
        return LMGATCodeBERTVulnDetector(
            pretrained_lm=pretrained_lm,
            func_lm=func_lm,
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            num_heads=cfg.model.heads,
            edge_dim=getattr(cfg.model, "edge_dim", 7),
            add_self_loops=_add_self_loops,
            use_skip=_use_skip,
            matryoshka_dim=_matryoshka_dim,
        )
    if arch == "lmgat_mcs":
        return LMGATMCSVulnDetector(
            pretrained_lm=pretrained_lm,
            func_lm=func_lm,
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            num_heads=cfg.model.heads,
            edge_dim=getattr(cfg.model, "edge_dim", 7),
            add_self_loops=_add_self_loops,
            use_skip=_use_skip,
            matryoshka_dim=_matryoshka_dim,
        )
    if arch == "lmgin":
        return LMGINVulnDetector(
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            edge_dim=getattr(cfg.model, "edge_dim", 7),
            use_skip=_use_skip,
        )
    if arch == "lmgat_interp":
        return LMGATInterpVulnDetector(
            pretrained_lm=pretrained_lm,
            func_lm=func_lm,
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            num_heads=cfg.model.heads,
            edge_dim=getattr(cfg.model, "edge_dim", 7),
            init_lambda=getattr(cfg.model, "init_lambda", 0.5),
            add_self_loops=_add_self_loops,
            use_skip=_use_skip,
            matryoshka_dim=_matryoshka_dim,
        )
    if arch == "lmgat_seq":
        return LMGATSeqVulnDetector(
            pretrained_lm=pretrained_lm,
            func_lm=func_lm,
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            num_heads=cfg.model.heads,
            edge_dim=getattr(cfg.model, "edge_dim", 7),
            stage2_node_input=getattr(cfg.model, "stage2_node_input", "raw"),
            add_self_loops=_add_self_loops,
            use_skip=_use_skip,
            matryoshka_dim=_matryoshka_dim,
        )
    if arch == "lmggnn":
        return LMGNNVulnDetector(
            pretrained_lm=pretrained_lm,
            func_lm=func_lm,
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            use_skip=_use_skip,
            matryoshka_dim=_matryoshka_dim,
            alpha=getattr(cfg.model, "alpha", 0.1),
        )
    if arch == "lmgat_waves_seq":
        return LMGATWavesSeqVulnDetector(
            pretrained_lm=pretrained_lm,
            func_lm=func_lm,
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            num_heads=cfg.model.heads,
            edge_dim=getattr(cfg.model, "edge_dim", 7),
            stmt_transformer_layers=getattr(cfg.model, "stmt_transformer_layers", 2),
            stmt_transformer_heads=getattr(cfg.model, "stmt_transformer_heads", 4),
            add_self_loops=_add_self_loops,
            use_skip=_use_skip,
            matryoshka_dim=_matryoshka_dim,
        )
    if arch == "lmgat_dualflow":
        return LMGATDualFlowVulnDetector(
            pretrained_lm=pretrained_lm,
            func_lm=func_lm,
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            num_heads=cfg.model.heads,
            edge_dim=getattr(cfg.model, "edge_dim", 7),
            add_self_loops=_add_self_loops,
            use_skip=_use_skip,
            matryoshka_dim=_matryoshka_dim,
        )
    if arch == "lmgat_codebert_mtl":
        return LMGATCodeBERTMTLVulnDetector(
            pretrained_lm=pretrained_lm,
            func_lm=func_lm,
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            num_groups=getattr(cfg.model, "num_groups", 16),
            num_heads=cfg.model.heads,
            edge_dim=getattr(cfg.model, "edge_dim", 7),
            use_group_cond=getattr(cfg.model, "use_group_cond", True),
            add_self_loops=_add_self_loops,
            use_skip=_use_skip,
            use_edge_emb=getattr(cfg.model, "use_edge_emb", False),
            edge_emb_dim=getattr(cfg.model, "edge_emb_dim", 32),
            edge_coarse_dim=getattr(cfg.model, "edge_coarse_dim", 16),
            matryoshka_dim=_matryoshka_dim,
        )
    if arch == "lmgat_hcdfgat":
        return LMGATHCDFGATVulnDetector(
            pretrained_lm=pretrained_lm,
            func_lm=func_lm,
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            num_groups=getattr(cfg.model, "num_groups", 16),
            num_heads=cfg.model.heads,
            edge_dim=getattr(cfg.model, "edge_dim", 7),
            use_group_cond=getattr(cfg.model, "use_group_cond", True),
            add_self_loops=_add_self_loops,
            use_skip=_use_skip,
            matryoshka_dim=_matryoshka_dim,
        )
    raise ValueError(
        f"Unknown architecture: {arch!r}. "
        "Available: lmgcn, lmgat, lmgat_codebert, lmgat_codebert_mtl, lmgat_mcs, "
        "lmgin, lmgat_interp, lmgat_seq, lmgat_waves_seq, lmggnn, "
        "lmgat_dualflow, lmgat_hcdfgat"
    )


# ---------------------------------------------------------------------------
# Optimizer / scheduler factory
# ---------------------------------------------------------------------------

def build_optimizer_and_scheduler(
    model: nn.Module,
    cfg: Config,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, object, bool]:
    """
    Build optimizer and LR scheduler appropriate for the architecture.

    Returns (optimizer, scheduler, step_scheduler_per_batch).
    step_scheduler_per_batch=True  → linear warmup, must call scheduler.step()
                                     inside train_one_epoch after each batch.
    step_scheduler_per_batch=False → ReduceLROnPlateau, call once per epoch
                                     after validation.
    """
    arch = cfg.model.architecture.lower()
    is_ft_arch = arch in ("lmgat_codebert", "lmgat_mcs", "lmgat_seq", "lmgat_waves_seq", "lmggnn", "lmgat_codebert_mtl", "lmgat_dualflow", "lmgat_hcdfgat")

    if is_ft_arch:
        lm_lr = getattr(cfg.train, "lm_lr", 2e-5)
        warmup_ratio = getattr(cfg.train, "warmup_ratio", 0.1)

        lm_param_ids = {id(p) for p in model.codebert.parameters()}
        lm_params = [p for p in model.parameters() if id(p) in lm_param_ids]
        other_params = [p for p in model.parameters() if id(p) not in lm_param_ids]

        optimizer = torch.optim.AdamW([
            {"params": lm_params,    "lr": lm_lr,          "weight_decay": 0.01},
            {"params": other_params, "lr": cfg.train.lr,   "weight_decay": cfg.train.weight_decay},
        ])

        warmup_steps = max(1, int(total_steps * warmup_ratio))
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        step_per_batch = True

        logger.info(
            f"AdamW: CodeBERT lr={lm_lr:.1e}  GNN lr={cfg.train.lr:.1e} | "
            f"linear warmup {warmup_steps}/{total_steps} steps"
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        step_per_batch = False

    return optimizer, scheduler, step_per_batch


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Focal loss for multiclass classification.
    gamma=0 reduces to standard cross-entropy.
    Combines with class weights the same way F.cross_entropy does.
    """
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    # p_t = probability assigned to the correct class
    p_t = torch.exp(-ce)
    return ((1.0 - p_t) ** gamma * ce).mean()


# ---------------------------------------------------------------------------
# LIVABLE epoch-adaptive class weights (arXiv:2306.06935)
# ---------------------------------------------------------------------------

def livable_weights(
    counts: torch.Tensor,
    epoch: int,
    total_epochs: int,
    num_classes: int,
    device: torch.device,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """
    w_i(t) = (N / (K * n_i)) ^ (t / T)
    Ramps from uniform (epoch 1) to full inverse-frequency (epoch total_epochs).
    """
    t_ratio = epoch / total_epochs
    base = counts.sum().float() / (num_classes * counts.float().clamp(min=1))
    return torch.clamp(base ** t_ratio, max=max_weight).to(device)


# ---------------------------------------------------------------------------
# MIL loss — binary statement head (Architectures 1, 2, 3)
# ---------------------------------------------------------------------------

def mil_loss(
    stmt_scores_list: list[torch.Tensor],
    labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    MIL binary cross-entropy for statement-level localisation.

    Each stmt produces a scalar logit. Top-k stmts per function:
      benign (label=0)  → pseudo-label 0
      vulnerable (label>0) → pseudo-label 1
    """
    device = labels.device
    total = torch.tensor(0.0, device=device)
    count = 0

    for scores, label in zip(stmt_scores_list, labels):
        if len(scores) == 0:
            continue
        n = len(scores)
        actual_k = min(k, n)
        _, topk_idx = scores.topk(actual_k)
        topk_scores = scores[topk_idx]
        binary_label = float(label.item() > 0)
        pseudo = torch.full((actual_k,), binary_label, device=device)
        total = total + F.binary_cross_entropy_with_logits(topk_scores, pseudo)
        count += 1

    return total / count if count > 0 else total


# ---------------------------------------------------------------------------
# MIL loss — multiclass statement head (Architecture 4)
# ---------------------------------------------------------------------------

def mil_loss_multiclass(
    stmt_scores_list: list[torch.Tensor],
    labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    MIL multiclass cross-entropy for Architecture 4's [n_stmts, num_classes] head.

    For a function of class c, select top-k statements ranked by their score
    for class c, then apply CE loss pushing them toward class c.
    """
    device = labels.device
    total = torch.tensor(0.0, device=device)
    count = 0

    for scores, label in zip(stmt_scores_list, labels):
        if scores.shape[0] == 0:
            continue
        c = int(label.item())
        n = scores.shape[0]
        actual_k = min(k, n)

        # Select top-k by score for this function's class
        class_scores = scores[:, c]
        _, topk_idx = class_scores.topk(actual_k)
        topk_scores = scores[topk_idx]  # [k, num_classes]

        targets = torch.full((actual_k,), c, dtype=torch.long, device=device)
        total = total + F.cross_entropy(topk_scores, targets)
        count += 1

    return total / count if count > 0 else total


# ---------------------------------------------------------------------------
# Pairwise ranking loss for statement localisation
# ---------------------------------------------------------------------------

def ranking_loss(
    stmt_scores_list: list[torch.Tensor],
    batch_idx: torch.Tensor,
    node_line: torch.Tensor,
    flaw_line_mask: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Margin ranking loss: flaw statements must score higher than non-flaw
    statements by at least `margin`. Only used with binary stmt heads (Arch 1-3).
    """
    device = labels.device
    total = torch.tensor(0.0, device=device)
    count = 0

    for b, (scores, label) in enumerate(zip(stmt_scores_list, labels)):
        if label.item() == 0 or len(scores) == 0:
            continue

        mask = batch_idx == b
        lines_b = node_line[mask]
        flaw_b = flaw_line_mask[mask]

        valid = lines_b >= 0
        if not valid.any():
            continue
        lines_b = lines_b[valid]
        flaw_b = flaw_b[valid]

        unique_lines = lines_b.unique(sorted=True)
        flaw_flags = torch.stack([
            (flaw_b[lines_b == line]).max() for line in unique_lines
        ]).bool()

        if not flaw_flags.any() or flaw_flags.all():
            continue

        flaw_scores = scores[flaw_flags]
        safe_scores = scores[~flaw_flags]
        diff = flaw_scores.unsqueeze(1) - safe_scores.unsqueeze(0)
        loss_b = F.relu(margin - diff).mean()
        total = total + loss_b
        count += 1

    return total / count if count > 0 else total


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def _forward(
    model: nn.Module,
    batch,
    mil_k: int,
    mil_weight: float,
    class_weight: torch.Tensor | None = None,
    rank_loss_weight: float = 0.0,
    focal_gamma: float = 0.0,
    group_loss_weight: float = 0.0,
    binary_loss_weight: float = 0.0,
    supcon_fn: nn.Module | None = None,
    supcon_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single forward pass returning (logit_func, total_loss).

    Handles:
      - Standard 2-tuple  (logit, stmt_scores)                                   Arch1/2/4/5/6/8/9
      - SupCon 3-tuple    (logit, stmt_scores, z)                                Arch3/7/10
      - MTL 4-tuple       (logit_cwe, logit_group, logit_binary, stmt_scores)    (unused — kept for compat)
      - MTL+SupCon 5-tuple (logit_cwe, logit_group, logit_binary, stmt_scores, z) Arch11/12
    """
    node_line = getattr(batch, "node_line", None)
    edge_attr = getattr(batch, "edge_attr", None)

    if hasattr(model, "codebert"):
        func_input_ids = getattr(batch, "func_input_ids", None)
        func_attention_mask = getattr(batch, "func_attention_mask", None)
        out = model(
            batch.x, batch.edge_index, batch.batch, node_line, edge_attr,
            func_input_ids, func_attention_mask,
        )
    else:
        out = model(
            batch.x, batch.edge_index, batch.batch, node_line, edge_attr
        )

    # Dispatch on return-tuple length
    if len(out) == 5:
        logit_func, logit_group, logit_binary, stmt_scores, z_combined = out
    elif len(out) == 4:
        logit_func, logit_group, logit_binary, stmt_scores = out
        z_combined = None
    elif len(out) == 3:
        logit_func, stmt_scores, z_combined = out
        logit_group = logit_binary = None
    else:
        logit_func, stmt_scores = out
        logit_group = logit_binary = z_combined = None

    if focal_gamma > 0.0:
        loss = focal_loss(logit_func, batch.y, gamma=focal_gamma, weight=class_weight)
    else:
        loss = F.cross_entropy(logit_func, batch.y, weight=class_weight)

    # MTL auxiliary losses
    if logit_group is not None and group_loss_weight > 0.0:
        group_labels = getattr(batch, "group_id", None)
        if group_labels is not None:
            loss = loss + group_loss_weight * F.cross_entropy(logit_group, group_labels)

    if logit_binary is not None and binary_loss_weight > 0.0:
        binary_labels = (batch.y > 0).long()
        loss = loss + binary_loss_weight * F.cross_entropy(logit_binary, binary_labels)

    if stmt_scores is not None and mil_weight > 0.0:
        # Detect multiclass stmt head: each element is [n_stmts, num_classes]
        is_mc_stmt = (
            len(stmt_scores) > 0
            and stmt_scores[0].dim() == 2
        )
        if is_mc_stmt:
            loss = loss + mil_weight * mil_loss_multiclass(stmt_scores, batch.y, mil_k)
        else:
            loss = loss + mil_weight * mil_loss(stmt_scores, batch.y, mil_k)

    # Ranking loss: only for binary stmt heads (1D scores)
    if (
        stmt_scores is not None
        and rank_loss_weight > 0.0
        and node_line is not None
        and (len(stmt_scores) == 0 or stmt_scores[0].dim() == 1)
    ):
        flaw_mask = getattr(batch, "flaw_line_mask", None)
        if flaw_mask is not None:
            rl = ranking_loss(
                stmt_scores, batch.batch, node_line, flaw_mask, batch.y
            )
            loss = loss + rank_loss_weight * rl

    # Hierarchical SupCon on Z_combined (HC-DFGAT only)
    if z_combined is not None and supcon_fn is not None and supcon_weight > 0.0:
        group_ids = getattr(batch, "group_id", None)
        if group_ids is not None:
            # cwe_id stores the raw CWE vocab index (-1 for benign/unknown),
            # used by the loss to look up continuous tree-distance weights.
            cwe_vocab_ids = getattr(batch, "cwe_id", None)
            sc_loss = supcon_fn(z_combined, batch.y, group_ids, cwe_vocab_ids)
            loss = loss + supcon_weight * sc_loss

    return logit_func, loss


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mil_k: int,
    mil_weight: float,
    rank_loss_weight: float,
    class_weight: torch.Tensor | None,
    epoch: int,
    total_epochs: int,
    grad_clip: float = 0.0,
    batch_scheduler=None,
    focal_gamma: float = 0.0,
    group_loss_weight: float = 0.0,
    binary_loss_weight: float = 0.0,
    supcon_fn: nn.Module | None = None,
    supcon_weight: float = 0.0,
    scaler: GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None and device.type == "cuda"
    pbar = tqdm(loader, desc=f"  Train {epoch:03d}/{total_epochs}", unit="batch", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            _, loss = _forward(model, batch, mil_k, mil_weight, class_weight, rank_loss_weight, focal_gamma, group_loss_weight, binary_loss_weight, supcon_fn, supcon_weight)
        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if batch_scheduler is not None:
            batch_scheduler.step()
        total_loss += loss.item() * batch.num_graphs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mil_k: int,
    mil_weight: float,
    rank_loss_weight: float = 0.0,
    class_weight: torch.Tensor | None = None,
    is_binary: bool = True,
    focal_gamma: float = 0.0,
    group_loss_weight: float = 0.0,
    binary_loss_weight: float = 0.0,
    supcon_fn: nn.Module | None = None,
    supcon_weight: float = 0.0,
) -> tuple[float, float, float, float, float]:
    """Return (loss, accuracy, mean_confidence, f1_macro, f1_weighted)."""
    model.eval()
    total_loss = 0.0
    total_conf = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    for batch in loader:
        batch = batch.to(device)
        logits, loss = _forward(model, batch, mil_k, mil_weight, class_weight, rank_loss_weight, focal_gamma, group_loss_weight, binary_loss_weight, supcon_fn, supcon_weight)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * batch.num_graphs
        total_conf += probs.max(dim=-1).values.sum().item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())
    n = len(loader.dataset)
    avg = "binary" if is_binary else "macro"
    f1_macro = f1_score(all_labels, all_preds, average=avg, zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return total_loss / n, float(acc), total_conf / n, float(f1_macro), float(f1_weighted)


# ---------------------------------------------------------------------------
# Inference: statement-level localisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def localise(
    model: nn.Module,
    data,
    device: torch.device,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Return the top-k most suspicious source lines for a single graph."""
    model.eval()
    data = data.to(device)
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    node_line = getattr(data, "node_line", None)
    func_input_ids = getattr(data, "func_input_ids", None)
    func_attention_mask = getattr(data, "func_attention_mask", None)

    if hasattr(model, "codebert"):
        fids = func_input_ids.unsqueeze(0) if func_input_ids is not None else None
        fmask = func_attention_mask.unsqueeze(0) if func_attention_mask is not None else None
        _, stmt_scores_list = model(data.x, data.edge_index, batch, node_line, None, fids, fmask)
    else:
        _, stmt_scores_list = model(data.x, data.edge_index, batch, node_line)

    if stmt_scores_list is None or len(stmt_scores_list[0]) == 0:
        return []

    scores_raw = stmt_scores_list[0]
    # Multiclass stmt head: use 1 - p_benign as vulnerability score
    if scores_raw.dim() == 2:
        scores = 1.0 - torch.softmax(scores_raw, dim=-1)[:, 0]
    else:
        scores = torch.sigmoid(scores_raw)

    valid_lines = data.node_line[data.node_line >= 0].unique(sorted=True)
    k = min(top_k, len(valid_lines))
    top_scores, top_idx = scores.topk(k)
    return [(int(valid_lines[i].item()), float(top_scores[j].item()))
            for j, i in enumerate(top_idx)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GNN vulnerability detector")
    parser.add_argument("--config", type=str, default="configs/lmgcn/binary.yaml")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from checkpoints/<arch>_last.pt if it exists.",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else load_default_config()
    set_seed(cfg.train.seed)
    setup_logging(cfg.train.log_dir)
    device = get_device(cfg.train.device)

    mil_k = getattr(cfg.model, "mil_k", 3)
    mil_weight = getattr(cfg.model, "mil_weight", 0.5)
    rank_loss_weight = getattr(cfg.model, "rank_loss_weight", 0.0)
    group_loss_weight = getattr(cfg.model, "group_loss_weight", 0.0)
    binary_loss_weight = getattr(cfg.model, "binary_loss_weight", 0.0)
    use_supcon = getattr(cfg.model, "use_supcon", False)
    supcon_weight = getattr(cfg.model, "supcon_weight", 0.1) if use_supcon else 0.0
    if use_supcon:
        # Load CWE vocab from raw dir so the matrix lookup can map vocab_idx → matrix_idx
        _cwe_vocab_path = (
            Path(getattr(cfg.data, "processed_dir", Path("data/processed"))).parent
            / "raw" / getattr(cfg.data, "source", "megavul") / "cwe_vocab.json"
        )
        _cwe_vocab: dict[str, int] | None = None
        _use_dist_matrix = getattr(cfg.model, "supcon_use_distance_matrix", False)
        _dist_matrix_path: Path | None = None
        if _use_dist_matrix:
            if _cwe_vocab_path.exists():
                import json as _json
                with open(_cwe_vocab_path, encoding="utf-8") as _f:
                    _cwe_vocab = _json.load(_f)
            _p = Path(getattr(cfg.model, "cwe_dist_matrix", "data/cwe/cwe_distance_matrix.json"))
            _dist_matrix_path = _p if _p.exists() else None
            if _dist_matrix_path is None:
                print(f"[train] supcon_use_distance_matrix=true but matrix not found at {_p}; falling back to alpha.")
        supcon_fn = HierarchicalSupConLoss(
            temperature=getattr(cfg.model, "supcon_temperature", 0.07),
            alpha=getattr(cfg.model, "supcon_alpha", 0.5),
            dist_matrix_path=_dist_matrix_path,
            cwe_vocab=_cwe_vocab,
            weight_fn=getattr(cfg.model, "supcon_weight_fn", "linear"),
            exp_scale=getattr(cfg.model, "supcon_exp_scale", 5.0),
            power=getattr(cfg.model, "supcon_power", 2.0),
            min_weight=getattr(cfg.model, "supcon_min_weight", 0.0),
        )
    else:
        supcon_fn = None
    use_class_weights = getattr(cfg.train, "use_class_weights", True)
    use_livable = getattr(cfg.train, "livable_loss", False) and use_class_weights
    grad_clip = getattr(cfg.train, "grad_clip", 0.0)
    is_binary = getattr(cfg.data, "mode", "binary") == "binary"
    focal_gamma = getattr(cfg.train, "focal_loss_gamma", 0.0)

    pretrained_lm = getattr(cfg.model, "pretrained_lm", "microsoft/codebert-base")
    func_lm = getattr(cfg.model, "func_lm", "") or pretrained_lm
    add_func_tokens = getattr(cfg.model, "add_func_tokens", False)
    func_lm_source = getattr(cfg.model, "func_lm_source", "raw")
    source_val  = getattr(cfg.data, "source_val",  "")
    source_test = getattr(cfg.data, "source_test", "")
    use_official_splits = bool(source_val and source_test)

    logger.info(
        f"Loading dataset… (pretrained_lm={pretrained_lm}, "
        f"add_func_tokens={add_func_tokens}, "
        f"splits={'official' if use_official_splits else 'internal 70/15/15'})"
    )

    dataset_kwargs = dict(
        root=str(cfg.data.processed_dir.parent),
        max_nodes=cfg.data.max_nodes,
        embedder_device=str(device),
        mode=cfg.data.mode,
        pretrained_lm=pretrained_lm,
        func_lm=func_lm,
        add_func_tokens=add_func_tokens,
        func_lm_source=func_lm_source,
        top_cwe=getattr(cfg.data, "top_cwe", 0),
        cwe_list=getattr(cfg.data, "cwe_list", None),
        cwe_groups=getattr(cfg.data, "cwe_groups", None),
        filter_owasp_top10=getattr(cfg.data, "filter_owasp_top10", False),
        filter_top25=getattr(cfg.data, "filter_top25", False),
        max_per_class=getattr(cfg.data, "max_per_class", 0),
        resample_seed=getattr(cfg.data, "resample_seed", 42),
    )
    dataset = CodeBERTGraphDataset(
        source=getattr(cfg.data, "source", "bigvul"), **dataset_kwargs
    )

    if use_official_splits:
        logger.info(f"Official splits: train={cfg.data.source}  val={source_val}  test={source_test}")
        val_dataset  = CodeBERTGraphDataset(source=source_val,  **dataset_kwargs)
        test_dataset = CodeBERTGraphDataset(source=source_test, **dataset_kwargs)
        train_idx = list(range(len(dataset)))
        train_loader = DataLoader(dataset,      batch_size=cfg.train.batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,  batch_size=cfg.train.batch_size)
        test_loader  = DataLoader(test_dataset, batch_size=cfg.train.batch_size)
    else:
        train_idx, val_idx, test_idx = dataset.get_splits(seed=cfg.train.seed)
        train_loader = DataLoader(dataset[train_idx], batch_size=cfg.train.batch_size, shuffle=True)
        val_loader   = DataLoader(dataset[val_idx],   batch_size=cfg.train.batch_size)
        test_loader  = DataLoader(dataset[test_idx],  batch_size=cfg.train.batch_size)

    in_channels = dataset[0].x.size(1)
    logger.info(f"Dataset: {len(dataset)} graphs | in_channels={in_channels}")

    if dataset.num_classes != cfg.model.num_classes:
        raise ValueError(
            f"Config model.num_classes={cfg.model.num_classes} but the dataset has "
            f"{dataset.num_classes} classes. "
            "Update model.num_classes in your config to match, "
            "or check data/raw/cwe_vocab.json."
        )

    # Class weights — static inverse-freq or LIVABLE epoch-adaptive
    train_counts: torch.Tensor | None = None
    class_weight: torch.Tensor | None = None
    if use_class_weights:
        train_labels = torch.tensor(
            [int(dataset[i].y.item()) for i in train_idx], dtype=torch.long
        )
        train_counts = torch.bincount(train_labels, minlength=cfg.model.num_classes).float()
        if use_livable:
            # LIVABLE: weights computed per epoch; start with epoch=1 to show initial
            class_weight = livable_weights(train_counts, 1, cfg.train.epochs, cfg.model.num_classes, device)
            logger.info(f"LIVABLE adaptive weights enabled (epoch 1 init): {[f'{w:.3f}' for w in class_weight.tolist()]}")
        else:
            class_weight = torch.clamp(
                train_counts.sum() / (train_counts * cfg.model.num_classes), max=10.0
            ).to(device)
            logger.info(f"Static class weights: {[f'{w:.3f}' for w in class_weight.tolist()]}")

    model = build_model(cfg, in_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loss_mode = "livable" if use_livable else ("focal" if focal_gamma > 0 else "ce")
    logger.info(
        f"Model: {cfg.model.architecture.upper()} | params={n_params:,} | "
        f"mil_weight={mil_weight} | mil_k={mil_k} | "
        f"rank_loss_weight={rank_loss_weight} | class_weights={use_class_weights} | "
        f"loss={loss_mode} | focal_gamma={focal_gamma}"
    )

    # AMP GradScaler — only active on CUDA; disabled automatically on CPU/MPS
    use_amp = device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        logger.info("AMP (automatic mixed precision) enabled — halves LM activation VRAM")

    total_steps = len(train_loader) * cfg.train.epochs
    optimizer, scheduler, step_per_batch = build_optimizer_and_scheduler(
        model, cfg, total_steps
    )

    stop_on_f1 = getattr(cfg.train, "early_stop_metric", "f1") == "f1"
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 1

    run_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{cfg.model.architecture}_{cfg.data.mode}"
    )
    run_dir = cfg.train.checkpoint_dir / run_id

    if args.resume:
        existing = sorted(
            cfg.train.checkpoint_dir.glob(
                f"*_{cfg.model.architecture}_{cfg.data.mode}/last_{cfg.model.architecture}.pt"
            )
        )
        if existing:
            last_ckpt = existing[-1]
            run_dir = last_ckpt.parent
            run_id = run_dir.name
            logger.info(f"Resuming run: {run_id}")
        else:
            logger.warning("--resume set but no previous run found — starting a new run.")
            args.resume = False

    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = run_dir / f"best_{cfg.model.architecture}.pt"
    last_ckpt = run_dir / f"last_{cfg.model.architecture}.pt"

    config_src = Path(args.config) if Path(args.config).exists() else None
    if config_src:
        shutil.copy(config_src, run_dir / "config.yaml")

    logger.info(f"Run ID: {run_id}  |  checkpoints → {run_dir}")

    if args.resume and last_ckpt.exists():
        meta = load_resume_checkpoint(last_ckpt, model, optimizer, scheduler, device=str(device))
        start_epoch = meta["epoch"] + 1
        best_val_f1 = meta.get("best_val_f1", -1.0)
        best_val_loss = meta.get("best_val_loss", float("inf"))
        patience_counter = meta["patience_counter"]
        logger.info(f"Resuming from epoch {start_epoch} (patience={patience_counter})")
    elif args.resume:
        logger.warning(f"--resume set but {last_ckpt} not found — starting from scratch.")

    train_start = time.time()
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        if use_livable and train_counts is not None:
            class_weight = livable_weights(
                train_counts, epoch, cfg.train.epochs, cfg.model.num_classes, device
            )
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            mil_k, mil_weight, rank_loss_weight, class_weight,
            epoch=epoch, total_epochs=cfg.train.epochs,
            grad_clip=grad_clip,
            batch_scheduler=scheduler if step_per_batch else None,
            focal_gamma=focal_gamma,
            group_loss_weight=group_loss_weight,
            binary_loss_weight=binary_loss_weight,
            supcon_fn=supcon_fn,
            supcon_weight=supcon_weight,
            scaler=scaler,
        )
        val_loss, val_acc, val_conf, val_f1, val_f1w = evaluate(
            model, val_loader, device, mil_k, mil_weight, rank_loss_weight, class_weight,
            is_binary=is_binary, focal_gamma=focal_gamma,
            group_loss_weight=group_loss_weight,
            binary_loss_weight=binary_loss_weight,
            supcon_fn=supcon_fn,
            supcon_weight=supcon_weight,
        )
        if not step_per_batch:
            scheduler.step(val_loss)
        elapsed = time.time() - t0

        # Show GNN LR (last param group) to track the main learning rate
        current_lr = optimizer.param_groups[-1]["lr"]
        improved = (val_f1 > best_val_f1) if stop_on_f1 else (val_loss < best_val_loss)

        logger.info(
            f"Epoch {epoch:03d}/{cfg.train.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"acc={val_acc:.4f} | f1={val_f1:.4f} | f1w={val_f1w:.4f} | "
            f"conf={val_conf:.4f} | "
            f"lr={current_lr:.2e} | patience={patience_counter}/{cfg.train.patience} | "
            f"{elapsed:.0f}s"
            + (" *" if improved else "")
        )

        if improved:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, best_ckpt,
                epoch=epoch, val_loss=val_loss, val_acc=val_acc, val_conf=val_conf,
                val_f1=val_f1, val_f1_weighted=val_f1w,
            )
        else:
            patience_counter += 1

        save_last_every = getattr(cfg.train, "save_last_every", 1)
        if save_last_every > 0 and epoch % save_last_every == 0:
            save_resume_checkpoint(
                last_ckpt, model, optimizer, scheduler,
                epoch=epoch,
                best_val_f1=best_val_f1,
                best_val_loss=best_val_loss,
                patience_counter=patience_counter,
                val_loss=val_loss, val_acc=val_acc, val_conf=val_conf,
                val_f1=val_f1, val_f1_weighted=val_f1w,
            )

        if patience_counter >= cfg.train.patience:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break

    _, test_acc, test_conf, test_f1, test_f1w = evaluate(
        model, test_loader, device, mil_k, mil_weight, rank_loss_weight, class_weight,
        is_binary=is_binary, focal_gamma=focal_gamma,
        group_loss_weight=group_loss_weight,
        binary_loss_weight=binary_loss_weight,
        supcon_fn=supcon_fn,
        supcon_weight=supcon_weight,
    )
    logger.info(
        f"Test accuracy: {test_acc:.4f} | f1: {test_f1:.4f} | "
        f"f1_weighted: {test_f1w:.4f} | confidence: {test_conf:.4f}"
    )
    total_secs = time.time() - train_start
    hours, rem = divmod(int(total_secs), 3600)
    mins, secs = divmod(rem, 60)
    logger.info(f"Total training time: {hours:02d}h {mins:02d}m {secs:02d}s ({total_secs:.1f}s)")
    logger.info(f"Best checkpoint → {best_ckpt}")
    logger.info(
        f"To evaluate:  uv run evaluate "
        f"--checkpoint {best_ckpt} --config {run_dir / 'config.yaml'}"
    )


if __name__ == "__main__":
    main()
