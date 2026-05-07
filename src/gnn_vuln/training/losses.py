"""Loss functions for vulnerability detection training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ── Focal loss ────────────────────────────────────────────────────────────────

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Focal loss for multiclass classification.
    gamma=0 reduces to standard cross-entropy.
    """
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    p_t = torch.exp(-ce)
    return ((1.0 - p_t) ** gamma * ce).mean()


# ── LIVABLE epoch-adaptive class weights ──────────────────────────────────────

def livable_weights(
    counts: torch.Tensor,
    epoch: int,
    total_epochs: int,
    num_classes: int,
    device: torch.device,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """
    Epoch-adaptive inverse-frequency weights (arXiv:2306.06935).
    w_i(t) = (N / (K * n_i)) ^ (t / T)
    Ramps from uniform at epoch=0 to full inverse-frequency at epoch=total_epochs.
    """
    t_ratio = epoch / total_epochs
    base = counts.sum().float() / (num_classes * counts.float().clamp(min=1))
    return torch.clamp(base ** t_ratio, max=max_weight).to(device)


# ── MIL losses ────────────────────────────────────────────────────────────────

def mil_loss(
    stmt_scores_list: list[torch.Tensor],
    labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    MIL binary cross-entropy for statement-level localisation (binary stmt head).
    Top-k statements per function:
      benign (label=0)      → pseudo-label 0
      vulnerable (label>0)  → pseudo-label 1
    """
    device = labels.device
    total = torch.tensor(0.0, device=device)
    count = 0
    for scores, label in zip(stmt_scores_list, labels):
        if len(scores) == 0:
            continue
        actual_k = min(k, len(scores))
        _, topk_idx = scores.topk(actual_k)
        topk_scores = scores[topk_idx]
        binary_label = float(label.item() > 0)
        pseudo = torch.full((actual_k,), binary_label, device=device)
        total = total + F.binary_cross_entropy_with_logits(topk_scores, pseudo)
        count += 1
    return total / count if count > 0 else total


def mil_loss_multiclass(
    stmt_scores_list: list[torch.Tensor],
    labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    MIL multiclass cross-entropy for [n_stmts, num_classes] stmt heads (lmgat_mcs).
    Per function of class c, select top-k statements by score[:, c] and apply CE.
    """
    device = labels.device
    total = torch.tensor(0.0, device=device)
    count = 0
    for scores, label in zip(stmt_scores_list, labels):
        if scores.shape[0] == 0:
            continue
        c = int(label.item())
        actual_k = min(k, scores.shape[0])
        _, topk_idx = scores[:, c].topk(actual_k)
        topk_scores = scores[topk_idx]          # [k, num_classes]
        targets = torch.full((actual_k,), c, dtype=torch.long, device=device)
        total = total + F.cross_entropy(topk_scores, targets)
        count += 1
    return total / count if count > 0 else total


# ── Ranking loss ──────────────────────────────────────────────────────────────

def ranking_loss(
    stmt_scores_list: list[torch.Tensor],
    batch_idx: torch.Tensor,
    node_line: torch.Tensor,
    flaw_line_mask: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Margin ranking loss: flaw lines must score higher than non-flaw by ≥ margin.
    Only applies to vulnerable functions (label > 0) with binary stmt heads.
    """
    device = labels.device
    total = torch.tensor(0.0, device=device)
    count = 0
    for b, (scores, label) in enumerate(zip(stmt_scores_list, labels)):
        if label.item() == 0 or len(scores) == 0:
            continue
        mask = batch_idx == b
        lines_b = node_line[mask]
        flaw_b  = flaw_line_mask[mask]
        valid = lines_b >= 0
        if not valid.any():
            continue
        lines_b = lines_b[valid]
        flaw_b  = flaw_b[valid]
        unique_lines = lines_b.unique(sorted=True)
        flaw_flags = torch.stack([
            flaw_b[lines_b == line].max() for line in unique_lines
        ]).bool()
        if not flaw_flags.any() or flaw_flags.all():
            continue
        flaw_scores = scores[flaw_flags]
        safe_scores = scores[~flaw_flags]
        diff = flaw_scores.unsqueeze(1) - safe_scores.unsqueeze(0)
        total = total + F.relu(margin - diff).mean()
        count += 1
    return total / count if count > 0 else total
