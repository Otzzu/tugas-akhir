"""cross_task.py — Phase 2 bidirectional cross-task modules for lmgat_codebert.

Makes localization (stmt_head) and classification (func_head) inform each other.
Cross-signals are detached: each head trains on its own loss, the cross path
shares information forward only — no gradient loop between the two tasks.

All modules are localization-encoder aware. The localization "view" handed to
the classifier (loc_proto, attention K/V) uses the same encoder(s) the
stmt_head scores on:
  gnn  → GNN node features          (dim = hidden)
  lm   → LM token hidden states     (dim = lm_dim)
  both → GNN + LM concatenated      (dim = hidden + lm_dim)

Three methods (config `cross_task_method`):
  direct    — scalar conditioning (stmt suspicion ↔ classifier logit / score gate)
  film      — FiLM: cls embedding modulates stmt features, loc proto modulates fused
  attention — cross-task attention, encoder units as K/V, swapped task queries
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


def loc_dim(mode: str, hidden_dim: int, lm_dim: int) -> int:
    """Dimension of the localization-view representation for a given mode."""
    if mode == "gnn":
        return hidden_dim
    if mode == "lm":
        return lm_dim
    return hidden_dim + lm_dim   # both


def loc_proto_pool(
    h: torch.Tensor,
    batch: torch.Tensor,
    node_line: torch.Tensor,
    lm_hidden: torch.Tensor | None,
    func_token_lines: torch.Tensor | None,
    mode: str,
    B: int,
) -> torch.Tensor:
    """Per-graph pooled localization-view representation → [B, loc_dim(mode)].

    Pools the same encoder(s) the stmt_head scores on. Independent of stmt_head
    output, so it can feed the classifier without a circular dependency.
    """
    device = h.device
    parts: list[torch.Tensor] = []

    if mode in ("gnn", "both"):
        D = h.shape[1]
        proto = torch.zeros(B, D, device=device, dtype=h.dtype)
        valid = node_line >= 0
        if valid.any():
            hv, bv = h[valid], batch[valid]
            cnt = torch.zeros(B, 1, device=device, dtype=h.dtype)
            proto.scatter_add_(0, bv.unsqueeze(1).expand(-1, D), hv)
            cnt.scatter_add_(0, bv.unsqueeze(1),
                             torch.ones(hv.shape[0], 1, device=device, dtype=h.dtype))
            proto = proto / cnt.clamp(min=1)
        parts.append(proto)

    if mode in ("lm", "both"):
        LM_D = lm_hidden.shape[-1]
        if func_token_lines is not None:
            vmask = (func_token_lines >= 0).unsqueeze(-1).to(lm_hidden.dtype)   # [B,L,1]
        else:
            vmask = torch.ones(*lm_hidden.shape[:2], 1, device=device, dtype=lm_hidden.dtype)
        cnt = vmask.sum(dim=1).clamp(min=1)                                    # [B,1]
        proto_lm = (lm_hidden * vmask).sum(dim=1) / cnt                        # [B,LM_D]
        parts.append(proto_lm)

    return torch.cat(parts, dim=-1)


class DirectCrossTask(nn.Module):
    """B2 — scalar bidirectional conditioning.

    loc→cls: per-graph stmt suspicion summary → per-class logit bias.
    cls→loc: vuln confidence (1 − P(benign)) → additive gate on stmt scores.
    Mode-agnostic: operates on prediction outputs, not encoder features.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.stmt_to_logit = nn.Linear(1, num_classes)
        self.alpha = nn.Parameter(torch.zeros(1))   # cls→loc strength
        self.beta  = nn.Parameter(torch.zeros(1))   # loc→cls strength

    def cls_from_loc(self, logit_base: torch.Tensor,
                     stmt_summary: torch.Tensor) -> torch.Tensor:
        return logit_base + self.beta * self.stmt_to_logit(stmt_summary)

    def loc_from_cls(self, stmt_scores_list: list[torch.Tensor],
                     vuln_conf: torch.Tensor) -> list[torch.Tensor]:
        return [s + self.alpha * vuln_conf[b] for b, s in enumerate(stmt_scores_list)]


class FiLMCrossTask(nn.Module):
    """B3 — FiLM modulation.

    cls→loc: fused embedding → additive stmt-feature conditioning [B, loc_dim].
    loc→cls: loc proto → (γ, β) modulate fused [B, fused_dim].
    """

    def __init__(self, fused_dim: int, hidden_dim: int, lm_dim: int, mode: str):
        super().__init__()
        ld = loc_dim(mode, hidden_dim, lm_dim)
        self.cls_to_loc = nn.Linear(fused_dim, ld)
        self.loc_to_cls = nn.Linear(ld, 2 * fused_dim)
        self._fused_dim = fused_dim

    def forward(self, fused: torch.Tensor,
                loc_proto: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """fused [B, fused_dim], loc_proto [B, loc_dim] (detached by caller).

        Returns (fused_mod [B, fused_dim], stmt_cond [B, loc_dim]).
        """
        stmt_cond = self.cls_to_loc(fused.detach())
        gb = self.loc_to_cls(loc_proto)
        gamma, beta = gb[:, :self._fused_dim], gb[:, self._fused_dim:]
        fused_mod = fused * (1.0 + gamma) + beta
        return fused_mod, stmt_cond


class CrossTaskAttn(nn.Module):
    """B4 — cross-task attention. Encoder units = K/V, task protos = swapped queries.

    One attention per encoder source (GNN nodes, LM tokens) per the localization
    mode. cls_from_loc / loc_from_cls results are concatenated to dim loc_dim.
    """

    def __init__(self, fused_dim: int, hidden_dim: int, lm_dim: int,
                 mode: str, num_heads: int = 4):
        super().__init__()
        self._mode = mode
        ld = loc_dim(mode, hidden_dim, lm_dim)
        if mode in ("gnn", "both"):
            self.q_loc_g = nn.Linear(ld, hidden_dim)
            self.q_cls_g = nn.Linear(fused_dim, hidden_dim)
            self.attn_g  = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        if mode in ("lm", "both"):
            self.q_loc_l = nn.Linear(ld, lm_dim)
            self.q_cls_l = nn.Linear(fused_dim, lm_dim)
            self.attn_l  = nn.MultiheadAttention(lm_dim, num_heads, batch_first=True)
        self.to_cls = nn.Linear(ld, fused_dim)

    def forward(self, fused: torch.Tensor, loc_proto: torch.Tensor,
                h: torch.Tensor, batch: torch.Tensor, B: int,
                lm_hidden: torch.Tensor | None,
                func_token_lines: torch.Tensor | None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (fused_mod [B, fused_dim], stmt_cond [B, loc_dim])."""
        cls_parts: list[torch.Tensor] = []
        loc_parts: list[torch.Tensor] = []
        loc_q = loc_proto.detach()
        cls_q = fused.detach()

        if self._mode in ("gnn", "both"):
            kv, mask = to_dense_batch(h, batch, batch_size=B)        # [B,Nmax,H]
            key_pad = ~mask
            cls_g, _ = self.attn_g(self.q_loc_g(loc_q).unsqueeze(1), kv, kv,
                                   key_padding_mask=key_pad)
            loc_g, _ = self.attn_g(self.q_cls_g(cls_q).unsqueeze(1), kv, kv,
                                   key_padding_mask=key_pad)
            cls_parts.append(cls_g.squeeze(1))
            loc_parts.append(loc_g.squeeze(1))

        if self._mode in ("lm", "both"):
            kv = lm_hidden                                          # [B,L,LM_D]
            if func_token_lines is not None:
                key_pad = func_token_lines < 0                      # True = pad/special
                # guard: a row with no valid token would NaN — keep at least one key
                all_pad = key_pad.all(dim=1)
                if all_pad.any():
                    key_pad = key_pad.clone()
                    key_pad[all_pad, 0] = False
            else:
                key_pad = None
            cls_l, _ = self.attn_l(self.q_loc_l(loc_q).unsqueeze(1), kv, kv,
                                   key_padding_mask=key_pad)
            loc_l, _ = self.attn_l(self.q_cls_l(cls_q).unsqueeze(1), kv, kv,
                                   key_padding_mask=key_pad)
            cls_parts.append(cls_l.squeeze(1))
            loc_parts.append(loc_l.squeeze(1))

        cls_from_loc = torch.cat(cls_parts, dim=-1)                 # [B, loc_dim]
        stmt_cond    = torch.cat(loc_parts, dim=-1)                 # [B, loc_dim]
        fused_mod = fused + self.to_cls(cls_from_loc)
        return fused_mod, stmt_cond


class SelfAttnCrossTask(nn.Module):
    """B4-alt — EDAT-style cross-task self-attention.

    Self-attention over a task's own encoder units; the query is biased by the
    OTHER task's signal:  Q = units + bias(other),  K = V = units.
    Refined units are masked-mean pooled per graph. Mirrors EDAT's task-aware
    attention fusion (query bias) rather than decoder-style cross-attention.
    """

    def __init__(self, fused_dim: int, hidden_dim: int, lm_dim: int,
                 mode: str, num_heads: int = 4):
        super().__init__()
        self._mode = mode
        ld = loc_dim(mode, hidden_dim, lm_dim)
        if mode in ("gnn", "both"):
            self.bias_loc_g = nn.Linear(ld, hidden_dim)         # loc signal → cls-dir bias
            self.bias_cls_g = nn.Linear(fused_dim, hidden_dim)  # cls signal → loc-dir bias
            self.attn_g = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        if mode in ("lm", "both"):
            self.bias_loc_l = nn.Linear(ld, lm_dim)
            self.bias_cls_l = nn.Linear(fused_dim, lm_dim)
            self.attn_l = nn.MultiheadAttention(lm_dim, num_heads, batch_first=True)
        self.to_cls = nn.Linear(ld, fused_dim)

    @staticmethod
    def _masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x [B,N,D], valid [B,N,1] → [B,D]."""
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

    def forward(self, fused: torch.Tensor, loc_proto: torch.Tensor,
                h: torch.Tensor, batch: torch.Tensor, B: int,
                lm_hidden: torch.Tensor | None,
                func_token_lines: torch.Tensor | None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (fused_mod [B, fused_dim], stmt_cond [B, loc_dim])."""
        cls_parts: list[torch.Tensor] = []
        loc_parts: list[torch.Tensor] = []
        loc_sig = loc_proto.detach()
        cls_sig = fused.detach()

        if self._mode in ("gnn", "both"):
            kv, mask = to_dense_batch(h, batch, batch_size=B)        # [B,N,H]
            key_pad = ~mask
            mf = mask.unsqueeze(-1).to(kv.dtype)
            # cls direction: own units, query biased by loc signal
            q_c = kv + self.bias_loc_g(loc_sig).unsqueeze(1)
            ref_c, _ = self.attn_g(q_c, kv, kv, key_padding_mask=key_pad)
            cls_parts.append(self._masked_mean(ref_c, mf))
            # loc direction: own units, query biased by cls signal
            q_l = kv + self.bias_cls_g(cls_sig).unsqueeze(1)
            ref_l, _ = self.attn_g(q_l, kv, kv, key_padding_mask=key_pad)
            loc_parts.append(self._masked_mean(ref_l, mf))

        if self._mode in ("lm", "both"):
            kv = lm_hidden                                          # [B,L,LM_D]
            if func_token_lines is not None:
                key_pad = func_token_lines < 0
                all_pad = key_pad.all(dim=1)
                if all_pad.any():
                    key_pad = key_pad.clone()
                    key_pad[all_pad, 0] = False
                valid = (~key_pad).unsqueeze(-1).to(kv.dtype)
            else:
                key_pad = None
                valid = torch.ones(*kv.shape[:2], 1, device=kv.device, dtype=kv.dtype)
            q_c = kv + self.bias_loc_l(loc_sig).unsqueeze(1)
            ref_c, _ = self.attn_l(q_c, kv, kv, key_padding_mask=key_pad)
            cls_parts.append(self._masked_mean(ref_c, valid))
            q_l = kv + self.bias_cls_l(cls_sig).unsqueeze(1)
            ref_l, _ = self.attn_l(q_l, kv, kv, key_padding_mask=key_pad)
            loc_parts.append(self._masked_mean(ref_l, valid))

        cls_from_loc = torch.cat(cls_parts, dim=-1)                 # [B, loc_dim]
        stmt_cond    = torch.cat(loc_parts, dim=-1)                 # [B, loc_dim]
        fused_mod = fused + self.to_cls(cls_from_loc)
        return fused_mod, stmt_cond


def build_cross_task(method: str, fused_dim: int, hidden_dim: int,
                     num_classes: int, lm_dim: int, localization_encoder: str,
                     num_heads: int = 4) -> nn.Module | None:
    """Factory — returns the cross-task module for `method`, or None for 'none'."""
    if method == "none":
        return None
    if method == "direct":
        return DirectCrossTask(num_classes)
    if method == "film":
        return FiLMCrossTask(fused_dim, hidden_dim, lm_dim, localization_encoder)
    if method == "cross_attention":
        return CrossTaskAttn(fused_dim, hidden_dim, lm_dim, localization_encoder, num_heads)
    if method == "self_attention":
        return SelfAttnCrossTask(fused_dim, hidden_dim, lm_dim, localization_encoder, num_heads)
    raise ValueError(
        "cross_task_method must be none|direct|film|cross_attention|self_attention, "
        f"got {method!r}"
    )
