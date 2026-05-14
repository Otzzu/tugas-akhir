"""Shared output head modules for vulnerability detectors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_ALPHA_MAX = 0.8
_ALPHA_MEAN = 0.6


# ── Statement heads ───────────────────────────────────────────────────────────

class StmtHead(nn.Module):
    """
    Per-statement binary scorer.
    Groups CPG nodes by source line, max/mean-pools per line,
    returns list of [n_stmts_i] scalar tensors (one per graph in batch).

    localization_encoder controls feature source for scoring:
      "gnn"  — GNN node features only (default, no LM needed)
      "lm"   — LM token hidden states only (requires lm_dim > 0)
      "both" — concat GNN + LM (requires lm_dim > 0)
    """

    def __init__(self, hidden_dim: int, lm_dim: int = 0,
                 localization_encoder: str = "gnn"):
        super().__init__()
        assert localization_encoder in ("gnn", "lm", "both"), \
            f"localization_encoder must be gnn|lm|both, got {localization_encoder!r}"
        self._mode = localization_encoder
        if localization_encoder == "gnn":
            in_dim = hidden_dim
        elif localization_encoder == "lm":
            in_dim = lm_dim
        else:  # both
            in_dim = hidden_dim + lm_dim
        self.max_head  = nn.Linear(in_dim, 1)
        self.mean_head = nn.Linear(in_dim, 1)

    @staticmethod
    def _pool_lm_per_line(
        lm_hidden: torch.Tensor,
        func_token_lines: torch.Tensor,
        target_lines: torch.Tensor,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool LM last_hidden_state tokens by source line.
        lm_hidden: [L, lm_dim], func_token_lines: [L] (1-indexed, -1=special).
        Returns (max_per_line, mean_per_line) each [n_lines, lm_dim]."""
        max_res, mean_res = [], []
        lm_dim = lm_hidden.size(-1)
        for line in target_lines:
            mask = func_token_lines == line
            if mask.any():
                h_l = lm_hidden[mask]
                max_res.append(h_l.max(dim=0).values)
                mean_res.append(h_l.mean(dim=0))
            else:
                max_res.append(torch.zeros(lm_dim, device=device, dtype=lm_hidden.dtype))
                mean_res.append(torch.zeros(lm_dim, device=device, dtype=lm_hidden.dtype))
        return torch.stack(max_res), torch.stack(mean_res)

    def score(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor,
        lm_hidden: torch.Tensor | None = None,
        func_token_lines: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if getattr(self, "_vectorized", False):
            return self._score_vectorized(h, batch, node_line, lm_hidden, func_token_lines)
        return self._score_loop(h, batch, node_line, lm_hidden, func_token_lines)

    def _score_loop(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor,
        lm_hidden: torch.Tensor | None = None,
        func_token_lines: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        device = h.device
        B = int(batch.max().item()) + 1
        result: list[torch.Tensor] = []
        for b in range(B):
            mask = batch == b
            h_b, lines_b = h[mask], node_line[mask]
            valid = lines_b >= 0
            if not valid.any():
                result.append(torch.zeros(0, device=device))
                continue
            h_b, lines_b = h_b[valid], lines_b[valid]
            unique_lines = lines_b.unique(sorted=True)

            lm_max_emb = lm_mean_emb = None
            if self._mode != "gnn" and lm_hidden is not None and func_token_lines is not None:
                lh  = lm_hidden[b]
                ftl = func_token_lines[b]
                lm_max_emb, lm_mean_emb = self._pool_lm_per_line(lh, ftl, unique_lines, device)

            scores: list[torch.Tensor] = []
            for i, line in enumerate(unique_lines):
                nm = lines_b == line
                h_l = h_b[nm]

                if self._mode == "gnn":
                    feat_max  = h_l.max(dim=0).values
                    feat_mean = h_l.mean(dim=0)
                elif self._mode == "lm":
                    feat_max  = lm_max_emb[i]  if lm_max_emb  is not None else h_l.max(dim=0).values
                    feat_mean = lm_mean_emb[i] if lm_mean_emb is not None else h_l.mean(dim=0)
                else:
                    gnn_max  = h_l.max(dim=0).values
                    gnn_mean = h_l.mean(dim=0)
                    if lm_max_emb is not None:
                        feat_max  = torch.cat([gnn_max,  lm_max_emb[i]])
                        feat_mean = torch.cat([gnn_mean, lm_mean_emb[i]])
                    else:
                        feat_max, feat_mean = gnn_max, gnn_mean

                s = _ALPHA_MAX * self.max_head(feat_max.float()) + _ALPHA_MEAN * self.mean_head(feat_mean.float())
                scores.append(s.squeeze(-1))
            result.append(torch.stack(scores))
        return result

    def _score_vectorized(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor,
        lm_hidden: torch.Tensor | None = None,
        func_token_lines: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Scatter-based vectorized scorer. Same output as _score_loop, no Python inner loop."""
        device = h.device
        D = h.shape[1]
        MAX_LINE = 100_000  # line numbers well below this in practice

        # Filter valid nodes (line >= 0)
        valid_node = node_line >= 0
        B = int(batch.max().item()) + 1
        if not valid_node.any():
            return [torch.zeros(0, device=device) for _ in range(B)]

        h_v      = h[valid_node]
        lines_v  = node_line[valid_node]
        batch_v  = batch[valid_node]

        # Unique stmt ID per (graph, line) pair
        sid = batch_v * MAX_LINE + lines_v           # [N']
        unique_sid, inv = torch.unique(sid, sorted=True, return_inverse=True)
        S = unique_sid.shape[0]

        if self._mode in ("gnn", "both"):
            # scatter max
            gnn_max = torch.full((S, D), float('-inf'), device=device, dtype=h_v.dtype)
            gnn_max.scatter_reduce_(0, inv.unsqueeze(1).expand(-1, D), h_v,
                                    reduce='amax', include_self=True)
            # scatter mean via sum+count
            gnn_sum = torch.zeros(S, D, device=device, dtype=h_v.dtype)
            cnt_gnn = torch.zeros(S, 1, device=device, dtype=h_v.dtype)
            idx_exp = inv.unsqueeze(1).expand(-1, D)
            gnn_sum.scatter_add_(0, idx_exp, h_v)
            cnt_gnn.scatter_add_(0, inv.unsqueeze(1), torch.ones(h_v.shape[0], 1, device=device, dtype=h_v.dtype))
            gnn_mean = gnn_sum / cnt_gnn.clamp(min=1)

        lm_max = lm_mean = None  # assigned below if mode requires LM

        if self._mode in ("lm", "both") and lm_hidden is not None and func_token_lines is not None:
            LM_D = lm_hidden.shape[-1]
            L    = lm_hidden.shape[1]
            # Flatten [B, L] → [B*L]
            g_tok  = torch.arange(B, device=device).unsqueeze(1).expand(-1, L).reshape(-1)
            tl_tok = func_token_lines.reshape(-1)
            lm_flat = lm_hidden.reshape(-1, LM_D)

            valid_tok = tl_tok >= 0
            if valid_tok.any():
                g_tok   = g_tok[valid_tok]
                tl_tok  = tl_tok[valid_tok]
                lm_flat = lm_flat[valid_tok]
                tsid = g_tok * MAX_LINE + tl_tok
                unique_tsid, inv_tok = torch.unique(tsid, sorted=True, return_inverse=True)
                ST = unique_tsid.shape[0]

                lm_dtype = lm_flat.dtype
                lm_max_all = torch.full((ST, LM_D), float('-inf'), device=device, dtype=lm_dtype)
                lm_max_all.scatter_reduce_(0, inv_tok.unsqueeze(1).expand(-1, LM_D),
                                           lm_flat, reduce='amax', include_self=True)
                lm_sum = torch.zeros(ST, LM_D, device=device, dtype=lm_dtype)
                cnt_tok = torch.zeros(ST, 1, device=device, dtype=lm_dtype)
                lm_sum.scatter_add_(0, inv_tok.unsqueeze(1).expand(-1, LM_D), lm_flat)
                cnt_tok.scatter_add_(0, inv_tok.unsqueeze(1),
                                     torch.ones(lm_flat.shape[0], 1, device=device, dtype=lm_dtype))
                lm_mean_all = lm_sum / cnt_tok.clamp(min=1)

                # Align LM stmts with GNN stmts (unique_sid)
                pos   = torch.searchsorted(unique_tsid, unique_sid).clamp(0, ST - 1)
                found = unique_tsid[pos] == unique_sid

                lm_max  = torch.zeros(S, LM_D, device=device, dtype=lm_dtype)
                lm_mean = torch.zeros(S, LM_D, device=device, dtype=lm_dtype)
                lm_max[found]  = lm_max_all[pos[found]]
                lm_mean[found] = lm_mean_all[pos[found]]
            else:
                LM_D = lm_hidden.shape[-1]
                lm_max = lm_mean = torch.zeros(S, LM_D, device=device, dtype=lm_hidden.dtype)

        # Build feat_max / feat_mean for linear heads
        if self._mode == "gnn":
            feat_max, feat_mean = gnn_max, gnn_mean
        elif self._mode == "lm":
            if lm_max is None:  # func_token_lines was None — fallback to zeros
                LM_D = lm_hidden.shape[-1] if lm_hidden is not None else self.max_head.in_features
                lm_max = lm_mean = torch.zeros(S, LM_D, device=device)
            feat_max, feat_mean = lm_max, lm_mean
        else:  # both
            if lm_max is None:  # func_token_lines was None — fallback to zeros
                LM_D = lm_hidden.shape[-1] if lm_hidden is not None else (self.max_head.in_features - D)
                lm_max = lm_mean = torch.zeros(S, LM_D, device=device)
            feat_max  = torch.cat([gnn_max,  lm_max],  dim=-1)
            feat_mean = torch.cat([gnn_mean, lm_mean], dim=-1)

        scores_flat = (
            _ALPHA_MAX  * self.max_head(feat_max.float())
            + _ALPHA_MEAN * self.mean_head(feat_mean.float())
        ).squeeze(-1)   # [S]

        # Split by graph — one sync (tolist) instead of B×n_lines syncs
        stmt_graph = unique_sid // MAX_LINE          # [S] graph index per stmt
        counts = torch.bincount(stmt_graph, minlength=B).tolist()
        return list(torch.split(scores_flat, counts))


class MulticlassStmtHead(nn.Module):
    """
    Per-statement multiclass scorer (used by lmgat_mcs).
    Returns list of [n_stmts_i, num_classes] tensors.
    """

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.max_head  = nn.Linear(hidden_dim, num_classes)
        self.mean_head = nn.Linear(hidden_dim, num_classes)

    def score(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor,
    ) -> list[torch.Tensor]:
        device = h.device
        B = int(batch.max().item()) + 1
        result: list[torch.Tensor] = []
        for b in range(B):
            mask = batch == b
            h_b, lines_b = h[mask], node_line[mask]
            valid = lines_b >= 0
            if not valid.any():
                result.append(torch.zeros(0, self.max_head.out_features, device=device))
                continue
            h_b, lines_b = h_b[valid], lines_b[valid]
            scores: list[torch.Tensor] = []
            for line in lines_b.unique(sorted=True):
                nm = lines_b == line
                h_l = h_b[nm]
                s = (
                    _ALPHA_MAX  * self.max_head(h_l.max(dim=0).values.float())
                    + _ALPHA_MEAN * self.mean_head(h_l.mean(dim=0).float())
                )
                scores.append(s)          # [num_classes]
            result.append(torch.stack(scores))  # [n_stmts, num_classes]
        return result


# ── Function heads ────────────────────────────────────────────────────────────

class FuncHead(nn.Module):
    """Standard function-level MLP classifier: Linear→ReLU→Dropout→Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class SmallFuncHead(nn.Module):
    """Half-width variant: Linear→ReLU→Dropout→Linear (hidden_dim → hidden_dim//2 → C)."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


# ── MTL heads ─────────────────────────────────────────────────────────────────

class MTLHeads(nn.Module):
    """
    Three-head MTL output block for lmgat_codebert_mtl and lmgat_hcdfgat:
      binary_head  → [B, 2]
      group_head   → [B, num_groups]
      cwe_head     → [B, num_classes]  (conditioned on group_probs when use_group_cond=True)

    use_group_cond: feed softmax(group_logits).detach() into cwe_head input.
    """

    def __init__(
        self,
        fused_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_groups: int,
        dropout: float,
        use_group_cond: bool = True,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.use_group_cond = use_group_cond

        self.binary_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.group_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_groups),
        )
        # CWE head input: fused + group_probs (detached)
        self.cwe_head = nn.Sequential(
            nn.Linear(fused_dim + num_groups, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(
        self,
        z: torch.Tensor,  # [B, fused_dim]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logit_cwe [B,C], logit_group [B,G], logit_binary [B,2])."""
        z = z.float()
        logit_binary = self.binary_head(z)
        logit_group  = self.group_head(z)

        if self.use_group_cond:
            group_probs = F.softmax(logit_group.detach(), dim=-1)
        else:
            group_probs = torch.zeros(z.size(0), self.num_groups, device=z.device)

        cwe_in    = torch.cat([z, group_probs], dim=-1)
        logit_cwe = self.cwe_head(cwe_in)

        return logit_cwe, logit_group, logit_binary
