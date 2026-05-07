"""Trainer class: forward pass dispatch, train loop, evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from gnn_vuln.training.losses import (
    focal_loss,
    mil_loss,
    mil_loss_multiclass,
    ranking_loss,
)


class Trainer:
    """
    Encapsulates the training and evaluation loop for all architectures.

    Handles:
      - Unified forward dispatch (2-tuple / 3-tuple / 5-tuple returns)
      - MTL auxiliary losses (group, binary, SupCon)
      - MIL and ranking localisation losses
      - AMP (automatic mixed precision)
      - Gradient clipping
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        step_per_batch: bool,
        device: torch.device,
        *,
        mil_k: int = 3,
        mil_weight: float = 0.0,
        rank_loss_weight: float = 0.0,
        focal_gamma: float = 0.0,
        group_loss_weight: float = 0.0,
        binary_loss_weight: float = 0.0,
        supcon_fn: nn.Module | None = None,
        supcon_weight: float = 0.0,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        scaler: GradScaler | None = None,
    ):
        self.model              = model
        self.optimizer          = optimizer
        self.scheduler          = scheduler
        self.step_per_batch     = step_per_batch
        self.device             = device
        self.mil_k              = mil_k
        self.mil_weight         = mil_weight
        self.rank_loss_weight   = rank_loss_weight
        self.focal_gamma        = focal_gamma
        self.group_loss_weight  = group_loss_weight
        self.binary_loss_weight = binary_loss_weight
        self.supcon_fn          = supcon_fn
        self.supcon_weight      = supcon_weight
        self.use_amp            = use_amp
        self.amp_dtype          = amp_dtype
        self.scaler             = scaler

    # ── Forward ──────────────────────────────────────────────────────────────

    def _forward(
        self,
        batch,
        class_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single forward pass → (logit_func, total_loss).

        Handles all return-tuple lengths:
          2-tuple (logit, stmt_scores)                                 — standard
          3-tuple (logit, stmt_scores, z)                              — SupCon
          5-tuple (logit_cwe, logit_group, logit_binary, stmt, z)      — MTL+SupCon
        """
        node_line  = getattr(batch, "node_line",  None)
        edge_attr  = getattr(batch, "edge_attr",  None)

        if hasattr(self.model, "codebert"):
            func_ids  = getattr(batch, "func_input_ids",      None)
            func_mask = getattr(batch, "func_attention_mask", None)
            out = self.model(
                batch.x, batch.edge_index, batch.batch,
                node_line, edge_attr, func_ids, func_mask,
            )
        else:
            out = self.model(batch.x, batch.edge_index, batch.batch, node_line, edge_attr)

        # Unpack return tuple
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

        # Primary loss
        if self.focal_gamma > 0.0:
            loss = focal_loss(logit_func, batch.y, gamma=self.focal_gamma, weight=class_weight)
        else:
            loss = F.cross_entropy(logit_func, batch.y, weight=class_weight)

        # MTL auxiliary losses
        if logit_group is not None and self.group_loss_weight > 0.0:
            group_labels = getattr(batch, "group_id", None)
            if group_labels is not None:
                loss = loss + self.group_loss_weight * F.cross_entropy(logit_group, group_labels)

        if logit_binary is not None and self.binary_loss_weight > 0.0:
            binary_labels = (batch.y > 0).long()
            loss = loss + self.binary_loss_weight * F.cross_entropy(logit_binary, binary_labels)

        # MIL localisation loss
        if stmt_scores is not None and self.mil_weight > 0.0:
            is_mc_stmt = len(stmt_scores) > 0 and stmt_scores[0].dim() == 2
            if is_mc_stmt:
                loss = loss + self.mil_weight * mil_loss_multiclass(
                    stmt_scores, batch.y, self.mil_k
                )
            else:
                loss = loss + self.mil_weight * mil_loss(
                    stmt_scores, batch.y, self.mil_k
                )

        # Ranking loss (binary stmt heads only)
        if (
            stmt_scores is not None
            and self.rank_loss_weight > 0.0
            and node_line is not None
            and (len(stmt_scores) == 0 or stmt_scores[0].dim() == 1)
        ):
            flaw_mask = getattr(batch, "flaw_line_mask", None)
            if flaw_mask is not None:
                rl = ranking_loss(
                    stmt_scores, batch.batch, node_line, flaw_mask, batch.y
                )
                loss = loss + self.rank_loss_weight * rl

        # Hierarchical SupCon (HC-DFGAT / MTL+SupCon)
        if z_combined is not None and self.supcon_fn is not None and self.supcon_weight > 0.0:
            group_ids = getattr(batch, "group_id", None)
            if group_ids is not None:
                cwe_vocab_ids = getattr(batch, "cwe_id", None)
                sc = self.supcon_fn(z_combined, batch.y, group_ids, cwe_vocab_ids)
                loss = loss + self.supcon_weight * sc

        return logit_func, loss

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        total_epochs: int,
        class_weight: torch.Tensor | None = None,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"  Train {epoch:03d}/{total_epochs}", unit="batch", leave=False)

        for batch in pbar:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                _, loss = self._forward(batch, class_weight)

            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                if hasattr(self, "_grad_clip") and self._grad_clip > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if hasattr(self, "_grad_clip") and self._grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)
                self.optimizer.step()

            if self.step_per_batch:
                self.scheduler.step()

            total_loss += loss.item() * batch.num_graphs
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader.dataset)

    def set_grad_clip(self, clip: float) -> None:
        self._grad_clip = clip

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        is_binary: bool = True,
        class_weight: torch.Tensor | None = None,
    ) -> tuple[float, float, float, float, float]:
        """Return (loss, accuracy, mean_confidence, f1_macro, f1_weighted)."""
        self.model.eval()
        total_loss = 0.0
        total_conf = 0.0
        all_preds:  list[int] = []
        all_labels: list[int] = []

        for batch in loader:
            batch = batch.to(self.device)
            logits, loss = self._forward(batch, class_weight)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            total_loss += loss.item() * batch.num_graphs
            total_conf += probs.max(dim=-1).values.sum().item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

        n   = len(loader.dataset)
        avg = "binary" if is_binary else "macro"
        f1_macro    = f1_score(all_labels, all_preds, average=avg,       zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
        return total_loss / n, acc, total_conf / n, float(f1_macro), float(f1_weighted)

    # ── Localisation ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def localise(self, data, top_k: int = 5) -> list[tuple[int, float]]:
        """Return top-k (line, score) for a single graph."""
        self.model.eval()
        data  = data.to(self.device)
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)
        node_line = getattr(data, "node_line", None)

        if hasattr(self.model, "codebert"):
            fids  = getattr(data, "func_input_ids",      None)
            fmask = getattr(data, "func_attention_mask", None)
            fids  = fids.unsqueeze(0)  if fids  is not None else None
            fmask = fmask.unsqueeze(0) if fmask is not None else None
            out = self.model(data.x, data.edge_index, batch, node_line, None, fids, fmask)
        else:
            out = self.model(data.x, data.edge_index, batch, node_line)

        # Extract stmt_scores
        stmt_scores_list = out[1] if len(out) >= 2 else None
        if stmt_scores_list is None or len(stmt_scores_list[0]) == 0:
            return []

        scores_raw = stmt_scores_list[0]
        scores = (
            1.0 - torch.softmax(scores_raw, dim=-1)[:, 0]
            if scores_raw.dim() == 2
            else torch.sigmoid(scores_raw)
        )

        valid_lines = data.node_line[data.node_line >= 0].unique(sorted=True)
        k = min(top_k, len(valid_lines))
        top_scores, top_idx = scores.topk(k)
        return [
            (int(valid_lines[i].item()), float(top_scores[j].item()))
            for j, i in enumerate(top_idx)
        ]
