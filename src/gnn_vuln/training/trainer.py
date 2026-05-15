"""Trainer class: forward pass dispatch, train loop, evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class _CUDAPrefetcher:
    """Overlap CPU collation with GPU compute using a background CUDA stream."""

    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device
        self._use_cuda = device.type == "cuda"
        self.stream = torch.cuda.Stream() if self._use_cuda else None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self._iter = iter(self.loader)
        self._next = None
        self._preload()
        return self

    def _preload(self):
        try:
            batch = next(self._iter)
        except StopIteration:
            self._next = None
            return
        if self._use_cuda:
            with torch.cuda.stream(self.stream):
                batch = batch.to(self.device, non_blocking=True)
        self._next = batch

    def __next__(self):
        if self._next is None:
            raise StopIteration
        if self._use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self._next
        self._preload()
        return batch

from gnn_vuln.training.losses import (
    focal_loss,
    livable_loss,
    mil_loss,
    mil_loss_multiclass,
    ranking_loss,
)

# Import lazily to avoid circular imports; resolved at runtime
_EWCDR_TYPE = None


def _ewcdr_type():
    global _EWCDR_TYPE
    if _EWCDR_TYPE is None:
        from gnn_vuln.training.ewc import EWCDR
        _EWCDR_TYPE = EWCDR
    return _EWCDR_TYPE


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
        ewc=None,   # EWCDR | None
        grad_accum_steps: int = 1,
        label_smoothing: float = 0.0,
        use_livable_real: bool = False,
        livable_focal_gamma: float = 2.0,
        livable_label_smoothing: float = 0.1,
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
        self.ewc                = ewc
        self.grad_accum_steps   = max(1, grad_accum_steps)
        self.label_smoothing    = label_smoothing
        self.use_livable_real   = use_livable_real
        self.livable_focal_gamma = livable_focal_gamma
        self.livable_label_smoothing = livable_label_smoothing
        self._current_epoch     = 1
        self._total_epochs      = 100

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
            func_ids   = getattr(batch, "func_input_ids",      None)
            func_mask  = getattr(batch, "func_attention_mask", None)
            func_tlines = getattr(batch, "func_token_lines",   None)
            out = self.model(
                batch.x, batch.edge_index, batch.batch,
                node_line, edge_attr, func_ids, func_mask, func_tlines,
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
        if self.use_livable_real:
            loss = livable_loss(
                logit_func, batch.y,
                epoch=self._current_epoch,
                total_epochs=self._total_epochs,
                focal_gamma=self.livable_focal_gamma,
                label_smoothing=self.livable_label_smoothing,
                weight=class_weight,
            )
        elif self.focal_gamma > 0.0:
            loss = focal_loss(logit_func, batch.y, gamma=self.focal_gamma,
                              weight=class_weight, label_smoothing=self.label_smoothing)
        else:
            loss = F.cross_entropy(logit_func, batch.y, weight=class_weight,
                                   label_smoothing=self.label_smoothing)

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

        # EWC-DR continual learning regularization
        if self.ewc is not None:
            loss = loss + self.ewc.penalty(self.model)

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
        self._current_epoch = epoch
        self._total_epochs  = total_epochs
        accum = self.grad_accum_steps
        self.optimizer.zero_grad()

        # Accumulate loss on GPU — avoids per-batch .item() sync which stalls
        # training waiting for GPU. One sync at end of epoch + throttled tqdm.
        loss_sum = torch.zeros(1, device=self.device)
        n_graphs = 0
        n_steps  = len(loader)
        # Throttle tqdm refresh: update display ~100 times/epoch regardless of size
        refresh_every = max(1, n_steps // 100)

        pbar = tqdm(loader, desc=f"  Train {epoch:03d}/{total_epochs}", unit="batch", leave=False)

        for step, batch in enumerate(pbar):
            batch = batch.to(self.device, non_blocking=True)
            is_last = (step == n_steps - 1)
            should_step = ((step + 1) % accum == 0) or is_last

            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                _, loss = self._forward(batch, class_weight)

            # Scale loss so gradient magnitude is independent of accum_steps
            loss = loss / accum

            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                if should_step:
                    if hasattr(self, "_grad_clip") and self._grad_clip > 0.0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if should_step:
                    if hasattr(self, "_grad_clip") and self._grad_clip > 0.0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if self.step_per_batch and should_step:
                self.scheduler.step()

            # Accumulate on GPU (no sync). Multiply by accum to undo the earlier
            # loss / accum scaling so total_loss reflects un-normalized loss sum.
            loss_sum = loss_sum + loss.detach() * (accum * batch.num_graphs)
            n_graphs += batch.num_graphs

            # Only sync for tqdm display every refresh_every steps
            if (step % refresh_every == 0) or is_last:
                pbar.set_postfix(loss=f"{(loss_sum / n_graphs).item():.4f}")

        # Single sync at epoch end
        return (loss_sum.item() / n_graphs) if n_graphs > 0 else 0.0

    def set_grad_clip(self, clip: float) -> None:
        self._grad_clip = clip

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def evaluate(
        self,
        loader: DataLoader,
        is_binary: bool = True,
        class_weight: torch.Tensor | None = None,
    ) -> dict:
        """Return metrics dict: loss, acc, conf, f1_macro, f1_weighted, precision_macro, recall_macro, precision_weighted, recall_weighted, per_class."""
        self.model.eval()
        loss_sum  = torch.zeros(1, device=self.device)
        conf_sum  = torch.zeros(1, device=self.device)
        preds_buf:  list[torch.Tensor] = []
        labels_buf: list[torch.Tensor] = []

        for batch in loader:
            batch = batch.to(self.device, non_blocking=True)
            logits, loss = self._forward(batch, class_weight)
            probs = F.softmax(logits, dim=-1)
            preds_buf.append(logits.argmax(dim=-1))
            labels_buf.append(batch.y)
            loss_sum = loss_sum + loss.detach() * batch.num_graphs
            conf_sum = conf_sum + probs.max(dim=-1).values.sum()

        # Single CPU sync for all accumulated tensors
        all_preds  = torch.cat(preds_buf).cpu().tolist()
        all_labels = torch.cat(labels_buf).cpu().tolist()
        n          = len(all_labels)
        avg = "binary" if is_binary else "macro"
        f1_macro         = f1_score(all_labels, all_preds, average=avg,        zero_division=0)
        f1_weighted      = f1_score(all_labels, all_preds, average="weighted",  zero_division=0)
        precision_main   = precision_score(all_labels, all_preds, average=avg,        zero_division=0)
        recall_main      = recall_score(all_labels, all_preds, average=avg,           zero_division=0)
        precision_w      = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall_w         = recall_score(all_labels, all_preds, average="weighted",    zero_division=0)
        acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        per_class = {
            k: {
                "precision": round(v["precision"], 6),
                "recall":    round(v["recall"],    6),
                "f1":        round(v["f1-score"],  6),
                "support":   int(v["support"]),
            }
            for k, v in report.items()
            if k not in ("macro avg", "weighted avg", "accuracy")
        }
        return {
            "loss":               (loss_sum / n).item(),
            "acc":                acc,
            "conf":               (conf_sum / n).item(),
            "f1_macro":           float(f1_macro),
            "f1_weighted":        float(f1_weighted),
            "precision_macro":    float(precision_main),
            "recall_macro":       float(recall_main),
            "precision_weighted": float(precision_w),
            "recall_weighted":    float(recall_w),
            "per_class":          per_class,
        }

    # ── Localisation ──────────────────────────────────────────────────────────

    @torch.inference_mode()
    def localise(self, data, top_k: int = 5) -> list[tuple[int, float]]:
        """Return top-k (line, score) for a single graph."""
        self.model.eval()
        data  = data.to(self.device, non_blocking=True)
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
