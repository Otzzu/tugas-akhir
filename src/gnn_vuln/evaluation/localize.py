"""LocalizationExtractor — runs model inference and collects per-function scores."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from gnn_vuln.metrics import LocalizationMetrics


class LocalizationExtractor:
    """
    Runs a trained model over a DataLoader and collects:
      - function-level predictions (y_true, y_pred, y_prob, confidence)
      - per-function statement-level localization dicts
    """

    def __init__(self, model: nn.Module, loader: DataLoader, device: torch.device) -> None:
        self.model = model
        self.loader = loader
        self.device = device

    @torch.no_grad()
    def run(self):
        """
        Returns
        -------
        y_true      : np.ndarray [N]
        y_pred      : np.ndarray [N]
        y_prob      : np.ndarray [N, C]
        confidence  : np.ndarray [N]
        loc_results : list[dict]  — one per function
        """
        import numpy as np

        self.model.eval()
        all_y, all_pred, all_prob, all_conf = [], [], [], []
        loc_results: list[dict] = []

        for batch in self.loader:
            batch = batch.to(self.device)
            node_line = getattr(batch, "node_line", None)
            flaw_mask = getattr(batch, "flaw_line_mask", None)
            edge_attr  = getattr(batch, "edge_attr", None)

            out = self._forward(batch, node_line, edge_attr)

            logit_func, stmt_scores_list = self._unpack(out)

            probs = torch.softmax(logit_func, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)
            confs = probs.max(axis=1)

            all_y.extend(batch.y.cpu().numpy().tolist())
            all_pred.extend(preds.tolist())
            all_prob.append(probs)
            all_conf.extend(confs.tolist())

            if stmt_scores_list is not None and node_line is not None:
                B = int(batch.batch.max().item()) + 1
                for b in range(B):
                    loc_results.append(
                        self._extract_func(
                            stmt_scores_list[b], batch.batch, node_line, flaw_mask, b
                        )
                    )
            else:
                B = int(batch.batch.max().item()) + 1
                for _ in range(B):
                    loc_results.append(LocalizationMetrics.make_result([], [], []))

        import numpy as np
        return (
            np.array(all_y),
            np.array(all_pred),
            np.vstack(all_prob) if all_prob else np.zeros((0, 2)),
            np.array(all_conf),
            loc_results,
        )

    def _forward(self, batch, node_line, edge_attr):
        if hasattr(self.model, "codebert"):
            func_input_ids      = getattr(batch, "func_input_ids", None)
            func_attention_mask = getattr(batch, "func_attention_mask", None)
            return self.model(
                batch.x, batch.edge_index, batch.batch, node_line, edge_attr,
                func_input_ids, func_attention_mask,
            )
        return self.model(batch.x, batch.edge_index, batch.batch, node_line, edge_attr)

    @staticmethod
    def _unpack(out):
        """Unpack model output regardless of tuple length."""
        if isinstance(out, (tuple, list)):
            if len(out) >= 5:
                return out[0], out[3]
            if len(out) == 4:
                return out[0], out[3]
            if len(out) == 3:
                return out[0], out[2]
            return out[0], out[1]
        return out, None

    @staticmethod
    def _extract_func(scores_b, batch_idx, node_line, flaw_mask, b) -> dict:
        """Extract per-function localization data for graph b."""
        if len(scores_b) == 0:
            return LocalizationMetrics.make_result([], [], [])

        if scores_b.dim() == 2:
            probs = torch.softmax(scores_b, dim=-1)
            scores_scalar = (1.0 - probs[:, 0]).cpu()
        else:
            scores_scalar = torch.sigmoid(scores_b).cpu()

        graph_mask  = batch_idx == b
        node_line_b = node_line[graph_mask]
        flaw_b = (
            flaw_mask[graph_mask] if flaw_mask is not None
            else torch.zeros_like(node_line_b)
        )

        valid = node_line_b >= 0
        if not valid.any():
            return LocalizationMetrics.make_result([], [], [])

        lines_b = node_line_b[valid]
        flaw_b  = flaw_b[valid]
        unique_lines = lines_b.unique(sorted=True)

        line_scores: list[float] = []
        line_labels: list[int]  = []
        for line in unique_lines:
            mask = lines_b == line
            line_scores.append(scores_scalar[mask].max().item())
            line_labels.append(int(flaw_b[mask].any().item()))

        return LocalizationMetrics.make_result(
            unique_lines.cpu().tolist(),
            line_scores,
            line_labels,
        )
