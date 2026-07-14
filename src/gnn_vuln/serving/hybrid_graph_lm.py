"""Arsitektur hibrida graph-LM — jalur graph seperti graph_based, plus jalur LM hidup
atas fungsi utuh lewat sliding window (5.120 token, 1.024 per jendela)."""
from __future__ import annotations

from dataclasses import dataclass

from gnn_vuln.serving._common import SHARED_DESIGN

NAME = "hybrid_graph_lm"

DESIGN = {   # Bab III — locked
    **SHARED_DESIGN,
    "architecture": "lmgat_codebert",
    "func_max_length": 5120,
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
}


@dataclass
class Params:
    hidden_dim: int = 768
    num_layers: int = 4
    dropout: float = 0.3
    heads: int = 4
    num_classes: int = 26
