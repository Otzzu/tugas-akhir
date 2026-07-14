"""Arsitektur berbasis graph — CPG -> GATv2/GNN+ -> JK-Net readout -> 2 head.
LM hanya memasok embedding node beku, tidak ada jalur LM hidup."""
from __future__ import annotations

from dataclasses import dataclass

from gnn_vuln.serving._common import SHARED_DESIGN

NAME = "graph_based"

DESIGN = {   # Bab III — locked
    **SHARED_DESIGN,
    "architecture": "lmgat_codebert",
    "live_lm": "none",
    "func_max_length": 1024,
    "localization_encoder": "gnn",
    "graph_pool": "jknet",
}


@dataclass
class Params:
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.3
    heads: int = 4
    num_classes: int = 26
