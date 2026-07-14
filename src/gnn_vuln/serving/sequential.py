"""Arsitektur sekuensial — tahap 1 memberi skor kecurigaan tiap node, tahap 2 membaca ulang
graph dengan skor itu sebagai kondisi, sehingga klasifikasi melihat hasil lokalisasi."""
from __future__ import annotations

from dataclasses import dataclass

from gnn_vuln.serving._common import SHARED_DESIGN

NAME = "sequential"

DESIGN = {   # Bab III — locked
    **SHARED_DESIGN,
    "architecture": "lmgat_seqgnn",
    "live_lm": "none",
    "func_max_length": 1024,
    "localization_encoder": "gnn",
    "jknet_mode": "concat",
    "jknet_readout": "meanmax",
    "jknet_loc": False,
    "seq_stage2_input": "raw",
    "seq_susp_pool": False,
    "seq_susp_pool_k": 4.0,
    "seq_detach_susp": False,
}


@dataclass
class Params:
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.3
    heads: int = 4
    num_classes: int = 26
