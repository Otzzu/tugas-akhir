"""Reusable GNN encoder blocks shared across architectures."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, GINEConv, GatedGraphConv, RGCNConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import degree, to_torch_csr_tensor


def _compute_rwse(edge_index: torch.Tensor, num_nodes: int, walk_length: int = 16) -> torch.Tensor:
    """Random Walk Structural Encoding (Dwivedi 2022, used in GNN+ 2025).
    Returns [N, walk_length] tensor: diagonal of (D^-1 A)^k for k=1..walk_length.
    Each row = probability of returning to start node after k steps.
    """
    device = edge_index.device
    row, col = edge_index[0], edge_index[1]
    deg_inv = 1.0 / degree(row, num_nodes=num_nodes).clamp(min=1.0)
    # Build sparse D^-1 A as edge weights (CSR for efficient mm)
    edge_w = deg_inv[row]
    adj = torch.sparse_coo_tensor(edge_index, edge_w, size=(num_nodes, num_nodes), device=device).coalesce()
    pe = torch.zeros(num_nodes, walk_length, device=device)
    # Iteratively: M_k = (D^-1 A)^k. Track diagonal each step.
    M_k = adj
    for k in range(walk_length):
        # diag(M_k) — sum over output index for entries where row == col
        idx = M_k.indices()
        vals = M_k.values()
        self_mask = idx[0] == idx[1]
        if self_mask.any():
            pe[idx[0][self_mask], k] = vals[self_mask]
        if k < walk_length - 1:
            M_k = torch.sparse.mm(M_k, adj).coalesce()
    return pe


def _build_norm(norm_type: str, hidden_dim: int) -> nn.Module:
    """Build per-layer normalization. 'batch' (default) or 'graph' (Cai 2021 ICML)."""
    if norm_type == "graph":
        return GraphNorm(hidden_dim)
    if norm_type == "batch":
        return nn.BatchNorm1d(hidden_dim)
    raise ValueError(f"norm_type must be 'batch' or 'graph', got {norm_type!r}")


def _activation(act: str) -> callable:
    """Pick activation. 'relu' (default) or 'elu' (original GAT 2018)."""
    if act == "relu":
        return F.relu
    if act == "elu":
        return F.elu
    raise ValueError(f"activation must be 'relu' or 'elu', got {act!r}")


@torch.no_grad()
def apply_balanced_init(layers, beta: float = 2.0) -> None:
    """Mustafa et al. NeurIPS 2023 'Are GATs Out of Balance?' — Procedure 2.6.

    Approximated BalO for GATv2Conv layers:
      1. Zero attention vector a^l for all layers l (Xav+ZeroAtt)
      2. Apply orthogonal init to lin_l / lin_r weights (LL-Ortho base)
      3. Scale first layer row norms to sqrt(beta) (default beta=2.0)
      4. Balance inter-layer per-neuron norms (simplified: skip for multi-head GAT
         due to shape complexity — Xav+Bal approximation still effective)

    Paper showed BalO → 80.2% Cora vs Xavier 39.3% at L=10.
    """
    if not layers:
        return
    for conv in layers:
        # Zero ALL attention parameters (handles GATConv 'att_src/dst' and GATv2Conv 'att')
        for attr in ("att", "att_src", "att_dst", "att_l", "att_r"):
            if hasattr(conv, attr):
                p = getattr(conv, attr)
                if isinstance(p, nn.Parameter):
                    p.zero_()
        # Orthogonal init for linear weights (LL-Ortho base)
        for lin_name in ("lin", "lin_l", "lin_r", "lin_src", "lin_dst", "lin_edge"):
            if hasattr(conv, lin_name):
                lin = getattr(conv, lin_name)
                if lin is not None and hasattr(lin, "weight") and lin.weight is not None:
                    nn.init.orthogonal_(lin.weight)
                    if hasattr(lin, "bias") and lin.bias is not None:
                        lin.bias.zero_()
    # Step 3: Scale first layer row norms to sqrt(beta)
    first = layers[0]
    for lin_name in ("lin_l", "lin_r"):
        if hasattr(first, lin_name):
            lin = getattr(first, lin_name)
            if lin is None or not hasattr(lin, "weight"):
                continue
            W = lin.weight  # [out, in]
            norms = W.norm(dim=1, keepdim=True).clamp(min=1e-6)
            W.copy_(W / norms * (beta ** 0.5))

# CPG edge types: AST, CFG, CDG, DDG, PDG, CALL, REACHING_DEF
NUM_EDGE_TYPES = 7


# ── Shared residual projection helper ─────────────────────────────────────────

def _build_res_projs(
    in_channels: int, hidden_dim: int, num_layers: int
) -> nn.ModuleList:
    """Residual projections: Linear for layer 0, Identity for rest."""
    projs = nn.ModuleList()
    projs.append(nn.Linear(in_channels, hidden_dim, bias=False))
    for _ in range(num_layers - 1):
        projs.append(nn.Identity())
    return projs


# ── GAT Encoder ───────────────────────────────────────────────────────────────

class GATEncoder(nn.Module):
    """
    Stack of GATv2Conv layers with Norm + Activation + Dropout.
    Optional residual skip connections.

    block_style:
      - "resnet"   (legacy default): Conv → Norm → +residual → Act → Dropout
      - "gnn_plus" (Luo 2025 ICML SOTA): Conv → Norm → Act → Dropout → +residual
    norm_type:
      - "batch" (default) — BatchNorm1d
      - "graph" — GraphNorm (Cai 2021 ICML, per-graph normalization, needs batch index)
    activation:
      - "relu" (default) — ReLU
      - "elu" — ELU (original GAT 2018 activation)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        edge_dim: int = NUM_EDGE_TYPES,
        add_self_loops: bool = False,
        use_skip: bool = False,
        fill_value: float = 0.0,
        block_style: str = "resnet",
        norm_type: str = "batch",
        activation: str = "relu",
        use_ffn: bool = False,
        ffn_expansion: int = 2,
        use_pe: bool = False,
        pe_walk_length: int = 16,
        pe_dim: int = 28,
        balanced_init: bool = False,
        balanced_init_beta: float = 2.0,
    ):
        super().__init__()
        assert block_style in ("resnet", "gnn_plus"), \
            f"block_style must be 'resnet' or 'gnn_plus', got {block_style!r}"
        self.dropout = dropout
        self.use_skip = use_skip
        self.block_style = block_style
        self.norm_type = norm_type
        self.act_fn = _activation(activation)
        self._needs_batch = (norm_type == "graph")
        self.use_ffn = use_ffn
        self.use_pe = use_pe
        self.pe_walk_length = pe_walk_length

        # PE encoder (GNN+ 2025 RWSE): random walk PE → BN → Linear → dim_pe.
        # Concatenated to node features before first conv. Increases in_channels by pe_dim.
        if use_pe:
            self.pe_raw_norm = nn.BatchNorm1d(pe_walk_length)
            self.pe_encoder = nn.Linear(pe_walk_length, pe_dim)
            in_channels_eff = in_channels + pe_dim
        else:
            in_channels_eff = in_channels

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(
            GATv2Conv(
                in_channels_eff, hidden_dim, heads=num_heads, concat=False,
                dropout=dropout, edge_dim=edge_dim,
                add_self_loops=add_self_loops, fill_value=fill_value,
            )
        )
        self.bns.append(_build_norm(norm_type, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(
                    hidden_dim, hidden_dim, heads=num_heads, concat=False,
                    dropout=dropout, edge_dim=edge_dim,
                    add_self_loops=add_self_loops, fill_value=fill_value,
                )
            )
            self.bns.append(_build_norm(norm_type, hidden_dim))

        if use_skip:
            self.res_projs = _build_res_projs(in_channels_eff, hidden_dim, num_layers)

        # Apply Mustafa NeurIPS 2023 balanced init AFTER all layers built but BEFORE FFN add.
        # Zeros attention vectors + orthogonal init + sqrt(beta) first-layer row scaling.
        if balanced_init:
            apply_balanced_init(self.convs, beta=balanced_init_beta)

        # Per-layer FFN block (GNN+ 2025 — matches official github.com/LUOyk1999/GNNPlus _ff_block).
        # Block: BN(norm1) → [Linear → Act → Drop → Linear → Drop] → +residual → BN(norm2)
        # 2x expansion (W1: D→2D, W2: 2D→D). Three BNs total per layer when FFN is on.
        if use_ffn:
            ffn_dim = hidden_dim * ffn_expansion
            self.ffn_w1 = nn.ModuleList([nn.Linear(hidden_dim, ffn_dim) for _ in range(num_layers)])
            self.ffn_w2 = nn.ModuleList([nn.Linear(ffn_dim, hidden_dim) for _ in range(num_layers)])
            self.ffn_norm1 = nn.ModuleList([_build_norm(norm_type, hidden_dim) for _ in range(num_layers)])
            self.ffn_norm2 = nn.ModuleList([_build_norm(norm_type, hidden_dim) for _ in range(num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        rwse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # PE: prefer precomputed RWSE from dataset (batch.rwse). Fallback: compute on-the-fly.
        if self.use_pe:
            if rwse is None:
                rwse = _compute_rwse(edge_index, x.size(0), walk_length=self.pe_walk_length)
            pe = self.pe_raw_norm(rwse)
            pe = self.pe_encoder(pe)
            x = torch.cat([x, pe], dim=-1)
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            residual = self.res_projs[i](x) if self.use_skip else None
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x, batch) if self._needs_batch else bn(x)
            if self.block_style == "gnn_plus":
                x = self.act_fn(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if residual is not None:
                    x = x + residual
            else:
                x = self.act_fn(x + residual) if residual is not None else self.act_fn(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # FFN sub-block (GNN+ 2025, matches official _ff_block exactly):
            # x → BN(norm1) → [Linear1 → Act → Drop → Linear2 → Drop] → +x → BN(norm2)
            if self.use_ffn:
                x = self.ffn_norm1[i](x, batch) if self._needs_batch else self.ffn_norm1[i](x)
                ff = self.act_fn(self.ffn_w1[i](x))
                ff = F.dropout(ff, p=self.dropout, training=self.training)
                ff = self.ffn_w2[i](ff)
                ff = F.dropout(ff, p=self.dropout, training=self.training)
                x = x + ff
                x = self.ffn_norm2[i](x, batch) if self._needs_batch else self.ffn_norm2[i](x)
        return x


# ── GCN Encoder ───────────────────────────────────────────────────────────────

class GCNEncoder(nn.Module):
    """
    Stack of GCNConv layers. Edge features are ignored (GCN is edge-agnostic).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        add_self_loops: bool = True,
        use_skip: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_skip = use_skip

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim, add_self_loops=add_self_loops))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        if use_skip:
            self.res_projs = _build_res_projs(in_channels, hidden_dim, num_layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,  # ignored
    ) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            residual = self.res_projs[i](x) if self.use_skip else None
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x + residual) if residual is not None else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ── RGCN Encoder ──────────────────────────────────────────────────────────────

class RGCNEncoder(nn.Module):
    """
    Relational GCN: one weight matrix per CPG edge type.
    Converts one-hot edge_attr [E, num_relations] → integer edge_type [E].
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_relations: int = NUM_EDGE_TYPES,
        num_bases: int | None = None,
        use_skip: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_skip = use_skip

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_dim, num_relations, num_bases=num_bases))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        if use_skip:
            self.res_projs = _build_res_projs(in_channels, hidden_dim, num_layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_attr is not None and edge_attr.shape[0] > 0:
            edge_type = edge_attr.argmax(dim=-1)
        else:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            residual = self.res_projs[i](x) if self.use_skip else None
            x = conv(x, edge_index, edge_type=edge_type)
            x = bn(x)
            x = F.relu(x + residual) if residual is not None else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ── GGNN Encoder ──────────────────────────────────────────────────────────────

class GGNNEncoder(nn.Module):
    """
    Linear projection + GatedGraphConv (GatedGraphConv requires in==out).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_skip: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_skip = use_skip
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,  # ignored
    ) -> torch.Tensor:
        proj = self.input_proj(x)
        h = self.ggnn(proj, edge_index)
        if self.use_skip:
            h = F.relu(h + proj)
        else:
            h = F.relu(h)
        return F.dropout(h, p=self.dropout, training=self.training)


# ── GIN Encoder ───────────────────────────────────────────────────────────────

def _gin_mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim, out_dim),
    )


class GINEncoder(nn.Module):
    """
    GINEConv with per-layer edge feature projection.
    Layer 0: edge projection 7→in_channels; layers 1+: 7→hidden_dim.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        edge_dim: int = NUM_EDGE_TYPES,
        use_skip: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_skip = use_skip

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        # layer 0: edge proj to in_channels
        self.edge_projs = nn.ModuleList()
        self.edge_projs.append(nn.Linear(edge_dim, in_channels))
        self.convs.append(GINEConv(_gin_mlp(in_channels, hidden_dim, dropout), edge_dim=in_channels))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.edge_projs.append(nn.Linear(edge_dim, hidden_dim))
            self.convs.append(GINEConv(_gin_mlp(hidden_dim, hidden_dim, dropout), edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        if use_skip:
            self.res_projs = _build_res_projs(in_channels, hidden_dim, num_layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), NUM_EDGE_TYPES, device=x.device)

        for i, (ep, conv, bn) in enumerate(zip(self.edge_projs, self.convs, self.bns)):
            residual = self.res_projs[i](x) if self.use_skip else None
            ea = ep(edge_attr)
            x = conv(x, edge_index, edge_attr=ea)
            x = bn(x)
            x = F.relu(x + residual) if residual is not None else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ── Encoder factory ───────────────────────────────────────────────────────────

def build_gnn_encoder(
    gnn_model: str,
    in_channels: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    num_heads: int = 4,
    edge_dim: int = NUM_EDGE_TYPES,
    add_self_loops: bool = False,
    use_skip: bool = False,
    num_relations: int = NUM_EDGE_TYPES,
    num_bases: int | None = None,
    block_style: str = "resnet",
    norm_type: str = "batch",
    activation: str = "relu",
    use_ffn: bool = False,
    ffn_expansion: int = 2,
    use_pe: bool = False,
    pe_walk_length: int = 16,
    pe_dim: int = 28,
    balanced_init: bool = False,
    balanced_init_beta: float = 2.0,
) -> nn.Module:
    """Build a GNN encoder by name. All encoders share forward(x, edge_index, edge_attr).

    gat  — GATv2Conv  (uses num_heads, edge_dim, add_self_loops)
    gcn  — GCNConv     (edge-agnostic; uses add_self_loops)
    gin  — GINEConv    (uses edge_dim)
    rgcn — RGCNConv    (uses num_relations, num_bases)
    ggnn — GatedGraphConv (edge-agnostic)

    block_style: "resnet" (legacy) or "gnn_plus" (Luo 2025) — currently only GAT supports.
    """
    m = gnn_model.lower()
    if m == "gat":
        return GATEncoder(in_channels, hidden_dim, num_layers, num_heads, dropout,
                          edge_dim, add_self_loops, use_skip,
                          block_style=block_style, norm_type=norm_type, activation=activation,
                          use_ffn=use_ffn, ffn_expansion=ffn_expansion,
                          use_pe=use_pe, pe_walk_length=pe_walk_length, pe_dim=pe_dim,
                          balanced_init=balanced_init, balanced_init_beta=balanced_init_beta)
    if m == "gcn":
        return GCNEncoder(in_channels, hidden_dim, num_layers, dropout,
                          add_self_loops, use_skip)
    if m == "gin":
        return GINEncoder(in_channels, hidden_dim, num_layers, dropout,
                          edge_dim, use_skip)
    if m == "rgcn":
        return RGCNEncoder(in_channels, hidden_dim, num_layers, dropout,
                           num_relations, num_bases, use_skip)
    if m == "ggnn":
        return GGNNEncoder(in_channels, hidden_dim, num_layers, dropout, use_skip)
    raise ValueError(
        f"gnn_model must be gat|gcn|gin|rgcn|ggnn, got {gnn_model!r}"
    )
