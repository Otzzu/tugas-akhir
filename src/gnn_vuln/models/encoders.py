"""Reusable GNN encoder blocks shared across architectures."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, GINEConv, GatedGraphConv, RGCNConv

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
    Stack of GATv2Conv layers with BatchNorm + ReLU + Dropout.
    Optional residual skip connections.
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
    ):
        super().__init__()
        self.dropout = dropout
        self.use_skip = use_skip

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(
            GATv2Conv(
                in_channels, hidden_dim, heads=num_heads, concat=False,
                dropout=dropout, edge_dim=edge_dim,
                add_self_loops=add_self_loops, fill_value=fill_value,
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(
                    hidden_dim, hidden_dim, heads=num_heads, concat=False,
                    dropout=dropout, edge_dim=edge_dim,
                    add_self_loops=add_self_loops, fill_value=fill_value,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        if use_skip:
            self.res_projs = _build_res_projs(in_channels, hidden_dim, num_layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            residual = self.res_projs[i](x) if self.use_skip else None
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x + residual) if residual is not None else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
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
