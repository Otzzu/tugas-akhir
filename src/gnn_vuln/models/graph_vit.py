"""Graph-ViT / Graph-MLP-Mixer vuln detector (He et al. 2023, faithful).

GNN-only arch (live_lm=none). Pipeline: METIS patches → GATv2 patch-GNN (+ U
cross-scale mix, reference model.py) → token-mixer over patches → pool → func_head;
per-node embeds → stmt_head (localization). mixer_type selects MLP-Mixer or attention.

Partition runs ONLINE in forward (metis.online=True). Separate from lmgat_codebert.
Cloud-only (needs metis+networkx); local smoke-test uses random-partition fallback.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import scatter

from gnn_vuln.data.graph_partition import build_patches
from gnn_vuln.models.base import VulnDetectorBase
from gnn_vuln.models.encoders import _compute_rwse
from gnn_vuln.models.heads import FuncHead, LinearFuncHead, ThinFuncHead, StmtHead


# ── token mixers ───────────────────────────────────────────────────────────────

class _FeedForward(nn.Module):
    """Reference FeedForward: Dropout -> Linear -> GELU -> Dropout -> Linear."""

    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


class _MixerBlock(nn.Module):
    """MLP-Mixer block (reference mlp_mixer.py): token-mix over patches + channel-mix.
    token_dim = dim*4, channel_dim = dim//2 (faithful to MixerBlock(nhid, P, nhid*4, nhid//2))."""

    def __init__(self, dim, n_patches, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = _FeedForward(n_patches, dim * 4, dropout)        # token_dim = nhid*4
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = _FeedForward(dim, max(dim // 2, 1), dropout)   # channel_dim = nhid//2

    def forward(self, x, key_padding_mask=None):  # x [B, P, D]
        y = self.norm1(x).transpose(1, 2)          # [B, D, P]
        x = x + self.token_mix(y).transpose(1, 2)
        x = x + self.channel_mix(self.norm2(x))
        return x


class _AttnBlock(nn.Module):
    """Graph-ViT token-mixer: standard self-attention over patches."""

    def __init__(self, dim, n_heads=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim, dim * 2, dropout)

    def forward(self, x, key_padding_mask=None):
        y = self.norm1(x)
        a, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class _TokenMixer(nn.Module):
    def __init__(self, mixer_type, dim, n_patches, n_layers, n_heads, dropout):
        super().__init__()
        if mixer_type == "mlp":
            self.blocks = nn.ModuleList([_MixerBlock(dim, n_patches, dropout) for _ in range(n_layers)])
        elif mixer_type == "attention":
            self.blocks = nn.ModuleList([_AttnBlock(dim, n_heads, dropout) for _ in range(n_layers)])
        else:
            raise ValueError(f"mixer_type must be 'mlp' or 'attention', got {mixer_type!r}")
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        for b in self.blocks:
            x = b(x, key_padding_mask)
        return self.norm(x)


# ── model ──────────────────────────────────────────────────────────────────────

class GraphViTVulnDetector(VulnDetectorBase):
    def __init__(self, in_channels, hidden_dim, num_classes,
                 nlayer_gnn=2, n_patches=32, num_hops=1, patch_drop_rate=0.0,
                 heads=4, edge_dim=31, dropout=0.3, mixer_type="attention",
                 mixer_layers=4, pe_walk_length=16, func_head_type="linear",
                 add_self_loops=False):
        super().__init__()
        self.n_patches = n_patches
        self.num_hops = num_hops
        self.patch_drop_rate = patch_drop_rate
        self.hidden_dim = hidden_dim
        self.pe_walk_length = pe_walk_length
        self._live_lm = "none"

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        # node RWSE positional encoding (reference: x += rw_encoder(rw_pos_enc) before patch gather)
        self.node_pe = nn.Linear(pe_walk_length, hidden_dim) if pe_walk_length > 0 else None
        # patch GNN: each "block" = GATv2 → BN → ReLU → drop → +residual → Linear
        # (reference GNN(nlayer=1) block: conv→bn→relu→drop→+prev→output_encoder), GATv2 per user.
        self.gnns = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False,
                      dropout=dropout, edge_dim=hidden_dim, add_self_loops=add_self_loops)
            for _ in range(nlayer_gnn)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(nlayer_gnn)])
        self.out_lin = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(nlayer_gnn)])
        # U: cross-scale node↔patch mix between GNN layers (reference model.py: U=MLP nlayer1 w/ act)
        self.U = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                                for _ in range(nlayer_gnn - 1)])

        self.mixer = _TokenMixer(mixer_type, hidden_dim, n_patches, mixer_layers, heads, dropout)
        self.dropout = dropout

        if func_head_type == "linear":
            self.func_head = LinearFuncHead(hidden_dim, num_classes, dropout=dropout)
        elif func_head_type == "thin":
            self.func_head = ThinFuncHead(hidden_dim, num_classes)
        else:
            self.func_head = FuncHead(hidden_dim, hidden_dim, num_classes, dropout)
        self.stmt_head = StmtHead(hidden_dim, lm_dim=0, localization_encoder="gnn")

    def has_live_lm(self) -> bool:
        return False

    def lm_parameters(self):
        return []

    def _partition_batch(self, x, edge_index, batch):
        """Build the batched combined-patch structure (online METIS per graph)."""
        device = x.device
        B = int(batch.max().item()) + 1 if batch.numel() else 1
        node_off = patch_off = slot_off = edge_off = 0
        nm, po, ce, ei, mem_g = [], [], [], [], []
        for g in range(B):
            mask = batch == g
            nodes = mask.nonzero(as_tuple=True)[0]
            ng = nodes.numel()
            if ng == 0:
                continue
            remap = torch.full((x.size(0),), -1, dtype=torch.long, device=device)
            remap[nodes] = torch.arange(ng, device=device)
            e_mask = mask[edge_index[0]] & mask[edge_index[1]]
            ge = remap[edge_index[:, e_mask]]                       # local edges [2, Eg]
            node_mapper, patch_of, comb_edge, e_idx, membership = build_patches(
                ge, ng, self.n_patches, self.num_hops,
                self.patch_drop_rate if self.training else 0.0)
            nm.append(node_mapper + node_off)
            po.append(patch_of + patch_off)
            ce.append(comb_edge + slot_off)
            ei.append(e_idx + edge_off)                            # index into this graph's edges
            mem_g.append(membership + patch_off)                  # per original node (graph order)
            node_off += ng
            patch_off += self.n_patches
            slot_off += node_mapper.numel()
            edge_off += int(e_mask.sum().item())
        # global edge index list (to gather edge_attr): edges in graph order
        return (torch.cat(nm), torch.cat(po), torch.cat(ce, dim=1),
                torch.cat(ei), torch.cat(mem_g), B)

    def forward(self, x, edge_index, batch, node_line=None, edge_attr=None,
                rwse=None, **kwargs):
        device = x.device
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), self.edge_proj.in_features, device=device)

        nm, po, comb_edge, e_glob_idx, membership, B = self._partition_batch(x, edge_index, batch)
        # build global edge order matching e_glob_idx (per-graph edge slices concatenated)
        # e_glob_idx already offset by per-graph edge counts in graph order; gather edge_attr
        # via the same graph-order edge listing:
        order = torch.argsort(batch[edge_index[0]], stable=True)   # edges grouped by graph
        edge_attr_ord = edge_attr[order]
        e_slots = self.edge_proj(edge_attr_ord[e_glob_idx])        # [M_edges, hid]

        # node features + RWSE PE (reference: x += rw_encoder(rwse)), then gather into patch slots
        x_proj = self.input_proj(x)
        if self.node_pe is not None:
            if rwse is None:
                rwse = _compute_rwse(edge_index, x.size(0), walk_length=self.pe_walk_length)
            x_proj = x_proj + self.node_pe(rwse)
        xs = x_proj[nm]                                            # slot node feats [M, hid]
        TP = B * self.n_patches                                    # total patch slots

        for i in range(len(self.gnns)):
            if i > 0:
                patch_sum = scatter(xs, po, dim=0, dim_size=TP, reduce="mean")[po]
                xs = xs + self.U[i - 1](patch_sum)
                xs = scatter(xs, nm, dim=0, dim_size=x.size(0), reduce="mean")[nm]  # sync node copies
            res = xs
            h = self.gnns[i](xs, comb_edge, edge_attr=e_slots)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            xs = self.out_lin[i](h + res)                          # +residual → output_encoder (reference)

        # patch tokens [TP, hid] → [B, P, hid]
        patch_tok = scatter(xs, po, dim=0, dim_size=TP, reduce="mean")
        valid = scatter(torch.ones_like(po, dtype=torch.float), po, dim=0, dim_size=TP, reduce="sum") > 0
        patch_tok = patch_tok.view(B, self.n_patches, self.hidden_dim)
        valid = valid.view(B, self.n_patches)

        # mixer over patches (mask empty patches)
        patch_tok = self.mixer(patch_tok, key_padding_mask=~valid)

        # graph rep = masked mean over valid patches
        m = valid.unsqueeze(-1).float()
        h_graph = (patch_tok * m).sum(1) / m.sum(1).clamp(min=1.0)

        # localization: per-original-node embed (sync copies) + its patch's mixer context
        node_embed = scatter(xs, nm, dim=0, dim_size=x.size(0), reduce="mean")  # [N, hid]
        patch_flat = patch_tok.reshape(TP, self.hidden_dim)
        h_loc = node_embed + patch_flat[membership]                # broadcast patch context

        logit = self.func_head(h_graph)
        stmt_scores = self.stmt_head.score(h_loc, batch, node_line) if node_line is not None else None
        return logit, stmt_scores

    @classmethod
    def from_config(cls, cfg, in_channels, **kwargs):
        m = cfg.model
        return cls(
            in_channels=in_channels,
            hidden_dim=m.hidden_dim,
            num_classes=getattr(m, "num_classes", 26),
            nlayer_gnn=getattr(m, "num_layers", 2),
            n_patches=getattr(m, "n_patches", 32),
            num_hops=getattr(m, "num_hops", 1),
            patch_drop_rate=getattr(m, "patch_drop_rate", 0.0),
            heads=getattr(m, "heads", 4),
            edge_dim=getattr(m, "edge_dim", 31),
            dropout=getattr(cfg.train, "dropout", 0.3) if hasattr(cfg, "train") else 0.3,
            mixer_type=getattr(m, "mixer_type", "attention"),
            mixer_layers=getattr(m, "mixer_layers", 4),
            pe_walk_length=getattr(m, "pe_walk_length", 16),
            func_head_type=getattr(m, "func_head_type", "linear"),
            add_self_loops=getattr(m, "add_self_loops", False),
        )
