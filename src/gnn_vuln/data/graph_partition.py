"""Graph patch partitioning for Graph-ViT / Graph-MLP-Mixer (He et al. 2023).

Faithful to github.com/XiaoxinHe/Graph-ViT-MLPMixer transform: METIS partition into
n_patches, 1-hop expansion (overlapping patches), then the combined-subgraph index
machinery so the patch-GNN can run on all patches in ONE pass.

Runs ONLINE inside the model forward (= metis.online=True). METIS needs the `metis`
python pkg + libmetis + networkx (cloud only). For LOCAL smoke-testing without METIS,
falls back to a random balanced partition (set GRAPHVIT_FORCE_RANDOM=1 or auto on
ImportError) — shapes/logic identical, only cluster quality differs.
"""
from __future__ import annotations

import os
import numpy as np
import torch

try:
    import metis as _metis          # cloud: pip install metis (+ libmetis)
    import networkx as _nx
    _HAS_METIS = True
except Exception:                    # local: no metis → random fallback
    _HAS_METIS = False

_FORCE_RANDOM = os.environ.get("GRAPHVIT_FORCE_RANDOM", "0") == "1"


def _metis_membership(edge_index: torch.Tensor, num_nodes: int, n_patches: int,
                      drop_rate: float) -> torch.Tensor:
    """Patch id per node [num_nodes] via METIS (faithful) or random fallback."""
    if num_nodes <= n_patches or _FORCE_RANDOM or not _HAS_METIS:
        # paper does randperm when num_nodes < n_patches; else random fallback
        m = torch.arange(num_nodes) % n_patches
        return m[torch.randperm(num_nodes)]
    # data augmentation: randomly drop edges before partitioning (paper drop_rate)
    adj = edge_index.t().cpu()
    if drop_rate > 0.0:
        keep = torch.rand(adj.size(0)) > drop_rate
        adj = adj[keep]
    G = _nx.Graph()
    G.add_nodes_from(np.arange(num_nodes))
    G.add_edges_from(adj.tolist())
    _, membership = _metis.part_graph(G, n_patches, recursive=True)
    membership = torch.tensor(np.asarray(membership[:num_nodes]), dtype=torch.long)
    # shift so max patch id == n_patches-1 (paper: membership + (n_patches - max-1))
    membership = membership + (n_patches - (membership.max() + 1))
    return membership


def build_patches(
    edge_index: torch.Tensor,
    num_nodes: int,
    n_patches: int,
    num_hops: int = 1,
    drop_rate: float = 0.0,
):
    """Partition ONE graph into n_patches overlapping patches.

    Returns:
      node_mapper [M]   — original node id for each (patch, node) slot, M = sum of patch sizes
      patch_of    [M]   — patch id for each slot (= subgraphs_batch)
      comb_edge   [2,Ec]— edges re-indexed into the [M]-node combined patch graph
      comb_eattr_idx [Ec] — index into original edges (to gather edge_attr)
    """
    device = edge_index.device
    membership = _metis_membership(edge_index, num_nodes, n_patches, drop_rate).to(device)

    # node_mask [n_patches, num_nodes]: base membership
    node_mask = torch.zeros(n_patches, num_nodes, dtype=torch.bool, device=device)
    node_mask[membership, torch.arange(num_nodes, device=device)] = True

    # 1-hop expansion (sparse, O(E)): a patch absorbs the 1-hop neighbours of its nodes.
    if num_hops > 0 and edge_index.numel() > 0:
        src, dst = edge_index[0], edge_index[1]
        # for each edge (src->dst): if src in patch p, add dst to patch p (and undirected)
        for s, d in ((src, dst), (dst, src)):
            patch_of_s = membership[s]                       # [E] patch of source endpoint
            node_mask[patch_of_s, d] = True

    # to_sparse: (patch, node) slots in row-major patch order
    patch_of, node_mapper = node_mask.nonzero(as_tuple=True)  # [M], [M]

    # combine_subgraphs: re-index edges into the [M]-slot combined graph.
    # edge kept in patch p iff both endpoints are in patch p.
    edge_in_patch = node_mask[:, edge_index[0]] & node_mask[:, edge_index[1]]  # [n_patches, E]
    e_patch, e_idx = edge_in_patch.nonzero(as_tuple=True)     # [Ec], [Ec]
    # global node id -> slot id, per patch. Build a [n_patches, num_nodes] mapper.
    slot = torch.full((n_patches, num_nodes), -1, dtype=torch.long, device=device)
    slot[patch_of, node_mapper] = torch.arange(node_mapper.size(0), device=device)
    src2 = slot[e_patch, edge_index[0, e_idx]]
    dst2 = slot[e_patch, edge_index[1, e_idx]]
    comb_edge = torch.stack([src2, dst2], dim=0)             # [2, Ec]
    return node_mapper, patch_of, comb_edge, e_idx, membership
