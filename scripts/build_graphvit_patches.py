"""Offline transform: existing lazy CodeBERTGraphDataset .pt -> graph_vit patched .pt.

Adds METIS patch fields (SubgraphsData) per graph so graph_vit reads precomputed
patches instead of partitioning online. Cloud-only (needs metis + libmetis + networkx).

Reads  {processed_dir}/{ds_name}_graphs/{i}.pt        -> partition_data -> SubgraphsData
Writes {processed_dir}/{ds_name}{suffix}_graphs/{i}.pt + copies _meta.pt / _labels.pt

n_patches / num_hops / patch_rw_dim MUST match the graph_vit config (the model assumes
the precomputed n_patches). After running, set `data.ds_name_suffix: <suffix>` in the config.

Run (cloud, after the base dataset is downloaded):
  uv run python scripts/build_graphvit_patches.py \
      --processed-dir data/processed \
      --ds-name lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42 \
      --suffix _gvitp16 --n-patches 16 --num-hops 1 --patch-rw-dim 8
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from gnn_vuln.data import graph_partition as _gp
from gnn_vuln.data.graph_partition import partition_data


def _require_metis():
    """Fail LOUDLY if real METIS isn't available — never build random partitions offline."""
    if _gp._FORCE_RANDOM:
        sys.exit("ERROR: GRAPHVIT_FORCE_RANDOM=1 is set — would build RANDOM (not METIS) "
                 "partitions. Unset it before the offline build.")
    if not _gp._HAS_METIS:
        sys.exit("ERROR: metis / networkx not importable — offline build requires real METIS.\n"
                 "  apt-get install -y libmetis-dev && pip install metis networkx\n"
                 "Refusing to build random partitions (would silently corrupt the dataset).")
    # runtime preflight: actually run METIS (catches 'import ok but libmetis not found at call').
    try:
        ei = torch.tensor([[0, 1, 2, 3, 4, 0, 2], [1, 2, 3, 4, 0, 3, 4]])
        m = _gp._metis_membership(ei, num_nodes=40, n_patches=8, drop_rate=0.0)
        assert int(m.max()) == 7 and m.numel() == 40
    except Exception as e:
        sys.exit(f"ERROR: METIS failed to run (libmetis not located?).\n"
                 f"  export METIS_DLL=/usr/lib/x86_64-linux-gnu/libmetis.so   # then retry\n"
                 f"  {type(e).__name__}: {e}")


def _require_torch_sparse():
    """reference engine needs torch_sparse (k_hop SparseTensor)."""
    try:
        import torch_sparse  # noqa: F401
    except Exception as e:
        sys.exit("ERROR: torch_sparse not importable — the reference engine needs it "
                 "(k_hop SparseTensor).\n  pip install torch_sparse   (or pass --engine plain)\n"
                 f"  {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True, help="dir holding {ds_name}_meta.pt + _graphs/")
    ap.add_argument("--ds-name", required=True, help="base dataset name (no _meta.pt suffix)")
    ap.add_argument("--suffix", default="_gvitp16", help="appended to ds_name for the patched output")
    ap.add_argument("--n-patches", type=int, default=16)
    ap.add_argument("--num-hops", type=int, default=1)
    ap.add_argument("--patch-rw-dim", type=int, default=8)
    ap.add_argument("--engine", choices=["reference", "plain"], default="reference",
                    help="reference = verbatim authors' partition (cloud, needs torch_sparse); "
                         "plain = local-tested reimpl (verified equal)")
    a = ap.parse_args()
    _require_metis()   # fail loudly if METIS unavailable — never silently build random patches
    if a.engine == "reference":
        _require_torch_sparse()
        from gnn_vuln.data.graph_partition_ref import partition_data_ref as _partition
        print("engine: reference (verbatim Graph-ViT/MLP-Mixer partition)", file=sys.stderr)
    else:
        _partition = partition_data
        print("engine: plain (verified-equivalent reimpl)", file=sys.stderr)

    pdir = Path(a.processed_dir)
    src_graphs = pdir / f"{a.ds_name}_graphs"
    if not src_graphs.exists():
        sys.exit(f"missing graphs dir: {src_graphs}")
    dst_name = f"{a.ds_name}{a.suffix}"
    dst_graphs = pdir / f"{dst_name}_graphs"
    dst_graphs.mkdir(parents=True, exist_ok=True)

    files = sorted(src_graphs.glob("*.pt"),
                   key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    print(f"{len(files)} graphs  {src_graphs}  ->  {dst_graphs}", file=sys.stderr)
    for f in tqdm(files):
        g = torch.load(f, weights_only=False)
        sd = _partition(g, a.n_patches, a.num_hops, 0.0, a.patch_rw_dim)
        torch.save(sd, dst_graphs / f.name)

    for tail in ("_meta.pt", "_labels.pt"):
        s = pdir / f"{a.ds_name}{tail}"
        if s.exists():
            shutil.copy(s, pdir / f"{dst_name}{tail}")
            print(f"copied {tail}", file=sys.stderr)

    print(f"DONE -> {dst_name}.  set  data.ds_name_suffix: {a.suffix}  in the graph_vit config",
          file=sys.stderr)


if __name__ == "__main__":
    main()
