"""
build_vuln_only_subset.py — drop benign (class 0) from a built LAZY dataset and relabel
the 25 CWE classes to a contiguous 0..24, so OUR model can train vuln-only and compare
FAIRLY against LOSVER (which classifies CWE type given a vulnerable function, no benign).

Same CPU-only reuse trick as build_top_cwe_subset.py — pure subset + relabel of the
already-built graphs, NO Joern, NO re-embed.

Reads  {processed_dir}/{ds_name}_meta.pt + {ds_name}_graphs/{i}.pt (+ optional _labels.pt)
Writes {processed_dir}/{ds_name}{suffix}_graphs/{j}.pt + {ds_name}{suffix}_meta.pt + _labels.pt

Keeps y > 0 (vulnerable), remaps old CWE ids 1..K -> 0..K-1 (benign was 0).
class_names = old class_names without 'benign'.

After running, set in the downstream config:
    data.ds_name_suffix: {suffix}   data.top_cwe: 0   model.num_classes: <K>  (printed)

Reuse note: strip from ml5120 for LM models (O/H series) and ml1024 for GNN-only — node
embeddings are identical across ml; only func-token length differs (re-tokenizable).

Run (cloud, Linux):
    PYTHONPATH=src python scripts/build_vuln_only_subset.py \
        --processed-dir data/processed --ds-name <base_no_meta>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--ds-name", required=True, help="base dataset name (no _meta.pt suffix)")
    ap.add_argument("--suffix", default="_vulnonly", help="appended to ds_name (default: _vulnonly)")
    ap.add_argument("--benign-id", type=int, default=0, help="class id of benign (default 0)")
    a = ap.parse_args()

    pdir = Path(a.processed_dir)
    base_meta_p = pdir / f"{a.ds_name}_meta.pt"
    base_graphs = pdir / f"{a.ds_name}_graphs"
    if not base_meta_p.exists():
        sys.exit(f"missing meta: {base_meta_p}")
    if not base_graphs.exists():
        sys.exit(f"missing graphs dir: {base_graphs}")

    meta = torch.load(base_meta_p, weights_only=False)
    n = int(meta["n_graphs"])
    class_names = list(meta["class_names"])
    b = a.benign_id

    # new contiguous ids: drop benign, shift everything above it down by 1
    kept_old = [c for c in range(len(class_names)) if c != b]
    old2new = {old: i for i, old in enumerate(kept_old)}
    new_class_names = [class_names[old] for old in kept_old]

    print(f"base: {n} graphs, {len(class_names)} classes (benign id={b}='{class_names[b]}')", file=sys.stderr)
    print(f"keep: {len(new_class_names)} vuln CWE classes -> {new_class_names}", file=sys.stderr)

    dst_graphs = pdir / f"{a.ds_name}{a.suffix}_graphs"
    dst_graphs.mkdir(parents=True, exist_ok=True)

    j = 0
    new_labels: list[int] = []
    for i in tqdm(range(n), desc="  drop-benign+relabel", unit="g"):
        g = torch.load(base_graphs / f"{i}.pt", weights_only=False)
        old_y = int(g.y)
        if old_y == b:
            continue
        new_y = old2new[old_y]
        if torch.is_tensor(g.y):
            g.y = torch.full_like(g.y, new_y)
        else:
            g.y = new_y
        torch.save(g, dst_graphs / f"{j}.pt")
        new_labels.append(new_y)
        j += 1

    torch.save({"n_graphs": j, "class_names": new_class_names},
               pdir / f"{a.ds_name}{a.suffix}_meta.pt")
    torch.save(torch.tensor(new_labels, dtype=torch.long), dst_graphs / "_labels.pt")

    print(f"DONE -> {a.ds_name}{a.suffix}  ({j} graphs, {len(new_class_names)} classes)", file=sys.stderr)
    print(f"set in config:  data.ds_name_suffix: {a.suffix}   data.top_cwe: 0   "
          f"model.num_classes: {len(new_class_names)}", file=sys.stderr)


if __name__ == "__main__":
    main()
