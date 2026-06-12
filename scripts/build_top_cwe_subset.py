"""
build_top_cwe_subset.py — tail-strip a built LAZY dataset by keeping benign +
the top-N CWE classes (by sample count), WITHOUT re-running Joern or re-embedding
nodes. Pure subset + relabel of already-built graphs (node embeddings, edge feats,
func tokens, raw_func all reused verbatim).

Reads  {processed_dir}/{ds_name}_meta.pt
       {processed_dir}/{ds_name}_graphs/{i}.pt           (+ optional _labels.pt)
Writes {processed_dir}/{ds_name}{suffix}_graphs/{j}.pt
       {processed_dir}/{ds_name}{suffix}_meta.pt
       {processed_dir}/{ds_name}{suffix}_graphs/_labels.pt

Keeps benign (class 0) + the top-N CWE classes by count; drops the rest.
Remaps y to a contiguous 0..K range (benign stays 0).

After running, set in the downstream config:
    data.ds_name_suffix: {suffix}
    data.top_cwe: 0            # already filtered in the .pt — don't re-filter
    model.num_classes: <K>     # printed at the end

Reuse note: ml5120 and ml1024 share identical node embeddings (ml only affects the
func-branch token length). Strip from the ml5120 base; a 1024 model re-tokenizes the
func tokens from g.raw_func via the dataset patch path (no re-embed).

Run (cloud, Linux):
    PYTHONPATH=src python scripts/build_top_cwe_subset.py \
        --processed-dir data/processed --ds-name <base_no_meta> --top-cwe 18
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True, help="dir holding {ds_name}_meta.pt + _graphs/")
    ap.add_argument("--ds-name", required=True, help="base dataset name (no _meta.pt suffix)")
    ap.add_argument("--top-cwe", type=int, required=True, help="keep benign + top-N CWE by sample count")
    ap.add_argument("--suffix", default=None, help="appended to ds_name (default: _top{N})")
    a = ap.parse_args()
    suffix = a.suffix or f"_top{a.top_cwe}"

    pdir = Path(a.processed_dir)
    base_meta_p = pdir / f"{a.ds_name}_meta.pt"
    base_graphs = pdir / f"{a.ds_name}_graphs"
    if not base_meta_p.exists():
        sys.exit(f"missing meta: {base_meta_p}")
    if not base_graphs.exists():
        sys.exit(f"missing graphs dir: {base_graphs}")

    meta = torch.load(base_meta_p, weights_only=False)
    n = int(meta["n_graphs"])
    class_names = list(meta["class_names"])        # index == class_id

    # ── labels: prefer cached _labels.pt, else scan graphs ──────────────────
    labels_cache = base_graphs / "_labels.pt"
    if labels_cache.exists():
        labels = [int(x) for x in torch.load(labels_cache, weights_only=False)]
    else:
        labels = []
        for i in tqdm(range(n), desc="  scan labels", unit="g"):
            g = torch.load(base_graphs / f"{i}.pt", weights_only=False)
            labels.append(int(g.y))

    # ── pick keep set: benign(0) + top-N CWE by count ───────────────────────
    cnt = Counter(labels)
    cwe_ids = sorted((c for c in cnt if c != 0), key=lambda c: -cnt[c])
    keep_cwe = cwe_ids[: a.top_cwe]
    keep_sorted = [0] + sorted(keep_cwe)           # benign first, then by original id (freq rank)
    keep_set = set(keep_sorted)
    old2new = {old: new for new, old in enumerate(keep_sorted)}
    new_class_names = [class_names[old] for old in keep_sorted]

    dropped = sorted((c for c in cnt if c not in keep_set))
    print(f"base: {n} graphs, {len(class_names)} classes", file=sys.stderr)
    print(f"keep: benign + top-{a.top_cwe} CWE -> {len(keep_sorted)} classes", file=sys.stderr)
    print(f"drop {len(dropped)} classes: {[class_names[c] for c in dropped]} "
          f"({sum(cnt[c] for c in dropped)} graphs)", file=sys.stderr)

    # ── write filtered + relabeled graphs ───────────────────────────────────
    dst_graphs = pdir / f"{a.ds_name}{suffix}_graphs"
    dst_graphs.mkdir(parents=True, exist_ok=True)

    j = 0
    new_labels: list[int] = []
    for i in tqdm(range(n), desc="  filter+relabel", unit="g"):
        g = torch.load(base_graphs / f"{i}.pt", weights_only=False)
        old_y = int(g.y)
        if old_y not in keep_set:
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
               pdir / f"{a.ds_name}{suffix}_meta.pt")
    torch.save(torch.tensor(new_labels, dtype=torch.long), dst_graphs / "_labels.pt")

    print(f"DONE -> {a.ds_name}{suffix}  ({j} graphs, {len(new_class_names)} classes)", file=sys.stderr)
    print(f"set in config:  data.ds_name_suffix: {suffix}   data.top_cwe: 0   "
          f"model.num_classes: {len(new_class_names)}", file=sys.stderr)


if __name__ == "__main__":
    main()
