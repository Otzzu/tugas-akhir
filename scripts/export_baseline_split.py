"""
export_baseline_split.py — export the EXACT same filtered/sampled/split set used by
our model (from the built lazy .pt) into the input formats LineVul and LineVD expect.
Guarantees identical functions, identical train/val/test split, identical flaw-line GT.

Reads  {processed_dir}/{ds_name}_meta.pt + {ds_name}_graphs/{i}.pt
Split  replicates dataset_lm.get_splits(): seed 42, shuffle, 80/10/10 (NOT stratified).
GT     flaw lines = sorted(unique(node_line[flaw_line_mask>0]))  (1-indexed source lines)
       — the SAME statement-level ground truth our rank loss + eval use. func_after is
       NOT stored in the .pt, so LineVD must consume these precomputed flaw lines (patch
       its loader) instead of deriving them from a before/after diff.

Writes per split (train/val/test):
  {out_dir}/linevul/{split}.csv      cols: processed_func,target,cwe_id,cwe_name,
                                           flaw_line,flaw_line_index,parquet_id
  {out_dir}/linevd/{split}.parquet   cols: id,func_before,vul,label,cwe_name,
                                           flaw_lines,flaw_line_index
  {out_dir}/splits.json              parquet_id lists per split (traceability)

LineVul format: target = binary 0/1; flaw_line = flaw line CONTENTS joined by "/~/";
                flaw_line_index = 0-based line indices comma-joined (BigVul convention).
LineVD format:  vul = binary 0/1; flaw_lines = 1-indexed line numbers (list).
Multiclass: use cwe_id (0=benign..K) / cwe_name. Binary: use target / vul.

Run:
    PYTHONPATH=src python scripts/export_baseline_split.py \
        --processed-dir data/processed \
        --ds-name lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42 \
        --out-dir data/baselines/megavul_ml1024
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

FLAW_SEP = "/~/"   # LineVul flaw_line separator


def get_splits(n: int, seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Replicates CodeBERTGraphDataset.get_splits exactly (seeded shuffle, 80/10/10)."""
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    return idx[:t], idx[t : t + v], idx[t + v :]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--ds-name", required=True, help="base name, no _meta.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    pdir = Path(a.processed_dir)
    meta_p = pdir / f"{a.ds_name}_meta.pt"
    gdir = pdir / f"{a.ds_name}_graphs"
    if not meta_p.exists() or not gdir.exists():
        sys.exit(f"missing {meta_p} or {gdir}")

    meta = torch.load(meta_p, weights_only=False)
    n = int(meta["n_graphs"])
    class_names = list(meta["class_names"])
    print(f"{n} graphs, {len(class_names)} classes", file=sys.stderr)

    tr, va, te = get_splits(n, a.seed)
    split_of = {}
    for s, ids in (("train", tr), ("val", va), ("test", te)):
        for i in ids:
            split_of[i] = s
    print(f"split: train={len(tr)} val={len(va)} test={len(te)}", file=sys.stderr)

    rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    pid_of: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for i in tqdm(range(n), desc="export", unit="g"):
        g = torch.load(gdir / f"{i}.pt", weights_only=False)
        y = int(g.y)
        raw = getattr(g, "raw_func", "") or ""
        pid = int(g.parquet_id) if hasattr(g, "parquet_id") else -1

        flaw_lines = sorted({int(x) for x in g.node_line[g.flaw_line_mask > 0].tolist()})
        src_lines = raw.split("\n")
        # keep only in-range flaw lines; build content + 0-based index strings
        flaw_lines = [ln for ln in flaw_lines if 1 <= ln <= len(src_lines)]
        flaw_contents = [src_lines[ln - 1] for ln in flaw_lines]
        flaw_idx0 = [ln - 1 for ln in flaw_lines]

        s = split_of[i]
        rows[s].append({
            "id": pid,
            "parquet_id": pid,
            "processed_func": raw,
            "func_before": raw,
            "target": int(y > 0),       # LineVul binary
            "vul": int(y > 0),          # LineVD binary
            "label": y,                 # multiclass class id (0=benign)
            "cwe_id": y,
            "cwe_name": class_names[y] if y < len(class_names) else str(y),
            "flaw_line": FLAW_SEP.join(flaw_contents),       # LineVul: contents
            "flaw_line_index": ",".join(map(str, flaw_idx0)),  # LineVul: 0-based idx
            "flaw_lines": flaw_lines,                        # LineVD: 1-based line nums
        })
        pid_of[s].append(pid)

    out = Path(a.out_dir)
    (out / "linevul").mkdir(parents=True, exist_ok=True)
    (out / "linevd").mkdir(parents=True, exist_ok=True)

    linevul_cols = ["processed_func", "target", "cwe_id", "cwe_name",
                    "flaw_line", "flaw_line_index", "parquet_id"]
    linevd_cols = ["id", "func_before", "vul", "label", "cwe_name",
                   "flaw_lines", "flaw_line_index"]

    for s in ("train", "val", "test"):
        df = pd.DataFrame(rows[s])
        df[linevul_cols].to_csv(out / "linevul" / f"{s}.csv", index=False)
        df[linevd_cols].to_parquet(out / "linevd" / f"{s}.parquet", index=False)
        n_vuln = int(df["target"].sum())
        print(f"  {s}: {len(df)} rows ({n_vuln} vuln) -> linevul/{s}.csv, linevd/{s}.parquet",
              file=sys.stderr)

    with open(out / "splits.json", "w") as f:
        json.dump({"seed": a.seed, "ds_name": a.ds_name, "class_names": class_names,
                   "parquet_ids": pid_of}, f)
    print(f"DONE -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
