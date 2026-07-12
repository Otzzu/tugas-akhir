"""
export_api_json.py — export the EXACT rows of a built lazy .pt dataset into the API
dataset-upload format (POST /datasets/upload): a plain JSON array of DatasetRow objects.

Row shape (API/schemas/dataset.py DatasetRow):
  benign      : {"code", "label": 0, "language"}
  vulnerable  : {"code", "cwe", "flaw_lines", "language"}          (mask-derived GT)
  vulnerable  : {"code", "cwe", "func_after", "language"}          (no mask lines; API diffs)
Rows are never given both flaw_lines and func_after (schema forbids it).

flaw_lines = sorted(unique(node_line[flaw_line_mask>0])) — the same statement-level GT
our training/eval uses. Vulnerable rows without mask lines fall back to func_after joined
from the original parquet by parquet_id; rows with neither are SKIPPED (counted) because
the API rejects a vulnerable row without a localization source.

Split replicates dataset_lm.get_splits(): seed 42, shuffle, 80/10/10 (NOT stratified).
Writes {out_dir}/all.json, train.json, val.json, test.json + counts to stderr.

Run (pod, dataset already extracted by train_cloud.sh):
    PYTHONPATH=src python scripts/export_api_json.py \
        --processed-dir data/processed \
        --ds-name lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42 \
        --parquet data/datasets/megavul/train.parquet \
        --out-dir data/api_export/megavul_real
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm


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
    ap.add_argument("--parquet", default=None, help="original parquet for func_after join")
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

    raw_pq = None
    if a.parquet:
        import pandas as pd
        raw_pq = pd.read_parquet(a.parquet, columns=["func_before", "func_after", "language"])
        print(f"parquet join: {len(raw_pq)} rows", file=sys.stderr)

    tr, va, te = get_splits(n, a.seed)
    split_of = {}
    for s, ids in (("train", tr), ("val", va), ("test", te)):
        for i in ids:
            split_of[i] = s

    rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    all_rows: list[dict] = []
    n_flaw, n_after, n_skip, n_mismatch = 0, 0, 0, 0

    for i in tqdm(range(n), desc="export", unit="g"):
        g = torch.load(gdir / f"{i}.pt", weights_only=False)
        y = int(g.y)
        code = getattr(g, "raw_func", "") or ""
        pid = int(g.parquet_id) if hasattr(g, "parquet_id") else -1
        lang = getattr(g, "language", None) or None

        pq = None
        if raw_pq is not None and 0 <= pid < len(raw_pq):
            pq = raw_pq.iloc[pid]
            if str(pq["func_before"]) != code:
                n_mismatch += 1
            if lang is None:
                lang = str(pq["language"]) or None

        row: dict = {"code": code}
        if lang:
            row["language"] = lang

        if y == 0:
            row["label"] = 0
        else:
            row["cwe"] = class_names[y] if y < len(class_names) else str(y)
            flaw = sorted({int(x) for x in g.node_line[g.flaw_line_mask > 0].tolist()})
            n_lines = len(code.split("\n"))
            flaw = [ln for ln in flaw if 1 <= ln <= n_lines]
            if flaw:
                row["flaw_lines"] = flaw
                n_flaw += 1
            elif pq is not None and isinstance(pq["func_after"], str) and pq["func_after"]:
                row["func_after"] = str(pq["func_after"])
                n_after += 1
            else:
                n_skip += 1
                continue  # vulnerable row without any localization source — API rejects

        rows[split_of[i]].append(row)
        all_rows.append(row)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("all", all_rows), ("train", rows["train"]),
                       ("val", rows["val"]), ("test", rows["test"])):
        p = out / f"{name}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"  {name}.json: {len(data)} rows", file=sys.stderr)

    print(f"vuln via flaw_lines={n_flaw}  via func_after={n_after}  "
          f"skipped(no source)={n_skip}  code!=parquet={n_mismatch}", file=sys.stderr)
    print(f"DONE -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
