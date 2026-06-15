"""
bigvul_survivor_to_parquet.py — build a parquet of LIVABLE's Big-Vul SURVIVORS so OUR model
trains on the SAME functions + 31-class labels + LIVABLE's EXACT split (fair head-to-head vs
LIVABLE-on-Big-Vul, 51.7 macro-F1). Trap-2 solved: only the funcs LIVABLE's joern kept.

Re-derives all_vul rows (for func_after -> flaw_lines) by REPLICATING livable_prepare_bigvul's
filter exactly, so df.index == the LIVABLE id (the "<id>.c" in used_*.json). Then keeps only
the joern survivors (used_{train,val,test}.json from the split-record bundle).

Emits into --out-dir:
  survivors.parquet   cols: id, func_before, func_after, "CWE ID"(=cwe_name), target, label
  cwe_vocab.json      {cwe_name: label}  == LIVABLE 31-class (pre-seed so prepare_dataset reuses it)
  split_file.json     {"train":[pos...],"val":[...],"test":[...]}  keyed on PARQUET ROW POSITION

parquet_id flow: parquet row position -> load_bigvul df["id"](=df.index, no rows dropped here) ->
prepare_dataset row_id -> meta "id" -> dataset_lm graph.parquet_id. get_splits(split_file) buckets
by parquet_id. So pass NO --limit / --sample-per-class to prepare_dataset (preserve row order).

Run:
    python scripts/bigvul_survivor_to_parquet.py --csv data/LIVABLE/all_vul.csv \
        --record-dir bigvul_livable --out-dir data/datasets/bigvul_survivor --top-cwe 31
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/LIVABLE/all_vul.csv", help="Big-Vul all_vul.csv")
    ap.add_argument("--record-dir", required=True,
                    help="dir with used_{train,val,test}.json + cwe_labels.json (split-record bundle)")
    ap.add_argument("--out-dir", default="data/datasets/bigvul_survivor")
    ap.add_argument("--top-cwe", type=int, default=31, help="must match the LIVABLE bigvul run (31)")
    a = ap.parse_args()
    rec, out = Path(a.record_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- replicate livable_prepare_bigvul filter EXACTLY so df.index == LIVABLE id ---
    df = pd.read_csv(a.csv, usecols=["func_before", "func_after", "CWE ID", "target"])
    df = df[df["target"] == 1].dropna(subset=["func_before"]).reset_index(drop=True)
    df["cwe_name"] = df["CWE ID"].fillna("Nonetype").astype(str).replace("", "Nonetype")
    if a.top_cwe and a.top_cwe > 0:
        keep = set(df["cwe_name"].value_counts().head(a.top_cwe).index)
        df = df[df["cwe_name"].isin(keep)].reset_index(drop=True)
    # df.index is now the LIVABLE id (matches "<id>.c" in used_*.json)

    cwe_labels = json.loads((rec / "cwe_labels.json").read_text(encoding="utf-8"))["cwe_labels"]
    cwe_to_label = {c: i for i, c in enumerate(cwe_labels)}

    split_of: dict[int, str] = {}
    for s in ("train", "val", "test"):
        for fp in json.loads((rec / f"used_{s}.json").read_text(encoding="utf-8")):
            split_of[int(str(fp).rsplit(".", 1)[0])] = s     # "123.c" -> 123

    rows, splits = [], []
    for lid, r in df.iterrows():
        if lid not in split_of:                 # not a joern survivor -> skip
            continue
        fb = r["func_before"]
        if not isinstance(fb, str) or not fb.strip():
            continue                             # mirror load_bigvul dropna(code) -> keep positions aligned
        cwe = r["cwe_name"]
        if cwe not in cwe_to_label:              # should never happen (survivors are in the 31)
            continue
        rows.append({"func_before": fb, "func_after": r["func_after"],
                     "CWE ID": cwe, "target": 1, "label": cwe_to_label[cwe]})
        splits.append(split_of[lid])

    odf = pd.DataFrame(rows)
    odf["id"] = range(len(odf))                  # parquet position == future parquet_id
    odf.to_parquet(out / "survivors.parquet", index=False)

    split_file: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for pos, s in enumerate(splits):
        split_file[s].append(pos)
    (out / "split_file.json").write_text(json.dumps(split_file), encoding="utf-8")
    (out / "cwe_vocab.json").write_text(json.dumps(cwe_to_label, indent=2), encoding="utf-8")

    sizes = {k: len(v) for k, v in split_file.items()}
    print(f"survivors: {len(odf)}  split: {sizes}")
    print(f"  num_classes: {len(cwe_labels)}")
    print(f"  parquet   -> {out / 'survivors.parquet'}")
    print(f"  split_file-> {out / 'split_file.json'} (keyed on parquet position == parquet_id)")
    print(f"  cwe_vocab -> {out / 'cwe_vocab.json'} (pre-seed into the CPG out-dir)")
    print(f"  class dist: {dict(sorted(Counter(odf['label']).items()))}")


if __name__ == "__main__":
    main()
