"""
livable_prepare_megavul.py — convert OUR exported baseline split into the inputs
LIVABLE's preprocessing consumes, so LIVABLE trains on our MegaVul split.

LIVABLE pipeline (src/LIVABLE): raw function .c files -> old-Joern parse -> the
Devign-style builder `preprocessing/ori_ourdevign+token.py` -> GGNNinput JSON
(per-func record: node_features word2vec, graph edges, targets[[label]], sequence)
-> DGL GGNN train (`code/main_sta.py`, num_classes).

We emit (a) one .c file per function for Joern, (b) a labels manifest (id -> int
class + split), and (c) the cwe_labels list + num_classes to wire into main_sta.py.

Vuln-only 25-class (vul==1, drop benign) to match LOSVER's no-benign label space so
the multiclass comparison is apples-to-apples. CWE label index = sorted-unique order
(shared with LOSVER if the same cwe set; printed so it can be reused).

Input  : {in_dir}/{train,val,test}.parquet  cols: func_before, vul, cwe_name, flaw_lines
Output : {out_dir}/functions/{split}/{id}.c
         {out_dir}/labels.json   [{"id", "split", "label", "cwe_name"}, ...]
         {out_dir}/cwe_labels.json  {"cwe_labels": [...], "num_classes": N}

Run:
    python scripts/livable_prepare_megavul.py --in-dir megavul_ml1024/linevd --out-dir megavul_livable
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="dir with train/val/test.parquet (our baseline export)")
    ap.add_argument("--out-dir", required=True, help="output dir for LIVABLE inputs")
    ap.add_argument("--keep-benign", action="store_true",
                    help="keep vul==0 as class 0 (default: vuln-only, matches LOSVER)")
    a = ap.parse_args()
    ind, out = Path(a.in_dir), Path(a.out_dir)

    frames = []
    for split in ("train", "val", "test"):
        df = pd.read_parquet(ind / f"{split}.parquet")
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if not a.keep_benign:
        df = df[df["vul"] == 1].reset_index(drop=True)   # vuln-only, no benign (== LOSVER)

    # Stable CWE -> int label. Vuln-only: 0..N-1 over sorted unique CWEs.
    df["cwe_name"] = df["cwe_name"].fillna("others").replace("", "others").astype(str)
    cwe_labels = sorted(df["cwe_name"].unique())
    if a.keep_benign:
        # benign reserved as class 0; CWEs shifted to 1..N
        cwe_labels = [c for c in cwe_labels]
    cwe_to_label = {c: i for i, c in enumerate(cwe_labels)}

    func_dir = out / "functions"
    manifest = []
    # per-split jsonl for the builder ori_ourdevign+token.py: reads entry['file_path'] +
    # entry['func'], emits targets[[label]]. file_path == the .c filename (Joern csv dir name).
    jsonl = {s: [] for s in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        (func_dir / split).mkdir(parents=True, exist_ok=True)
    for idx, r in df.iterrows():
        fid = int(r["id"]) if "id" in df.columns and pd.notna(r.get("id")) else idx
        split = r["split"]
        label = cwe_to_label[r["cwe_name"]]
        src = str(r["func_before"])
        fname = f"{fid}.c"
        (func_dir / split / fname).write_text(src, encoding="utf-8")
        manifest.append({"id": fid, "split": split, "label": label, "cwe_name": r["cwe_name"]})
        jsonl[split].append({"file_path": fname, "func": src, "target": label,
                             "label": label, "cwe_name": r["cwe_name"]})

    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.json").write_text(json.dumps(manifest), encoding="utf-8")
    for split in ("train", "val", "test"):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for rec in jsonl[split]:
                f.write(json.dumps(rec) + "\n")
    (out / "cwe_labels.json").write_text(
        json.dumps({"cwe_labels": cwe_labels, "num_classes": len(cwe_labels)}, indent=2),
        encoding="utf-8",
    )

    n = len(df)
    print(f"wrote {n} functions -> {func_dir}")
    print(f"  splits: {df['split'].value_counts().to_dict()}")
    print(f"  num_classes: {len(cwe_labels)}  (vuln-only={not a.keep_benign})")
    print(f"  set in src/LIVABLE/code/main_sta.py: num_classes = {len(cwe_labels)}")
    print(f"  cwe_labels (index order): {cwe_labels}")


if __name__ == "__main__":
    main()
