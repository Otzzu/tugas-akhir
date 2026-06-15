"""
vulpcl_prepare_megavul.py — convert OUR exported baseline split into VulPCL's
categorization inputs, so VulPCL (CodeBERT+BLSTM, multiclass CWE) trains on our split.

VulPCL categorization pipeline (src/VulPCL/vul_categorization):
  source .c -> data_preprocessing/graph_generation.py (OLD Neo4j-Gremlin Joern!) ->
  ast/cfg_dfg/ddg_cdg SVG graphs -> adc_features_extracting.py (deepwalk graph feats) +
  cd_features_extracting.py (FCDS sequence) + CodeBERT tokens -> codebert_blstm.py train.
  Label file: `<project>_labels.txt`, lines `<file>@@<label>`; file == `<seg0>@<seg1>.c`
  (their code keys graphs by the first two @-segments).

We emit (a) source .c files named `<id>@megavul.c` (fits their 2-@-segment parsing),
(b) `megavul_labels.txt` (`<id>@megavul.c@@<label>`), (c) split manifest + cwe_to_label.

Vuln-only 25-class (vul==1, drop benign) to match LOSVER + LIVABLE label space.
NOTE: VulPCL's native categorization is 12-class (benign + 11 CWE); we override to our
vuln-only set. Their `vul_files_label.py` cwe_to_label dict + codebert.py target_names
must be set to our num_classes (printed below).

Input  : {in_dir}/{train,val,test}.parquet  cols: func_before, vul, cwe_name, flaw_lines
Output : {out_dir}/source/{split}/<id>@megavul.c
         {out_dir}/megavul_labels.txt          (<file>@@<label>, all splits)
         {out_dir}/split_map.json              {file: split}
         {out_dir}/cwe_labels.json             {"cwe_labels":[...], "num_classes":N}

Run:
    python scripts/vulpcl_prepare_megavul.py --in-dir megavul_ml1024/linevd --out-dir megavul_vulpcl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="dir with train/val/test.parquet (our baseline export)")
    ap.add_argument("--out-dir", required=True, help="output dir for VulPCL inputs")
    ap.add_argument("--project", default="megavul", help="project tag (the 2nd @-segment)")
    ap.add_argument("--keep-benign", action="store_true",
                    help="keep vul==0 as class 0 (default: vuln-only, matches LOSVER)")
    a = ap.parse_args()
    ind, out = Path(a.in_dir), Path(a.out_dir)
    proj = a.project

    frames = []
    for split in ("train", "val", "test"):
        df = pd.read_parquet(ind / f"{split}.parquet")
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if not a.keep_benign:
        df = df[df["vul"] == 1].reset_index(drop=True)   # vuln-only, no benign (== LOSVER)

    # keep_benign (default for LIVABLE/VulPCL, == our model): class 0 = "benign" (vul==0),
    # classes 1..N = 25 CWE. vuln-only (== LOSVER): 0..N-1 over CWEs, no benign.
    df["cwe_name"] = df["cwe_name"].fillna("others").replace("", "others").astype(str)
    if a.keep_benign:
        df.loc[df["vul"] == 0, "cwe_name"] = "benign"
        vuln_cwes = sorted(df[df["vul"] == 1]["cwe_name"].unique())
        cwe_labels = ["benign"] + vuln_cwes        # benign == class 0
    else:
        cwe_labels = sorted(df["cwe_name"].unique())
    cwe_to_label = {c: i for i, c in enumerate(cwe_labels)}

    src_dir = out / "source"
    for split in ("train", "val", "test"):
        (src_dir / split).mkdir(parents=True, exist_ok=True)
    label_lines = []
    split_map = {}
    for idx, r in df.iterrows():
        fid = int(r["id"]) if "id" in df.columns and pd.notna(r.get("id")) else idx
        split = r["split"]
        label = cwe_to_label[r["cwe_name"]]
        fname = f"{fid}@{proj}.c"          # first two @-segments -> {fid}@{proj}.c
        (src_dir / split / fname).write_text(str(r["func_before"]), encoding="utf-8")
        label_lines.append(f"{fname}@@{label}")
        split_map[fname] = split

    out.mkdir(parents=True, exist_ok=True)
    (out / f"{proj}_labels.txt").write_text("\n".join(label_lines), encoding="utf-8")
    (out / "split_map.json").write_text(json.dumps(split_map), encoding="utf-8")
    (out / "cwe_labels.json").write_text(
        json.dumps({"cwe_labels": cwe_labels, "num_classes": len(cwe_labels)}, indent=2),
        encoding="utf-8",
    )

    n = len(df)
    print(f"wrote {n} functions -> {src_dir}")
    print(f"  splits: {df['split'].value_counts().to_dict()}")
    print(f"  num_classes: {len(cwe_labels)}  (vuln-only={not a.keep_benign})")
    print(f"  label file: {out / (proj + '_labels.txt')}")
    print(f"  set in vul_categorization/codebert.py target_names = {[str(i) for i in range(len(cwe_labels))]}")
    print(f"  cwe_labels (index order): {cwe_labels}")


if __name__ == "__main__":
    main()
