"""
livable_prepare_bigvul.py — convert Big-Vul (Fan et al., LIVABLE's OWN type-classification
dataset, data/LIVABLE/all_vul.csv) into the inputs LIVABLE's preprocessing consumes, so we can
run the FAITHFUL paper reproduction (vuln-only CWE-type classification on their data).

Mirrors livable_prepare_megavul.py output layout exactly so run_livable_cloud.sh steps 4-6
(joern parse -> builder -> train) work unchanged:
  {out}/functions/{split}/{id}.c
  {out}/{split}.jsonl            ({file_path, func, target, label, cwe_name})
  {out}/labels.json
  {out}/cwe_labels.json          ({cwe_labels, num_classes})

all_vul.csv = 10,900 rows, ALL vuln (target=1). Columns: func_before, "CWE ID", target.
CWE ID NaN/empty -> "Nonetype" (vuln-but-no-CWE = LIVABLE's Nonetype class, the 2nd most common).
--top-cwe K (default 31, == LIVABLE's class_num): keep the K most frequent CWE types (incl
Nonetype), drop functions of rarer CWE. Label 0..K-1 = sorted(kept CWE). Split = seed42 80/10/10
(mirrors dataset_lm.get_splits; LIVABLE released no split for type classification).

Run:
    python scripts/livable_prepare_bigvul.py --csv data/LIVABLE/all_vul.csv \
        --out-dir bigvul_livable --top-cwe 31
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to all_vul.csv (Big-Vul)")
    ap.add_argument("--out-dir", required=True, help="output dir for LIVABLE inputs")
    ap.add_argument("--top-cwe", type=int, default=31,
                    help="keep the K most frequent CWE types (incl Nonetype). 0 = keep all ~91 "
                         "(PAPER-faithful: paper states 91 types, 10667 funcs, 8:1:1). 31 = "
                         "CODE-faithful (their hardcoded class_num / num_class_list, the joern-"
                         "survived populated subset). all_vul.csv = 10900 vuln, 89 distinct CWE.")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out_dir)

    df = pd.read_csv(a.csv, usecols=["func_before", "CWE ID", "target"])
    df = df[df["target"] == 1].reset_index(drop=True)          # vuln-only (all_vul already is)
    df = df.dropna(subset=["func_before"]).reset_index(drop=True)
    df["cwe_name"] = df["CWE ID"].fillna("Nonetype").astype(str).replace("", "Nonetype")

    # top-K CWE by frequency (Nonetype competes like any class); drop rarer-CWE functions.
    if a.top_cwe and a.top_cwe > 0:
        keep = set(df["cwe_name"].value_counts().head(a.top_cwe).index)
        before = len(df)
        df = df[df["cwe_name"].isin(keep)].reset_index(drop=True)
        print(f"  top-{a.top_cwe} CWE kept; dropped {before - len(df)} rarer-CWE funcs")

    cwe_labels = sorted(df["cwe_name"].unique())               # 0..K-1, stable order
    cwe_to_label = {c: i for i, c in enumerate(cwe_labels)}

    # seed42 80/10/10 split (mirror dataset_lm.get_splits: seeded shuffle of range(n)).
    idx = list(range(len(df)))
    random.seed(a.seed)
    random.shuffle(idx)
    n = len(idx)
    t, v = int(n * 0.8), int(n * 0.1)
    split_of = {}
    for i in idx[:t]:
        split_of[i] = "train"
    for i in idx[t:t + v]:
        split_of[i] = "val"
    for i in idx[t + v:]:
        split_of[i] = "test"

    func_dir = out / "functions"
    for s in ("train", "val", "test"):
        (func_dir / s).mkdir(parents=True, exist_ok=True)
    manifest = []
    jsonl = {s: [] for s in ("train", "val", "test")}
    for i, r in df.iterrows():
        split = split_of[i]
        label = cwe_to_label[r["cwe_name"]]
        src = str(r["func_before"])
        fname = f"{i}.c"
        (func_dir / split / fname).write_text(src, encoding="utf-8")
        manifest.append({"id": int(i), "split": split, "label": label, "cwe_name": r["cwe_name"]})
        jsonl[split].append({"file_path": fname, "func": src, "target": label,
                             "label": label, "cwe_name": r["cwe_name"]})

    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.json").write_text(json.dumps(manifest), encoding="utf-8")
    for s in ("train", "val", "test"):
        with open(out / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for rec in jsonl[s]:
                f.write(json.dumps(rec) + "\n")
    (out / "cwe_labels.json").write_text(
        json.dumps({"cwe_labels": cwe_labels, "num_classes": len(cwe_labels)}, indent=2),
        encoding="utf-8",
    )

    counts = {s: sum(1 for x in split_of.values() if x == s) for s in ("train", "val", "test")}
    print(f"wrote {len(df)} functions -> {func_dir}")
    print(f"  splits: {counts}")
    print(f"  num_classes: {len(cwe_labels)}")
    print(f"  cwe_labels (index order): {cwe_labels}")


if __name__ == "__main__":
    main()
