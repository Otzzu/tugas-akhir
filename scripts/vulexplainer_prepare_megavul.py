"""
vulexplainer_prepare_megavul.py — convert OUR megavul split into VulExplainer's Big-Vul csv format so
VulExplainer (TSE'23, hierarchical CNN-teacher -> GraphCodeBERT-student distillation) trains on our
split + classes -> comparable VTP head-to-head (classification only, no localization).

FAITHFULNESS: model code (teacher_main / student_graphcodebert_main / textcnn / distillation) UNTOUCHED.
Only data swapped + cwe_label_map regenerated for our N vuln classes. Vuln-only (no benign), function
granularity, same 80/10/10 seed42 split as our model.

VulExplainer reads 3 csv columns: func_before, "CWE ID", cwe_abstract_group. The teacher groups by CWE
ABSTRACTION LEVEL (group_label_map hardcoded = category/class/variant/base/deprecated/pillar). Each CWE's
abstraction is a fixed MITRE property -> CWE_ABS below (the 12 CWEs present in their big_vul keep THEIR
label for consistency; the 13 not in their 44 filled from MITRE; unknown -> "base" + warning).

cwe_label_map.pkl = dict {cwe_str: [label_idx, one_hot(N), 0]} (3rd = freq counter, code increments it).

Input : {in_dir}/{train,val,test}.parquet  (func_before, vul, cwe_name, id)
Output (overwrites their data/big_vul/): {out_dir}/{train,val,test}.csv + cwe_label_map.pkl
Run: python scripts/vulexplainer_prepare_megavul.py --in-dir megavul_ml1024/linevd \
        --out-dir src/VulExplainer/data/big_vul
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

# CWE -> MITRE abstraction level. 12 from VulExplainer's own big_vul (kept for faithfulness),
# 13 filled from MITRE CWE taxonomy. Compound (CWE-352 CSRF) -> "base" (no compound slot in their 6).
CWE_ABS = {
    # present in their big_vul 44 (their labels)
    "CWE-787": "base", "CWE-125": "base", "CWE-20": "class", "CWE-476": "base",
    "CWE-416": "variant", "CWE-200": "class", "CWE-79": "base", "CWE-22": "base",
    "CWE-284": "deprecated", "CWE-78": "base", "CWE-94": "base", "CWE-77": "class",
    # not in their 44 -> MITRE abstraction
    "CWE-120": "base", "CWE-89": "base", "CWE-770": "base", "CWE-502": "base",
    "CWE-122": "variant", "CWE-863": "class", "CWE-862": "class", "CWE-918": "base",
    "CWE-434": "base", "CWE-352": "base", "CWE-121": "variant", "CWE-306": "base",
    "CWE-639": "base",
}
VALID_ABS = {"category", "class", "variant", "base", "deprecated", "pillar"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    ind, out = Path(a.in_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = {}
    for s in ("train", "val", "test"):
        df = pd.read_parquet(ind / f"{s}.parquet")
        df = df[df["vul"] == 1].copy()                          # vuln-only, no benign
        df["CWE ID"] = df["cwe_name"].astype(str).str.strip()
        df = df[df["CWE ID"].str.startswith("CWE-")]            # drop NaN/others/empty
        frames[s] = df

    # class set = sorted unique vuln CWE across all splits; stable idx by sorted order
    classes = sorted(set().union(*[set(frames[s]["CWE ID"]) for s in frames]))
    N = len(classes)
    c2i = {c: i for i, c in enumerate(classes)}
    cwe_label_map = {c: [c2i[c], [1 if j == c2i[c] else 0 for j in range(N)], 0] for c in classes}

    fallback = []
    def abslevel(c: str) -> str:
        v = CWE_ABS.get(c)
        if v is None or v not in VALID_ABS:
            fallback.append(c)
            return "base"
        return v

    for s in ("train", "val", "test"):
        df = frames[s]
        df["cwe_abstract_group"] = df["CWE ID"].map(abslevel)
        df[["func_before", "CWE ID", "cwe_abstract_group"]].to_csv(out / f"{s}.csv", index=False)
        print(f"  {s}: {len(df)} funcs")

    with open(out / "cwe_label_map.pkl", "wb") as f:
        pickle.dump(cwe_label_map, f)

    print(f"  num_classes={N}  classes={classes}")
    if fallback:
        print(f"  WARN fallback 'base' abstraction for unmapped CWEs: {sorted(set(fallback))}")
    print(f"  -> {out} (train/val/test.csv + cwe_label_map.pkl)")


if __name__ == "__main__":
    main()
