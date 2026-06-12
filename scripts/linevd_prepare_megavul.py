"""
linevd_prepare_megavul.py — convert our exported baseline split into LineVD's native
cache files so its `bigvul()` loads OUR MegaVul data with ZERO code edits.

LineVD's bigvul(minimal=True) reads:
  {cache_dir}/minimal_datasets/minimal_bigvul_False.pq   (the dataframe)
  {external_dir}/bigvul_rand_splits.csv                  (id -> label split map)
  {cache_dir}/bigvul/bigvul_metadata.csv                 (id -> project; groupby only)

We produce all three from data/baselines/.../linevd/{train,val,test}.parquet.
Flaw GT = our precomputed `flaw_lines` -> the `removed` column (LineVD's vuln-line source
via get_vuln_indices), so its statement labels match OUR model's exactly. func_after is
unavailable -> dummied to func_before (flaw GT comes from `removed`, not a diff).

Run ON THE POD from inside the cloned linevd repo (so `import sastvd` resolves):
    PYTHONPATH=. python /path/to/linevd_prepare_megavul.py --in-dir /path/megavul_ml1024/linevd
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sastvd as svd  # resolves when run from the linevd repo root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="dir with train/val/test.parquet (our linevd export)")
    a = ap.parse_args()
    ind = Path(a.in_dir)

    frames = []
    for split in ("train", "val", "test"):
        df = pd.read_parquet(ind / f"{split}.parquet")
        df["label"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Map our columns -> LineVD bigvul schema
    out = pd.DataFrame()
    out["id"] = df["id"].astype(int)
    out["vul"] = df["vul"].astype(int)
    out["func_before"] = df["func_before"].astype(str)
    out["func_after"] = df["func_before"].astype(str)   # dummy (no after; GT from `removed`)
    out["before"] = df["func_before"].astype(str)
    out["after"] = df["func_before"].astype(str)
    # removed = flaw line numbers (1-indexed) -> LineVD get_vuln_indices uses this
    out["removed"] = df["flaw_lines"].apply(lambda x: [int(i) for i in (x if x is not None else [])])
    out["added"] = [[] for _ in range(len(out))]
    out["diff"] = ""
    out["dataset"] = "megavul"
    out["label"] = df["label"].values

    cache = svd.cache_dir()
    ext = svd.external_dir()
    (cache / "minimal_datasets").mkdir(parents=True, exist_ok=True)
    (cache / "bigvul").mkdir(parents=True, exist_ok=True)
    ext.mkdir(parents=True, exist_ok=True)

    # 1. main dataframe (fastparquet — bigvul reads with engine="fastparquet")
    out.to_parquet(cache / "minimal_datasets" / "minimal_bigvul_False.pq", engine="fastparquet")
    # 2. split map
    out[["id", "label"]].to_csv(ext / "bigvul_rand_splits.csv", index=False)
    # 3. metadata (project col; bigvul only groupby-counts it)
    out[["id"]].assign(project="megavul").to_csv(cache / "bigvul" / "bigvul_metadata.csv", index=False)

    n = len(out)
    print(f"wrote {n} rows -> {cache/'minimal_datasets'/'minimal_bigvul_False.pq'}")
    print(f"  splits: {out.label.value_counts().to_dict()}")
    print(f"  vuln:   {int(out.vul.sum())} / {n}")
    print(f"  splits.csv -> {ext/'bigvul_rand_splits.csv'}")
    print("Next: getgraphs.py (Joern) -> codebert embeddings -> train_best.py")


if __name__ == "__main__":
    main()
