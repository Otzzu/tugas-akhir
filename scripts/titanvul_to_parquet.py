"""TitanVul → parquet berbentuk MegaVul, siap dipakai prepare.

Tiga filter:
  1. bahasa C/C++ (dari kolom extension)
  2. CWE ada di ruang label MegaVul 26 kelas (Top 25 Most Dangerous versi 2025)
  3. TIDAK tumpang tindih dengan MegaVul — fungsi yang teksnya sama (setelah
     normalisasi whitespace) dibuang, supaya dataset ini benar-benar baru terhadap
     data latih model

Vocab DITANAM di skrip (sama seperti benchvul_to_parquet.py), tidak dibaca dari file,
karena salinan cwe_vocab.json yang beredar sempat berbeda dan memakainya membuat label
meleset tanpa error.

Seluruh baris TitanVul adalah fungsi rentan, jadi tidak ada kelas tidak rentan.

Usage:
    uv run python scripts/titanvul_to_parquet.py \
        --input data/datasets/titanvul/raw.parquet \
        --megavul data/datasets/megavul/train.parquet \
        --out-dir data/datasets/titanvul_cc
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

CC_EXT = {"c", "cc", "cpp", "h", "hpp", "cxx", "hh", "c++"}
CPP_EXT = {"cc", "cpp", "cxx", "hpp", "hh", "c++"}

# Ruang label model 26 kelas. Urutannya menentukan id, jadi JANGAN diubah.
CWE_VOCAB: dict[str, int] = {
    "benign": 0, "CWE-787": 1, "CWE-125": 2, "CWE-476": 3, "CWE-20": 4, "CWE-416": 5,
    "CWE-200": 6, "CWE-120": 7, "CWE-79": 8, "CWE-22": 9, "CWE-89": 10, "CWE-770": 11,
    "CWE-502": 12, "CWE-122": 13, "CWE-284": 14, "CWE-863": 15, "CWE-78": 16,
    "CWE-862": 17, "CWE-94": 18, "CWE-918": 19, "CWE-434": 20, "CWE-352": 21,
    "CWE-77": 22, "CWE-121": 23, "CWE-306": 24, "CWE-639": 25,
}
TOP25 = set(CWE_VOCAB) - {"benign"}


def norm(s: str) -> str:
    """Hash teks fungsi setelah whitespace diseragamkan — dasar dedup lintas dataset."""
    return hashlib.md5(re.sub(r"\s+", " ", str(s)).strip().encode("utf-8", "ignore")).hexdigest()


def primary_cwe(x) -> str | None:
    """CWE pertama yang ada di ruang label. Satu entri bisa memuat beberapa CWE."""
    for c in re.findall(r"CWE-\d+", str(x)):
        if c in TOP25:
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/datasets/titanvul/raw.parquet"))
    ap.add_argument("--megavul", type=Path, default=Path("data/datasets/megavul/train.parquet"),
                    help="dipakai untuk dedup — fungsi yang sudah ada di MegaVul dibuang")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    mv = pq.read_table(args.megavul, columns=["func_before"]).to_pylist()
    mv_hash = {norm(r["func_before"]) for r in mv}
    print(f"hash MegaVul: {len(mv_hash)}")

    rows = pq.read_table(
        args.input, columns=["func_before", "func_after", "cwe_id", "extension", "cve_id"]
    ).to_pylist()

    out, seen = [], set()
    n_cc = n_top = 0
    for r in rows:
        ext = str(r["extension"]).lower()
        if ext not in CC_EXT:
            continue
        n_cc += 1
        cwe = primary_cwe(r["cwe_id"])
        if not cwe:
            continue
        n_top += 1
        h = norm(r["func_before"])
        if not r["func_before"] or h in mv_hash or h in seen:
            continue
        seen.add(h)
        out.append({
            "func_before": r["func_before"],
            "func_after": r["func_after"] or "",
            "vul": 1,
            "CWE ID": cwe,
            "CVE ID": str(r.get("cve_id") or ""),
            "language": "C++" if ext in CPP_EXT else "C",
        })

    df = pd.DataFrame(out)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_dir / "train.parquet", index=False)
    (args.out_dir / "cwe_vocab.json").write_text(json.dumps(CWE_VOCAB, indent=2), encoding="utf-8")

    print(f"titanvul {len(rows)} -> C/C++ {n_cc} -> Top25 {n_top} -> baru (bukan MegaVul) {len(df)}")
    print(f"kelas hadir: {df['CWE ID'].nunique()} dari 25 | teratas: "
          f"{df['CWE ID'].value_counts().head(5).to_dict()}")
    print(f"punya func_after (flaw lines bisa didiff): {(df.func_after.str.len() > 0).sum()}")
    print(f"tulis {args.out_dir/'train.parquet'} + cwe_vocab.json ({len(CWE_VOCAB)} kelas)")


if __name__ == "__main__":
    main()
