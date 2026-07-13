"""BenchVul → parquet berbentuk MegaVul, siap dipakai prepare.

BenchVul memakai skema sendiri (cwe_id, programming_language, tanpa kolom vul).
Skrip ini menyaring C/C++, memetakan kolomnya, dan MEMBUANG CWE yang tidak ada di
ruang label MegaVul 26 kelas.

Vocab DITANAM di skrip ini, bukan dibaca dari file. Salinan cwe_vocab.json yang
beredar sempat berbeda (versi lama 193 kelas dengan id yang bergeser), dan memakai
yang salah membuat seluruh label meleset tanpa error apa pun.

Ruang label = Top 25 Most Dangerous CWE versi 2025 (CWE-1435) + kelas tidak rentan.
BenchVul memakai daftar Top 25 tahun lain, sehingga CWE-190, CWE-269, CWE-400, dan
CWE-798 tidak punya kelas pada model dan dibuang. Sisa ~156 fungsi dari 230 C/C++.

Usage:
    uv run python scripts/benchvul_to_parquet.py \
        --input data/datasets/benchvul/train.parquet \
        --out-dir data/datasets/benchvul_cc
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CPP = {"c", "cpp", "c++"}

# Ruang label model 26 kelas. Urutannya menentukan id, jadi JANGAN diubah.
CWE_VOCAB: dict[str, int] = {
    "benign": 0,
    "CWE-787": 1,
    "CWE-125": 2,
    "CWE-476": 3,
    "CWE-20": 4,
    "CWE-416": 5,
    "CWE-200": 6,
    "CWE-120": 7,
    "CWE-79": 8,
    "CWE-22": 9,
    "CWE-89": 10,
    "CWE-770": 11,
    "CWE-502": 12,
    "CWE-122": 13,
    "CWE-284": 14,
    "CWE-863": 15,
    "CWE-78": 16,
    "CWE-862": 17,
    "CWE-94": 18,
    "CWE-918": 19,
    "CWE-434": 20,
    "CWE-352": 21,
    "CWE-77": 22,
    "CWE-121": 23,
    "CWE-306": 24,
    "CWE-639": 25,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    n0 = len(df)

    df = df[df.programming_language.astype(str).str.lower().isin(CPP)]
    n_cc = len(df)

    known = set(CWE_VOCAB) - {"benign"}
    df = df[df.cwe_id.isin(known)]

    out = pd.DataFrame({
        "func_before": df.func_before.astype(str),
        "func_after": df.func_after.astype(str),
        "vul": 1,
        "CWE ID": df.cwe_id.astype(str),
        "CVE ID": df.cve_id.astype(str),
        "language": df.programming_language.astype(str).str.upper().str.replace("CPP", "C++"),
    }).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_dir / "train.parquet", index=False)
    (args.out_dir / "cwe_vocab.json").write_text(json.dumps(CWE_VOCAB, indent=2), encoding="utf-8")

    print(f"benchvul {n0} baris -> C/C++ {n_cc} -> CWE dikenal {len(out)}")
    print(f"kelas hadir: {out['CWE ID'].nunique()} dari 25 | teratas: "
          f"{out['CWE ID'].value_counts().head(5).to_dict()}")
    print(f"tulis {args.out_dir/'train.parquet'} + cwe_vocab.json ({len(CWE_VOCAB)} kelas)")


if __name__ == "__main__":
    main()
