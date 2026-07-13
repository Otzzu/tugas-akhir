"""BenchVul → parquet berbentuk MegaVul, siap dipakai prepare_dataset.

BenchVul memakai skema sendiri (cwe_id, programming_language, tanpa kolom vul).
Skrip ini menyaring C/C++, memetakan kolomnya, dan MEMBUANG CWE yang tidak ada di
cwe_vocab MegaVul agar ruang label identik dengan model 26 kelas.

Seluruh baris BenchVul adalah fungsi rentan, jadi tidak ada kelas tidak rentan.

Usage:
    uv run python scripts/benchvul_to_parquet.py \
        --input data/datasets/benchvul/train.parquet \
        --cwe-vocab data/raw/megavul/cwe_vocab.json \
        --out-dir data/datasets/benchvul_cc
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CPP = {"c", "cpp", "c++"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--cwe-vocab", type=Path, required=True,
                    help="cwe_vocab.json MegaVul — ruang label acuan")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--keep-synth", action="store_true",
                    help="ikutkan fungsi hasil sintesis (default: ikut, flag ini hanya penanda)")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    n0 = len(df)

    df = df[df.programming_language.astype(str).str.lower().isin(CPP)]
    n_cc = len(df)

    vocab = json.loads(args.cwe_vocab.read_text(encoding="utf-8"))
    known = set(vocab) - {"benign"}
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
    # vocab MegaVul dipakai apa adanya agar id kelas sama persis dengan model 26 kelas
    (args.out_dir / "cwe_vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    print(f"benchvul {n0} baris -> C/C++ {n_cc} -> CWE dikenal {len(out)}")
    print(f"kelas: {out['CWE ID'].nunique()} | distribusi teratas: "
          f"{out['CWE ID'].value_counts().head(5).to_dict()}")
    print(f"tulis {args.out_dir/'train.parquet'} + cwe_vocab.json ({len(vocab)} kelas)")


if __name__ == "__main__":
    main()
