"""Parquet hasil adapter → JSON format upload API.

Menghasilkan array DatasetRow yang bisa langsung dikirim ke POST /datasets/upload:

    [{"id": 0, "cve_id": "CVE-...", "code": "...", "cwe": "CWE-787",
      "flaw_lines": [3, 7], "func_after": "...", "language": "C"}, ...]

flaw_lines diturunkan dari diff func_before vs func_after, sama seperti pipeline .pt,
jadi API tidak perlu menghitungnya lagi. func_after tetap disertakan sebagai cadangan
(API memprioritaskan flaw_lines bila keduanya ada).

Usage:
    uv run python scripts/parquet_to_api_json.py \
        --input data/datasets/benchvul_cc/train.parquet \
        --out data/api_export/benchvul.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gnn_vuln.data.prepare import _diff_flaw_lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="parquet hasil adapter")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    rows = []
    n_flaw = 0
    for i, r in df.iterrows():
        before, after = str(r.func_before), str(r.func_after or "")
        flaw = _diff_flaw_lines(before, after) if after else []
        n_flaw += bool(flaw)
        row: dict = {
            "id": int(i),
            "code": before,
            "language": str(r.language),
        }
        if str(r.get("CVE ID") or ""):
            row["cve_id"] = str(r["CVE ID"])
        if int(r.vul) == 1:
            row["cwe"] = str(r["CWE ID"])
            if flaw:
                row["flaw_lines"] = flaw
            if after:
                row["func_after"] = after
        else:
            row["label"] = 0
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    size = args.out.stat().st_size / 1e6
    print(f"{len(rows)} baris -> {args.out} ({size:.1f} MB)")
    print(f"punya flaw_lines: {n_flaw} | rentan: {int((df.vul == 1).sum())} | "
          f"tidak rentan: {int((df.vul == 0).sum())}")


if __name__ == "__main__":
    main()
