"""
edat_prepare_megavul.py — convert OUR megavul split into EDAT's JSONL inputs so EDAT (VTP + LVD,
GraphCodeBERT + MTL + adversarial) trains on our split + classes -> comparable head-to-head on BOTH
tasks (classification + line-level), unlike LIVABLE/VulPCL (classification only).

EDAT is text-based (NO joern). Two JSONL per split:
  VTP (classification): {"func_before": src, "CWE ID": <cwe|benign>, "commit_id": <id>}
  LVD (line-level):     per source line -> {"line_text", "is_vulnerable_line": 0/1,
        "line_number_in_function": i(1-idx), "original_vul": <full func>, "commit_id","file_name",
        "original_function_identifier"}  (fields the graphcodebert-context _extract_context needs).

26-class (benign + 25 CWE) to match our model. is_vulnerable_line = line index in flaw_lines.

Input : {in_dir}/{train,val,test}.parquet  (func_before, vul, cwe_name, flaw_lines, id)
Output: {out_dir}/{train,val,test}.jsonl                (VTP)
        {out_dir}/{train,val,test}_line_level.jsonl      (LVD)
Run: python scripts/edat_prepare_megavul.py --in-dir megavul_ml1024/linevd --out-dir megavul_edat
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flaw_set(v) -> set[int]:
    if v is None:
        return set()
    try:
        return {int(x) for x in list(v)}
    except Exception:
        return set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    ind, out = Path(a.in_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        df = pd.read_parquet(ind / f"{split}.parquet")
        df["cwe_name"] = df["cwe_name"].fillna("others").replace("", "others").astype(str)
        vtp_f = open(out / f"{split}.jsonl", "w", encoding="utf-8")
        lvd_f = open(out / f"{split}_line_level.jsonl", "w", encoding="utf-8")
        n_v = n_l = 0
        for idx, r in df.iterrows():
            fid = str(int(r["id"]) if "id" in df.columns and pd.notna(r.get("id")) else idx)
            src = str(r["func_before"])
            cwe = "benign" if int(r["vul"]) == 0 else str(r["cwe_name"])
            vtp_f.write(json.dumps({"func_before": src, "CWE ID": cwe, "commit_id": fid}) + "\n")
            n_v += 1
            flaws = flaw_set(r.get("flaw_lines"))
            for i, line in enumerate(src.split("\n"), start=1):     # 1-indexed lines
                lvd_f.write(json.dumps({
                    "line_text": line,
                    "is_vulnerable_line": 1 if i in flaws else 0,
                    "line_number_in_function": i,
                    "original_vul": src,
                    "commit_id": fid,
                    "file_name": fid,
                    "original_function_identifier": fid,
                }) + "\n")
                n_l += 1
        vtp_f.close(); lvd_f.close()
        print(f"  {split}: {n_v} funcs (VTP), {n_l} lines (LVD)")
    print(f"  -> {out}  (classes = benign + 25 CWE = 26)")


if __name__ == "__main__":
    main()
