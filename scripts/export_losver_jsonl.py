"""
export_losver_jsonl.py — convert our exported baseline split into LOSVER's CWE jsonl,
using OUR precomputed flaw lines as lines_ground_truth (bypass LOSVER's func_after diff,
so its line GT == our model's). Vuln-only (LOSVER CWE classification filters vul==1).

Reads  our LineVD parquets: {in_dir}/{train,val,test}.parquet
       cols: func_before, vul, cwe_name, flaw_lines  (1-indexed source lines)
Writes {out_dir}/CWE_{split}_unix_512.jsonl   (token-limit-filtered, ready for run_*_CWE.py)
       each line: {"func_before", "func_after", "CWE ID", "lines_ground_truth"}
       lines_ground_truth = 0-indexed modifiable lines (our flaw_lines - 1)

Mirrors LOSVER bigvul_preprocess.convert_examples_to_features token filter: keep only lines
that fit in token_limit, drop functions with no modifiable line inside the limit.

After running, swap CWE label set in run_base_CWE.py / run_weighted_CWE.py / run_line_CWE.py
to the printed `cwe_labels` list + set num_labels accordingly.

Run:
    PYTHONPATH=src python scripts/export_losver_jsonl.py \
        --in-dir data/baselines/megavul_ml1024/linevd \
        --out-dir data/baselines/megavul_ml1024/losver \
        --tokenizer microsoft/unixcoder-base-nine --token-limit 512
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def line_filter(func_before: str, lines_gt: list[int], tokenizer, token_limit: int):
    """Replicate LOSVER convert_examples_to_features: per-line tokenize, keep lines that
    fit in token_limit; return True if >=1 modifiable line survives inside the limit."""
    lines = func_before.splitlines()
    if not lines or any(l < 0 for l in lines_gt):
        return False
    sparse = [1 if i in lines_gt else 0 for i in range(len(lines))][:token_limit]
    sparse += [-1] * (token_limit - len(lines))
    n_tok = 0
    for ind, line in enumerate(lines):
        lt = tokenizer.tokenize(line + "\n")
        if n_tok + len(lt) <= token_limit - 2:
            n_tok += len(lt)
        else:
            for i in range(ind, len(sparse)):
                sparse[i] = -1
            break
    return 1 in sparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="dir with train/val/test.parquet (our linevd export)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", default="microsoft/unixcoder-base-nine")
    ap.add_argument("--token-limit", type=int, default=512)
    a = ap.parse_args()

    ind = Path(a.in_dir)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    try:
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    except Exception:
        print(f"!! {a.tokenizer} failed, falling back to microsoft/unixcoder-base", file=sys.stderr)
        tok = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")

    all_cwes: Counter = Counter()
    for split in ("train", "val", "test"):
        df = pd.read_parquet(ind / f"{split}.parquet")
        df = df[df["vul"] == 1]                       # LOSVER CWE = vuln-only
        kept = 0
        with open(out / f"CWE_{split}_unix_{a.token_limit}.jsonl", "w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                fb = str(r["func_before"])
                flaw = [int(x) for x in (r["flaw_lines"] if r["flaw_lines"] is not None else [])]
                lines_gt = sorted({l - 1 for l in flaw if l >= 1})   # 1-indexed -> 0-indexed
                if not lines_gt:
                    continue
                if not line_filter(fb, lines_gt, tok, a.token_limit):
                    continue
                cwe = str(r["cwe_name"]) or "others"
                f.write(json.dumps({
                    "func_before": fb,
                    "func_after": fb,            # dummy; GT comes from lines_ground_truth
                    "CWE ID": cwe,
                    "lines_ground_truth": lines_gt,
                }) + "\n")
                all_cwes[cwe] += 1
                kept += 1
        print(f"  {split}: {kept} vuln rows -> CWE_{split}_unix_{a.token_limit}.jsonl", file=sys.stderr)

    cwe_labels = [c for c, _ in all_cwes.most_common()]
    print(f"\nnum_labels = {len(cwe_labels)}", file=sys.stderr)
    print("cwe_labels = " + json.dumps(cwe_labels), file=sys.stderr)
    print("^ paste into run_base_CWE.py / run_weighted_CWE.py / run_line_CWE.py", file=sys.stderr)


if __name__ == "__main__":
    main()
