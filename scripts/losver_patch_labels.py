"""
losver_patch_labels.py — swap LOSVER's hardcoded BigVul CWE label set for OUR MegaVul set
in the run_*_CWE.py scripts, so it classifies our 25 CWE classes (not BigVul's 36).

Replaces every `cwe_labels = [...]` assignment and every `num_labels=<N>` / `num_labels = <N>`
default in the target files.

Run (pod, from inside src/losver):
    python /path/scripts/losver_patch_labels.py \
        --files classification/run_line_CWE.py classification/run_weighted_CWE.py classification/run_base_CWE.py \
        --labels '["CWE-787","CWE-125",...]'
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--labels", required=True, help="JSON list of CWE label strings")
    a = ap.parse_args()

    labels = json.loads(a.labels)
    n = len(labels)
    new_list = "cwe_labels = " + json.dumps(labels)

    list_re = re.compile(r"cwe_labels\s*=\s*\[.*?\]", re.DOTALL)
    num_re = re.compile(r"num_labels\s*=\s*\d+")

    for fp in a.files:
        p = Path(fp)
        if not p.exists():
            print(f"  skip (missing): {fp}")
            continue
        src = p.read_text(encoding="utf-8")
        n_list = len(list_re.findall(src))
        n_num = len(num_re.findall(src))
        src = list_re.sub(new_list, src)
        src = num_re.sub(f"num_labels={n}", src)
        p.write_text(src, encoding="utf-8")
        print(f"  patched {fp}: cwe_labels x{n_list}, num_labels x{n_num} -> {n} classes")


if __name__ == "__main__":
    main()
