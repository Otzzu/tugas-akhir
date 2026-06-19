"""
build_megavul_cil_dataset.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Build the class-incremental (CIL) task-B dataset = MegaVul functions whose CWE is
in the Top-10 NON-top25 set (new classes, disjoint from task-A's Top-25 Most Dangerous).
Keeps func_graph_path so prepare_dataset can copy MegaVul's prebuilt CPGs (no Joern).

Output: data/datasets/megavul_cil/train.parquet (BigVul/megavul schema + func_graph_path).

Run:
    uv run python scripts/build_megavul_cil_dataset.py
Then generate CPG:
    uv run python scripts/prepare_dataset.py --input data/datasets/megavul_cil/train.parquet \
        --format megavul_cil --joern-cli C:/joern/joern-cli --out-dir data/raw --top-cwe 0 --workers 4
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "datasets"

# 10 new classes for CIL task-B (top non-top25 CWE by count in MegaVul).
NEW_CWE = ["CWE-119", "CWE-190", "CWE-362", "CWE-264", "CWE-399",
           "CWE-400", "CWE-401", "CWE-189", "CWE-617", "CWE-835"]
NEW = set(NEW_CWE)

def primary(x):
    for c in re.findall(r"CWE-\d+", str(x)):
        return c
    return None

cols = ["func_before", "func_after", "vul", "CWE ID", "language", "func_graph_path"]
mv = pq.read_table(DS / "megavul" / "train.parquet", columns=cols).to_pylist()

rows = []
for r in mv:
    if r["vul"] not in (1, True, "1"):
        continue
    c = primary(r["CWE ID"])
    if c not in NEW:
        continue
    rows.append({
        "func_before": r["func_before"],
        "func_after": r["func_after"] or "",
        "vul": 1,
        "CWE ID": c,
        "language": r["language"] or "C",
        "func_graph_path": r["func_graph_path"] or "",
    })

df = pd.DataFrame(rows).reset_index(drop=True)   # reset id so prebuilt-graph join aligns
out_dir = DS / "megavul_cil"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "train.parquet"
df.to_parquet(out, index=False)

print(f"Total CIL task-B: {len(df)}")
print(f"with prebuilt graph: {(df['func_graph_path'] != '').sum()}")
print("CWE distribution:\n" + df["CWE ID"].value_counts().to_string())
print(f"Saved -> {out}")
