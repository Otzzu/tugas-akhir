"""
build_relearn_dataset.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Build the continual-learning task-B dataset = BigVul ∪ TitanVul, restricted to
vulnerable C/C++ functions whose CWE is in the Top-25 Most Dangerous set,
deduplicated against MegaVul (task-A) and against each other.

Output schema matches BigVul / `--format merged` so prepare_dataset.py can build
CPGs and derive flaw lines from func_before/func_after diff:
    func_before, func_after, vul, CWE ID, language

Run:
    uv run python scripts/build_relearn_dataset.py
"""
from __future__ import annotations
import re, hashlib
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "datasets"

TOP25 = {"CWE-787","CWE-125","CWE-476","CWE-20","CWE-416","CWE-200","CWE-120","CWE-79",
"CWE-22","CWE-89","CWE-770","CWE-502","CWE-122","CWE-284","CWE-863","CWE-78","CWE-862",
"CWE-94","CWE-918","CWE-434","CWE-352","CWE-77","CWE-121","CWE-306","CWE-639"}
CC_EXT = {"c","cc","cpp","h","hpp","cxx","hh","c++"}

def norm(s: str) -> str:
    if s is None:
        return ""
    return hashlib.md5(re.sub(r"\s+", " ", str(s)).strip().encode("utf-8", "ignore")).hexdigest()

def cwes(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    return re.findall(r"CWE-\d+", str(x))

def primary_top25(x):
    for c in cwes(x):
        if c in TOP25:
            return c
    return None

# ---- MegaVul vulnerable hashes (task-A) — to exclude from task-B ----
mv = pq.read_table(DS / "megavul" / "train.parquet", columns=["func_before", "vul"]).to_pylist()
MVH = {norm(r["func_before"]) for r in mv if r["vul"] in (1, True, "1")}
print(f"MegaVul vuln hashes: {len(MVH)}")

rows = []
seen: set[str] = set()

def add(func_before, func_after, cwe, language):
    h = norm(func_before)
    if not func_before or h in MVH or h in seen:
        return False
    seen.add(h)
    rows.append({
        "func_before": func_before,
        "func_after": func_after if func_after else "",
        "vul": 1,
        "CWE ID": cwe,
        "language": language,
    })
    return True

# ---- BigVul ----
bv = pq.read_table(DS / "bigvul" / "all.parquet",
                   columns=["func_before","func_after","CWE ID","vul","language"]).to_pylist()
nb = 0
for r in bv:
    if r["vul"] not in (1, True, "1"):
        continue
    if str(r["language"]).lower() not in {"c","c++","cpp"}:
        continue
    c = primary_top25(r["CWE ID"])
    if not c:
        continue
    lang = "C++" if str(r["language"]).lower() in {"c++","cpp"} else "C"
    if add(r["func_before"], r["func_after"], c, lang):
        nb += 1
print(f"BigVul novel added: {nb}")

# ---- TitanVul ----
tv = pq.read_table(DS / "titanvul" / "raw.parquet",
                   columns=["func_before","func_after","cwe_id","extension"]).to_pylist()
nt = 0
for r in tv:
    ext = str(r["extension"]).lower()
    if ext not in CC_EXT:
        continue
    c = primary_top25(r["cwe_id"])
    if not c:
        continue
    lang = "C++" if ext in {"cc","cpp","cxx","hpp","c++"} else "C"
    if add(r["func_before"], r["func_after"], c, lang):
        nt += 1
print(f"TitanVul novel added (after mutual dedup): {nt}")

df = pd.DataFrame(rows)

# ---- Benign (class 0): add as many as the largest vuln class, from BigVul vul==0 ----
# Dedup against MegaVul ALL func_before (vuln + benign) to avoid task-A leakage.
target_n = int(df["CWE ID"].value_counts().max())
MVH_all = set(norm(r["func_before"]) for r in mv)
bvb = pq.read_table(DS / "bigvul" / "all.parquet",
                    columns=["func_before", "vul", "language"]).to_pylist()
import random as _random
_random.Random(42).shuffle(bvb)
nbn = 0
for r in bvb:
    if r["vul"] not in (0, False, "0"):
        continue
    if str(r["language"]).lower() not in {"c", "c++", "cpp"}:
        continue
    h = norm(r["func_before"])
    if not r["func_before"] or h in MVH_all or h in seen:
        continue
    seen.add(h)
    lang = "C++" if str(r["language"]).lower() in {"c++", "cpp"} else "C"
    rows.append({"func_before": r["func_before"], "func_after": "", "vul": 0,
                 "CWE ID": "", "language": lang})
    nbn += 1
    if nbn >= target_n:
        break
print(f"Benign added: {nbn} (target = largest vuln class {target_n})")

df = pd.DataFrame(rows)
out_dir = DS / "relearn"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "train.parquet"
df.to_parquet(out, index=False)

print(f"\nTotal combined: {len(df)}")
print(f"CWE distribution:\n{df['CWE ID'].value_counts().to_string()}")
print(f"Language: {df['language'].value_counts().to_dict()}")
print(f"Saved -> {out}")
