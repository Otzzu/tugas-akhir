"""
analyze_dataset.py
~~~~~~~~~~~~~~~~~~
Re-generate DATASET_ANALYSIS.md from raw parquet files using the current
CWE_GROUP_MAP from dataset_lm.py.

Usage:
    uv run python scripts/analyze_dataset.py
    uv run python scripts/analyze_dataset.py --out DATASET_ANALYSIS.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_vuln.data.dataset_lm import CWE_GROUP_MAP, GROUP_VOCAB

# GROUP_VOCAB is str->int (name -> id)
GROUP_NAME_TO_ID = GROUP_VOCAB  # name -> int


def cwe_to_group(cwe_str: str) -> tuple[int, str]:
    """Map CWE string → (group_id, group_name). Returns (-1, 'UNKNOWN') if unmapped."""
    if not cwe_str or not isinstance(cwe_str, str):
        return -1, "UNKNOWN"
    key = cwe_str.strip()
    if not key.startswith("CWE-"):
        key = f"CWE-{key}"
    group_name = CWE_GROUP_MAP.get(key)
    if group_name is None:
        return -1, "UNKNOWN"
    gid = GROUP_NAME_TO_ID.get(group_name, -1)
    return gid, group_name


def parse_cwe(raw) -> str:
    """Normalise CWE field to 'CWE-NNN' string. Extracts primary CWE if comma-separated."""
    if pd.isna(raw) or raw is None:
        return ""
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    s = str(raw).strip().upper()
    if "," in s:
        s = s.split(",")[0].strip()
    if not s or s in ("NAN", "NONE", "UNKNOWN", "OTHER", "CWE-OTHER", "CWE-UNKNOWN", "NVD-CWE-NOINFO", "NVD-CWE-OTHER"):
        return ""
    if s.startswith("CWE-"):
        num = s[4:]
        if num.isdigit():
            return f"CWE-{int(num)}"
        return s
        
    if s.isdigit():
        return f"CWE-{int(s)}"
        
    return s


def get_cwe_set_from_xml(filepath: Path) -> set[str]:
    """Parse CWE XML files and return a set of CWE string IDs."""
    if not filepath.exists():
        return set()
    try:
        content = filepath.read_text(encoding="utf-8")
        cwes = re.findall(r'<Weakness [^>]*ID="(\d+)"', content)
        return set(f"CWE-{x}" for x in cwes)
    except Exception:
        return set()


def group_dist_table(group_counts: dict[tuple[int, str], int], cwe_rows: list[tuple[str, int, int, str]] = None, owasp_map: dict[str, str] = None) -> str:
    rows = sorted(group_counts.items(), key=lambda x: -x[1])
    
    if cwe_rows and owasp_map:
        import re
        gid_to_owasp = defaultdict(set)
        for cwe, cnt, gid, gname in cwe_rows:
            if cwe in owasp_map:
                m = re.match(r'(A\d+)', owasp_map[cwe])
                if m:
                    gid_to_owasp[gid].add(m.group(1))
                else:
                    gid_to_owasp[gid].add(owasp_map[cwe])
                    
        lines = ["| Group ID | Group | Count | OWASP Top 10 |", "|---|---|---|---|"]
        for (gid, gname), cnt in rows:
            owasps = ", ".join(sorted(gid_to_owasp.get(gid, [])))
            lines.append(f"| {gid} | {gname} | {cnt:,} | {owasps} |")
    else:
        lines = ["| Group ID | Group | Count |", "|---|---|---|"]
        for (gid, gname), cnt in rows:
            lines.append(f"| {gid} | {gname} | {cnt:,} |")
            
    lines.append(f"\n> **Unique Groups**: {len(rows):,}")
    return "\n".join(lines)


def get_owasp_category_map(filepath: Path) -> dict[str, str]:
    if not filepath.exists():
        return {}
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        mapping = {}
        categories = root.findall('.//{http://cwe.mitre.org/cwe-7}Category')
        for c in categories:
            name = c.attrib.get('Name')
            if not name or "OWASP Top Ten 2025 Category" not in name:
                continue
            clean_name = name.replace("OWASP Top Ten 2025 Category ", "")
            members = c.findall('.//{http://cwe.mitre.org/cwe-7}Has_Member')
            for m in members:
                cwe_id = m.attrib.get('CWE_ID')
                if cwe_id:
                    mapping[f"CWE-{cwe_id}"] = clean_name
        return mapping
    except Exception:
        return {}

def subset_dist_table(cwe_rows: list[tuple[str, int, int, str]], subset: set[str], subset_name: str, group_override: dict[str, str] = None) -> str:
    subset_rows = [r for r in cwe_rows if r[0] in subset]
    if not subset_rows:
        return f"*(No {subset_name} CWEs found)*"
        
    if group_override:
        lines = ["| CWE | Count | Group | OWASP |", "|---|---|---|---|"]
    else:
        lines = ["| CWE | Count | Group ID | Group |", "|---|---|---|---|"]
        
    total_cnt = 0
    group_counts = defaultdict(int)
    for cwe, cnt, gid, gname in subset_rows:
        if group_override and cwe in group_override:
            owasp_cat = group_override[cwe]
            # gname is still the original functional group from CWE_GROUP_MAP
            lines.append(f"| {cwe} | {cnt:,} | {gname} | {owasp_cat} |")
            group_counts[owasp_cat] += cnt
        else:
            lines.append(f"| {cwe} | {cnt:,} | {gid} | {gname} |")
            group_counts[(gid, gname)] += cnt
            
        total_cnt += cnt
        
    lines.append(f"| **Total** | **{total_cnt:,}** | | |")
    
    lines.append(f"\n> **Unique CWEs**: {len(subset_rows):,}")
    lines.append(f"> **Unique Groups**: {len(group_counts):,}")
    
    lines.append(f"\n#### {subset_name} Group Distribution\n")
    if group_override:
        lines.append("| OWASP Category | Count |")
        lines.append("|---|---|")
        for owasp_cat, cnt in sorted(group_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {owasp_cat} | {cnt:,} |")
    else:
        lines.append("| Group ID | Group | Count |")
        lines.append("|---|---|---|")
        for (gid, gname), cnt in sorted(group_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {gid} | {gname} | {cnt:,} |")
            
    lines.append(f"| **Total** | **{total_cnt:,}** |")
    
    return "\n".join(lines)


def cwe_dist_table(cwe_rows: list[tuple[str, int, int, str]], top25: set[str], owasp: set[str]) -> str:
    lines = ["| CWE | Count | Group ID | Group | Top 25 | OWASP Top 10 |", "|---|---|---|---|---|---|"]
    unique_groups = set()
    for cwe, cnt, gid, gname in cwe_rows:
        label = cwe if cwe else "*(empty/unknown)*"
        is_top25 = "✅" if cwe in top25 else ""
        is_owasp = "✅" if cwe in owasp else ""
        lines.append(f"| {label} | {cnt:,} | {gid} | {gname} | {is_top25} | {is_owasp} |")
        if cwe:
            unique_groups.add(gid)
    
    unique_cwes = len([c for c in cwe_rows if c[0]])
    lines.append(f"\n> **Unique CWEs**: {unique_cwes:,}")
    lines.append(f"> **Unique Groups**: {len(unique_groups):,}")
    return "\n".join(lines)


def analyze_df(df: pd.DataFrame, cwe_col: str, label_col: str | None = None,
               multi_label: bool = False) -> tuple[dict, list, int, int]:
    """
    Returns (group_counts, cwe_rows, n_benign, n_vuln).
    group_counts: {(gid, gname): count}
    cwe_rows: [(cwe_str, count, gid, gname), ...]
    """
    if label_col and label_col in df.columns:
        vuln_mask = df[label_col] > 0
    else:
        vuln_mask = df[cwe_col].notna() & (df[cwe_col].astype(str).str.strip().isin(["", "nan", "None"]) == False)

    benign_df = df[~vuln_mask]
    vuln_df = df[vuln_mask]
    n_benign = len(benign_df)
    n_vuln = len(vuln_df)

    group_counts: dict[tuple[int, str], int] = defaultdict(int)
    group_counts[(0, "benign")] = n_benign

    cwe_counter: Counter = Counter()
    cwe_to_grp: dict[str, tuple[int, str]] = {}

    for _, row in vuln_df.iterrows():
        raw = row[cwe_col]
        if multi_label and isinstance(raw, list):
            raw = raw[0] if raw else ""
        cwe_str = parse_cwe(raw)
        cwe_counter[cwe_str] += 1
        if cwe_str not in cwe_to_grp:
            cwe_to_grp[cwe_str] = cwe_to_group(cwe_str)
        gid, gname = cwe_to_grp[cwe_str]
        if not cwe_str:
            group_counts[(-1, "UNKNOWN")] += 1
        else:
            group_counts[(gid, gname)] += 1

    cwe_rows = []
    for cwe_str, cnt in cwe_counter.most_common():
        gid, gname = cwe_to_grp.get(cwe_str, (-1, "UNKNOWN"))
        cwe_rows.append((cwe_str, cnt, gid, gname))

    return dict(group_counts), cwe_rows, n_benign, n_vuln


def language_dist_from_parquet(
    df: pd.DataFrame,
    func_col: str = "func_before",
    lang_col: str | None = None,
    n_sample: int = 500,
) -> Counter:
    """Detect language distribution from a parquet DataFrame."""
    from gnn_vuln.data.joern_runner import detect_language
    # Normalize lang values to detect_language extension format (lowercase, cpp not C++)
    _LANG_NORM = {"C": "c", "CPP": "cpp", "C++": "cpp", "JAVA": "java",
                  "JAVASCRIPT": "js", "JS": "js", "PYTHON": "py", "GO": "go",
                  "RUBY": "rb", "PHP": "php", "KOTLIN": "kt", "CSHARP": "cs", "C#": "cs"}
    counts: Counter = Counter()
    if lang_col and lang_col in df.columns:
        for v in df[lang_col].dropna():
            key = str(v).strip().upper()
            counts[_LANG_NORM.get(key, key.lower())] += 1
        return counts
    sample = df[func_col].dropna().sample(min(n_sample, len(df)), random_state=42)
    for code in sample:
        counts[detect_language(str(code))] += 1
    return counts


def language_dist_from_raw(raw_dir: Path, n_sample: int = 500) -> Counter:
    """Detect language distribution from raw meta.json raw_func fields."""
    from gnn_vuln.data.joern_runner import detect_language
    meta_files = list(raw_dir.rglob("*.meta.json"))
    if not meta_files:
        return Counter()
    sample = random.sample(meta_files, min(n_sample, len(meta_files)))
    counts: Counter = Counter()
    for f in sample:
        try:
            meta = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
            code = meta.get("raw_func", "")
            if code:
                counts[detect_language(code)] += 1
        except Exception:
            pass
    return counts


def language_table(dists: dict[str, Counter]) -> str:
    """Format language distributions as a markdown table."""
    all_langs = sorted({lang for c in dists.values() for lang in c})
    header = "| Dataset | " + " | ".join(all_langs) + " |"
    sep    = "|---|" + "|".join(["---"] * len(all_langs)) + "|"
    lines  = [header, sep]
    for ds, counts in dists.items():
        total = sum(counts.values())
        cells = []
        for lang in all_langs:
            n = counts.get(lang, 0)
            cells.append(f"{n/total*100:.0f}%" if total and n else "—")
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    lines.append("\n> Detected via `joern_runner.detect_language()` on sampled `raw_func` / `func_before` fields.")
    lines.append("> Go/PHP fall back to C extension (Joern native frontends produce sparser CPGs).")
    return "\n".join(lines)


def cpg_coverage_table(raw_dir: Path, group_counts: dict) -> str:
    """Build CPG coverage table comparing data/raw/<source>/ files vs parquet."""
    vuln_dir = raw_dir / "vulnerable"
    lines = ["| Group ID | Group | CPG Files | all.parquet | Coverage |", "|---|---|---|---|---|"]

    benign_dir = raw_dir / "benign"
    n_benign_cpg = len(list(benign_dir.glob("*.xml"))) if benign_dir.exists() else "N/A"
    n_benign_pq = group_counts.get((0, "benign"), 0)
    lines.append(f"| 0 | benign | {n_benign_cpg:,} | {n_benign_pq:,} | subsampled |")

    if vuln_dir.exists():
        # Count .xml files per group by reading meta.json
        import json
        grp_cpg: dict[tuple[int, str], int] = defaultdict(int)
        for xml in vuln_dir.glob("*.xml"):
            meta = xml.with_suffix("").with_suffix(".meta.json")
            if not meta.exists():
                meta = xml.with_name(xml.stem + ".meta.json")
            if meta.exists():
                try:
                    m = json.loads(meta.read_text(encoding="utf-8"))
                    cwe_str = parse_cwe(m.get("cwe_id", ""))
                    gid, gname = cwe_to_group(cwe_str)
                    grp_cpg[(gid, gname)] += 1
                except Exception:
                    pass

        for (gid, gname), cpg_cnt in sorted(grp_cpg.items(), key=lambda x: -x[1]):
            if gid == -1: 
                continue

            pq_cnt = group_counts.get((gid, gname), 0)
            if pq_cnt > 0:
                pct = f"{100*cpg_cnt//pq_cnt}%"
            else:
                pct = "N/A"
            lines.append(f"| {gid} | {gname} | {cpg_cnt:,} | {pq_cnt:,} | {pct} |")

        unk_cpg = grp_cpg.get((-1, "UNKNOWN"), 0)
        unk_pq = group_counts.get((-1, "UNKNOWN"), 0)
        lines.append(f"| -1 | UNKNOWN | {unk_cpg:,} | {unk_pq:,} | filtered |")
    return "\n".join(lines)


def cross_dataset_table(all_group_counts: dict[str, dict]) -> str:
    datasets = list(all_group_counts.keys())
    header = "| Group ID | Group | " + " | ".join(datasets) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(datasets)) + "|"
    lines = [header, sep]

    all_groups: set[tuple[int, str]] = set()
    for gc in all_group_counts.values():
        all_groups.update(gc.keys())
    all_groups.discard((0, "benign"))

    for gid, gname in sorted(all_groups, key=lambda x: (-sum(
        all_group_counts[d].get((x[0], x[1]), 0) for d in datasets
    ),)):
        counts = [str(all_group_counts[d].get((gid, gname), 0)) for d in datasets]
        lines.append(f"| {gid} | {gname} | " + " | ".join(counts) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="DATASET_ANALYSIS.md")
    args = parser.parse_args()

    DATA = PROJECT_ROOT / "data" / "datasets"
    RAW = PROJECT_ROOT / "data" / "raw"

    sections = []
    all_group_counts: dict[str, dict] = {}
    
    TOP25_CWES = get_cwe_set_from_xml(DATA.parent / "cwe" / "top25.xml")
    OWASP_CWES = get_cwe_set_from_xml(DATA.parent / "cwe" / "owasptop10.xml")
    OWASP_MAP = get_owasp_category_map(DATA.parent / "cwe" / "owasptop10.xml")

    GROUP_VOCAB_LINE = " · ".join(
        f"{v}={k}" for k, v in sorted(GROUP_VOCAB.items(), key=lambda x: x[1])
    )

    sections.append(f"""# Dataset Analysis — Complete CWE and Group Distribution

Generated from raw parquet files. Group mapping via `CWE_GROUP_MAP` in `dataset_lm.py`.

**Fixed Group IDs:**
{GROUP_VOCAB_LINE} · -1=UNKNOWN (not in CWE_GROUP_MAP)

---

## Summary

| Dataset | Total | Benign | Vulnerable | Has CWE | Has Flaw Lines | Notes |
|---|---|---|---|---|---|---|
| BigVul | 217,007 | 206,112 | 10,895 | Yes | Yes (diff) | Primary training dataset |
| DiverseVul | 330,492 | 311,547 | 18,945 | Yes (multi-label) | No | Binary only |
| MegaVul | 55,868 | 27,934 | 27,934 | Yes | Yes (diff) | Balanced 1:1 |
| Devign | 27,318 | 14,858 | 12,460 | No | Yes (vul_lines) | Binary only |
| Merged (BigVul+MegaVul) | 176,674 | 154,205 | 22,469 | Yes | Yes (diff) | Combined |
| TitanVul | 77,096 | 38,548 | 38,548 | Yes | Yes (diff) | Balanced 1:1, Unfiltered |
| BenchVul | 2,100 | 1,050 | 1,050 | Yes | Yes (diff) | **Benchmark for Top 25 Most Dangerous CWEs** |
| ReposVul | 6,897 CVEs → ~115,670 funcs | ~113,952 | ~1,718 | Yes (236 CWEs) | Yes (patch diff) | Multi-granularity, untangled patches |

---

## Language Distribution (sampled from raw_func / func_before)

""")

    # ── Language distribution ─────────────────────────────────────────────────
    print("Detecting language distributions...")
    lang_dists: dict[str, Counter] = {}

    bv_lang_path = DATA / "bigvul" / "all.parquet"
    if bv_lang_path.exists():
        lang_dists["BigVul"] = language_dist_from_parquet(
            pd.read_parquet(bv_lang_path), func_col="func_before", lang_col="lang"
        )

    mv_lang_path = DATA / "megavul" / "train.parquet"
    if mv_lang_path.exists():
        lang_dists["MegaVul"] = language_dist_from_parquet(
            pd.read_parquet(mv_lang_path), func_col="func_before"
        )

    tv_lang_path = DATA / "titanvul" / "train.parquet"
    if tv_lang_path.exists():
        tv_df = pd.read_parquet(tv_lang_path)
        lang_col = "language" if "language" in tv_df.columns else None
        ext_col  = "extension" if "extension" in tv_df.columns and not lang_col else None
        lang_dists["TitanVul"] = language_dist_from_parquet(
            tv_df, func_col="func_before",
            lang_col=lang_col or ext_col,
        )

    bv_raw = RAW / "bigvul"
    if bv_raw.exists():
        lang_dists["BigVul (raw CPGs)"] = language_dist_from_raw(bv_raw)

    if lang_dists:
        sections.append(language_table(lang_dists) + "\n")
    else:
        sections.append("*Language distribution not computed — raw data not found.*\n\n---\n")

    # ── BigVul ───────────────────────────────────────────────────────────────
    print("Analyzing BigVul...")
    bv_path = DATA / "bigvul" / "all.parquet"
    if bv_path.exists():
        bv = pd.read_parquet(bv_path)
        gc, cr, nb, nv = analyze_df(bv, cwe_col="CWE ID", label_col="vul")
        all_group_counts["BigVul"] = gc
        cpg_tbl = cpg_coverage_table(RAW / "bigvul", gc)
        sections.append(f"""## 1. BigVul (`data/datasets/bigvul/all.parquet`)

Total: **{nb+nv:,}** | Benign: **{nb:,}** | Vulnerable: **{nv:,}**

### Group Distribution

{group_dist_table(gc, cr, OWASP_MAP)}

### CWE Distribution (all vulnerable)

{cwe_dist_table(cr, TOP25_CWES, OWASP_CWES)}

### Top 25 Most Dangerous CWEs

{subset_dist_table(cr, TOP25_CWES, "Top 25")}

### OWASP Top 10 (2025)

{subset_dist_table(cr, OWASP_CWES, "OWASP Top 10", group_override=OWASP_MAP)}

### CPG Files vs all.parquet (data/raw/bigvul/)

{cpg_tbl}

> All vulnerable CPG files = 100% coverage of all.parquet vulnerable.
> Benign subsampled to ~4,000.
""")
    else:
        print(f"  SKIP: {bv_path} not found")

    # ── DiverseVul ───────────────────────────────────────────────────────────
    print("Analyzing DiverseVul...")
    dv_path = DATA / "diversevul" / "all.parquet"
    if dv_path.exists():
        import numpy as np
        dv = pd.read_parquet(dv_path)
        # cwe column contains numpy arrays; extract first element as string
        def extract_dv_cwe(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return ""
            if hasattr(val, '__len__'):
                return str(val[0]) if len(val) > 0 else ""
            return str(val)
        dv["_cwe_str"] = dv["cwe"].apply(extract_dv_cwe)
        gc, cr, nb, nv = analyze_df(dv, cwe_col="_cwe_str", label_col="target")
        all_group_counts["DiverseVul"] = gc
        sections.append(f"""## 2. DiverseVul (`data/datasets/diversevul/all.parquet`)

Total: **{nb+nv:,}** | Benign: **{nb:,}** | Vulnerable: **{nv:,}**

> CWE column is multi-label array (e.g. `['CWE-787', 'CWE-119']`).
> Group assignment uses primary (first) CWE only.
> No flaw line ground truth available.

### Group Distribution

{group_dist_table(gc, cr, OWASP_MAP)}

### CWE Distribution (all vulnerable, primary CWE)

{cwe_dist_table(cr, TOP25_CWES, OWASP_CWES)}

### Top 25 Most Dangerous CWEs

{subset_dist_table(cr, TOP25_CWES, "Top 25")}

### OWASP Top 10 (2025)

{subset_dist_table(cr, OWASP_CWES, "OWASP Top 10", group_override=OWASP_MAP)}
""")
    else:
        print(f"  SKIP: {dv_path} not found")

    # ── MegaVul ──────────────────────────────────────────────────────────────
    print("Analyzing MegaVul...")
    mv_path = DATA / "megavul" / "train.parquet"
    if mv_path.exists():
        mv = pd.read_parquet(mv_path)
        gc, cr, nb, nv = analyze_df(mv, cwe_col="CWE ID", label_col="vul")
        all_group_counts["MegaVul"] = gc
        sections.append(f"""## 3. MegaVul (`data/datasets/megavul/train.parquet`)

Total: **{nb+nv:,}** | Benign: **{nb:,}** | Vulnerable: **{nv:,}**

> Perfectly balanced (1:1 ratio). Has `func_before` + `func_after` for diff-based flaw lines.

### Group Distribution

{group_dist_table(gc, cr, OWASP_MAP)}

### CWE Distribution (all vulnerable)

{cwe_dist_table(cr, TOP25_CWES, OWASP_CWES)}

### Top 25 Most Dangerous CWEs

{subset_dist_table(cr, TOP25_CWES, "Top 25")}

### OWASP Top 10 (2025)

{subset_dist_table(cr, OWASP_CWES, "OWASP Top 10", group_override=OWASP_MAP)}
""")
    else:
        print(f"  SKIP: {mv_path} not found")

    # ── TitanVul ─────────────────────────────────────────────────────────────
    print("Analyzing TitanVul...")
    tv_path = DATA / "titanvul" / "train.parquet"
    if tv_path.exists():
        tv = pd.read_parquet(tv_path)
        gc, cr, nb, nv = analyze_df(tv, cwe_col="CWE ID", label_col="vul")
        all_group_counts["TitanVul"] = gc
        sections.append(f"""## 6. TitanVul (`data/datasets/titanvul/train.parquet`)

Total: **{nb+nv:,}** | Benign: **{nb:,}** | Vulnerable: **{nv:,}**

> Aggregated from 7 public vulnerability datasets (BigVul, D2A, CVEfixes, Devign, ReVeal, DiverseVul, MegaVul),
> deduplicated and validated with a multi-agent LLM framework.
> Contains 38,548 multilingual pairs (unfiltered).
> Balanced 1:1 (func_after = benign). Has `func_before` + `func_after` for diff-based flaw lines.

### Group Distribution

{group_dist_table(gc)}

### CWE Distribution (all vulnerable)

{cwe_dist_table(cr, TOP25_CWES, OWASP_CWES)}

### Top 25 Most Dangerous CWEs

{subset_dist_table(cr, TOP25_CWES, "Top 25")}

### OWASP Top 10 (2025)

{subset_dist_table(cr, OWASP_CWES, "OWASP Top 10", group_override=OWASP_MAP)}
""")
    else:
        print(f"  SKIP: {tv_path} not found")

    # ── BenchVul ─────────────────────────────────────────────────────────────
    print("Analyzing BenchVul...")
    bvul_path = DATA / "benchvul" / "train.parquet"
    if bvul_path.exists():
        bvul = pd.read_parquet(bvul_path)
        gc, cr, nb, nv = analyze_df(bvul, cwe_col="CWE ID", label_col="vul")
        all_group_counts["BenchVul"] = gc
        sections.append(f"""## 7. BenchVul (`data/datasets/benchvul/train.parquet`)

# Benchmark for Top 25 Most Dangerous CWEs

Total: **{nb+nv:,}** | Benign: **{nb:,}** | Vulnerable: **{nv:,}**

> Manually verified benchmark designed for **evaluating** vulnerability detection models.
> Covers a refined set of the Top 25 Most Dangerous CWEs (MITRE 2024).
> 50 vulnerable + 50 fixed samples per CWE (before C/C++ filter).
> Labels achieve 92% correctness rate per manual review.
> Contains 1,050 multilingual pairs (unfiltered).
> **Intended for evaluation/testing only — not suitable for training.**

### Group Distribution

{group_dist_table(gc, cr, OWASP_MAP)}

### CWE Distribution (all vulnerable)

{cwe_dist_table(cr, TOP25_CWES, OWASP_CWES)}

### Top 25 Most Dangerous CWEs

{subset_dist_table(cr, TOP25_CWES, "Top 25")}

### OWASP Top 10 (2025)

{subset_dist_table(cr, OWASP_CWES, "OWASP Top 10", group_override=OWASP_MAP)}
""")
    else:
        print(f"  SKIP: {bvul_path} not found")

    # ── Devign ───────────────────────────────────────────────────────────────
    sections.append("""## 8. Devign (`data/datasets/devign/{train,validation,test}.parquet`)

Total: **27,318** | Benign: **14,858** | Vulnerable: **12,460**

> No CWE column. Binary only (`target` column). Cannot be used for multiclass/group mode.
> Flaw lines available via `vul_lines` column (Devign-format dict).
""")

    # ── Merged ───────────────────────────────────────────────────────────────
    print("Analyzing Merged...")
    mg_path = DATA / "merged" / "train.parquet"
    if mg_path.exists():
        mg = pd.read_parquet(mg_path)
        gc, cr, nb, nv = analyze_df(mg, cwe_col="CWE ID", label_col="vul")
        all_group_counts["Merged"] = gc
        sections.append(f"""## 9. Merged (`data/datasets/merged/train.parquet`)

Total: **{nb+nv:,}** | Benign: **{nb:,}** | Vulnerable: **{nv:,}**

> Combined BigVul + MegaVul. Has `CWE ID`, `func_before`, `func_after`.

### Group Distribution

{group_dist_table(gc, cr, OWASP_MAP)}

### CWE Distribution (all vulnerable)

{cwe_dist_table(cr, TOP25_CWES, OWASP_CWES)}

### Top 25 Most Dangerous CWEs

{subset_dist_table(cr, TOP25_CWES, "Top 25")}

### OWASP Top 10 (2025)

{subset_dist_table(cr, OWASP_CWES, "OWASP Top 10", group_override=OWASP_MAP)}
""")
    else:
        print(f"  SKIP: {mg_path} not found")

    # ── Cross-dataset table ──────────────────────────────────────────────────
    if len(all_group_counts) > 1:
        vuln_only = {}
        for ds, gc in all_group_counts.items():
            vuln_only[ds] = {k: v for k, v in gc.items() if k != (0, "benign")}
        sections.append(f"""## Cross-Dataset Group Coverage (vulnerable only)

{cross_dataset_table(vuln_only)}
""")

    out = "\n---\n\n".join(sections)
    out_path = PROJECT_ROOT / args.out
    out_path.write_text(out, encoding="utf-8")
    print(f"\nWritten -> {out_path}")


if __name__ == "__main__":
    main()
