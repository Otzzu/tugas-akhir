"""
patch_pt_inject_parquet_id.py — Inject parquet_id, cwe_id, raw_funcs into an old .pt
that was built before these fields were added.

Specifically targets:
  lm_dataset_bigvul_multiclass_unixcoder-base_ft_top10.pt
  (3-tuple, no parquet_id/cwe_id/raw_funcs, top_cwe=10, benign pre-benign_extra)

Strategy
--------
Reconstruct the EXACT processing order used by dataset_lm.py:
  1. Benign files (func_N.xml, N < 20000) sorted alphabetically → label 0
  2. Vulnerable files sorted alphabetically, grouped by CWE class_id (1..top_cwe)
     processed in class_id order

For each file position, check XML validity (count nodes > 0, within max_nodes).
Cross-check cumulative counts against old .pt y distribution.

For benign:  parquet_id = meta["id"], raw_func = meta["raw_func"]
For vulnerable: parquet_id = -1,        raw_func = METHOD CODE from XML

Usage
-----
    python scripts/patch_pt_inject_parquet_id.py \\
        --pt data/processed/lm_dataset_bigvul_multiclass_unixcoder-base_ft_top10.pt \\
        --raw-dir data/raw/bigvul \\
        --top-cwe 10 \\
        --max-nodes 1000

    # Then patch func tokens for alternative LMs:
    python scripts/patch_pt_add_func.py \\
        --pt data/processed/lm_dataset_bigvul_multiclass_unixcoder-base_ft_top10_injected.pt \\
        --func-lm Salesforce/codet5p-110m-embedding --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from tqdm import tqdm

_NS = "http://graphml.graphdrawing.org/xmlns"


# ---------------------------------------------------------------------------
# XML helpers (no embedder — lightweight only)
# ---------------------------------------------------------------------------

def _count_nodes(xml_path: Path, max_nodes: int) -> int:
    """Return number of nodes in CPG (capped at max_nodes). 0 = invalid/empty."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {"g": _NS}
        graph = root.find("g:graph", ns)
        if graph is None:
            return 0
        nodes = graph.findall("g:node", ns)
        n = len(nodes)
        if n == 0 or n > max_nodes:
            return 0
        return n
    except Exception:
        return 0


def _extract_method_code(xml_path: Path) -> str:
    """
    Extract METHOD node CODE attribute from CPG XML.
    Falls back to concatenating first 50 node code snippets.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {"g": _NS}

        # Build key_id → attr_name map
        key_map: dict[str, str] = {}
        for key in root.findall("g:key", ns):
            kid = key.get("id", "")
            aname = key.get("attr.name", "")
            if aname:
                key_map[kid] = aname

        graph = root.find("g:graph", ns)
        if graph is None:
            return ""

        # Find METHOD node CODE
        for node in graph.findall("g:node", ns):
            label = ""
            code = ""
            for data in node.findall("g:data", ns):
                k = key_map.get(data.get("key", ""), "")
                v = (data.text or "").strip()
                if k == "labelV":
                    label = v
                elif k == "CODE":
                    code = v
            if label == "METHOD" and code:
                return code

        # Fallback: collect all CODE snippets
        snippets = []
        for node in graph.findall("g:node", ns):
            for data in node.findall("g:data", ns):
                k = key_map.get(data.get("key", ""), "")
                v = (data.text or "").strip()
                if k == "CODE" and v:
                    snippets.append(v)
        return "\n".join(snippets[:50])
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Build ordered list of (xml_path, parquet_id, raw_func, cwe_id)
# ---------------------------------------------------------------------------

def _build_order(
    raw_dir: Path,
    cwe_vocab: dict[str, int],
    top_cwe: int,
    max_nodes: int,
    expected_y: list[int],
) -> list[tuple[Path, int, str, int]]:
    """
    Replay dataset_lm.py processing order and return list of
    (xml_path, parquet_id, raw_func, cwe_id) for each graph in the old .pt.
    Validates against expected_y.
    """
    benign_dir = raw_dir / "benign"
    vuln_dir = raw_dir / "vulnerable"

    # ── Benign ──────────────────────────────────────────────────────────────
    all_benign = sorted(
        f for f in benign_dir.iterdir()
        if f.suffix == ".xml" and ".meta." not in f.name
    )
    orig_benign = [f for f in all_benign if int(f.stem.replace("func_", "")) < 20000]
    print(f"Benign orig files (N<20000): {len(orig_benign)}")

    benign_entries: list[tuple[Path, int, str, int]] = []
    for f in tqdm(orig_benign, desc="Scanning benign", unit="file"):
        n = _count_nodes(f, max_nodes)
        if n == 0:
            continue
        meta_f = f.with_suffix("").with_suffix(".meta.json")
        parquet_id = -1
        raw_func = ""
        if meta_f.exists():
            try:
                m = json.loads(meta_f.read_text(encoding="utf-8"))
                parquet_id = int(m.get("id", -1))
                raw_func = m.get("raw_func", "") or ""
            except Exception:
                pass
        if not raw_func:
            raw_func = _extract_method_code(f)
        benign_entries.append((f, parquet_id, raw_func, 0))  # cwe_id=0 for benign

    print(f"  Valid benign graphs: {len(benign_entries)}")

    # ── Vulnerable (grouped by class_id, sorted within each group) ──────────
    filtered_vocab = {k: v for k, v in cwe_vocab.items() if 0 < v <= top_cwe}

    all_vuln = sorted(
        f for f in vuln_dir.iterdir()
        if f.suffix == ".xml" and ".meta." not in f.name
    )

    # Group files by cwe class_id
    cwe_class_groups: dict[int, list[Path]] = {}
    skipped_cwe = 0
    for f in tqdm(all_vuln, desc="Reading vuln meta", unit="file"):
        meta_f = f.with_suffix("").with_suffix(".meta.json")
        if not meta_f.exists():
            skipped_cwe += 1
            continue
        try:
            m = json.loads(meta_f.read_text(encoding="utf-8"))
            cwe_str = m.get("cwe", "")
            class_id = filtered_vocab.get(cwe_str, -1)
            if class_id < 0:
                continue
            cwe_class_groups.setdefault(class_id, []).append(f)
        except Exception:
            skipped_cwe += 1

    if skipped_cwe:
        print(f"  Skipped {skipped_cwe} vuln files (no/bad meta)")

    vuln_entries: list[tuple[Path, int, str, int]] = []
    for class_id, files in sorted(cwe_class_groups.items()):
        cwe_str = next(k for k, v in filtered_vocab.items() if v == class_id)
        count_before = len(vuln_entries)
        for f in files:  # already sorted (sorted() above)
            n = _count_nodes(f, max_nodes)
            if n == 0:
                continue
            parquet_id = -1
            raw_func = ""
            meta_f = f.with_suffix("").with_suffix(".meta.json")
            if meta_f.exists():
                try:
                    m = json.loads(meta_f.read_text(encoding="utf-8"))
                    parquet_id = int(m.get("id", -1))
                    raw_func = m.get("raw_func", "") or ""
                except Exception:
                    pass
            if not raw_func:
                raw_func = _extract_method_code(f)
            vuln_entries.append((f, parquet_id, raw_func, class_id))
        count_after = len(vuln_entries)
        print(f"  class_id={class_id} ({cwe_str}): {count_after - count_before} graphs")

    # ── Combine and validate ────────────────────────────────────────────────
    order = benign_entries + vuln_entries

    if len(order) != len(expected_y):
        print(
            f"\nWARNING: reconstructed {len(order)} graphs but .pt has {len(expected_y)}.",
            file=sys.stderr,
        )
        print("Order reconstruction may be wrong — check max_nodes and top_cwe.", file=sys.stderr)
    else:
        mismatches = sum(
            1 for (_, _, _, cwe_id), y in zip(order, expected_y)
            if cwe_id != y
        )
        if mismatches:
            print(
                f"\nWARNING: {mismatches}/{len(order)} cwe_id != y mismatches!",
                file=sys.stderr,
            )
            print("Processing order may differ from original build.", file=sys.stderr)
        else:
            print(f"\nOrder validated: all {len(order)} cwe_id match y ✓")

    return order


# ---------------------------------------------------------------------------
# Main patch
# ---------------------------------------------------------------------------

def patch(pt_path: Path, raw_dir: Path, out_path: Path, top_cwe: int, max_nodes: int) -> None:
    print(f"Loading {pt_path}")
    result = torch.load(pt_path, weights_only=False)

    if len(result) == 4:
        data, slices, class_names, existing_raw_funcs = result
        if existing_raw_funcs:
            print("raw_funcs already present — use --force to overwrite.")
            return
    elif len(result) == 3:
        data, slices, class_names = result
    else:
        print(f"ERROR: unexpected tuple length {len(result)}", file=sys.stderr)
        sys.exit(1)

    # Extract y per graph
    y_slices = slices["y"]  # [N+1]
    n_graphs = len(y_slices) - 1
    y_all: list[int] = []
    for i in range(n_graphs):
        s, e = int(y_slices[i]), int(y_slices[i + 1])
        y_all.append(int(data.y[s:e][0].item()))

    print(f"Total graphs in .pt: {n_graphs}")

    # Load cwe_vocab
    vocab_path = raw_dir / "cwe_vocab.json"
    if not vocab_path.exists():
        print(f"ERROR: {vocab_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(vocab_path) as f:
        cwe_vocab: dict[str, int] = json.load(f)

    # Build order
    order = _build_order(raw_dir, cwe_vocab, top_cwe, max_nodes, y_all)

    if len(order) != n_graphs:
        print(
            f"\nERROR: graph count mismatch ({len(order)} vs {n_graphs}). Cannot patch.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build raw_funcs list and parquet_id / cwe_id tensors to inject
    raw_funcs: list[str] = []
    parquet_ids: list[int] = []
    cwe_ids: list[int] = []

    for _, pid, rf, cid in order:
        raw_funcs.append(rf)
        parquet_ids.append(pid)
        cwe_ids.append(cid)

    # Inject parquet_id and cwe_id into collated data
    pid_tensor = torch.tensor(parquet_ids, dtype=torch.long)
    cid_tensor = torch.tensor(cwe_ids, dtype=torch.long)

    data.parquet_id = pid_tensor
    data.cwe_id = cid_tensor

    pid_slice = torch.arange(n_graphs + 1, dtype=torch.long)
    slices["parquet_id"] = pid_slice
    slices["cwe_id"] = pid_slice.clone()

    n_missing_raw = sum(1 for rf in raw_funcs if not rf)
    n_neg_pid = sum(1 for pid in parquet_ids if pid < 0)
    print(f"\nraw_funcs empty: {n_missing_raw}/{n_graphs}")
    print(f"parquet_id = -1: {n_neg_pid}/{n_graphs} (vulnerable — expected)")

    print(f"\nSaving → {out_path}")
    torch.save((data, slices, class_names, raw_funcs), out_path)
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject parquet_id, cwe_id, raw_funcs into old .pt lacking these fields.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pt", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--top-cwe", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pt_path = args.pt.resolve()
    if not pt_path.exists():
        print(f"ERROR: not found: {pt_path}", file=sys.stderr)
        sys.exit(1)

    raw_dir = args.raw_dir.resolve()

    out_path = args.out.resolve() if args.out else pt_path.parent / f"{pt_path.stem}_injected.pt"
    print(f"Output: {out_path}")

    if args.force and out_path.exists():
        out_path.unlink()

    patch(pt_path, raw_dir, out_path, args.top_cwe, args.max_nodes)


if __name__ == "__main__":
    main()
