"""
convert_raw_to_hdf5.py
~~~~~~~~~~~~~~~~~~~~~~
Convert data/raw/<dataset>/ graph files (JSON/XML) → compact HDF5.

Usage:
    uv run python scripts/convert_raw_to_hdf5.py --dataset megavul
    uv run python scripts/convert_raw_to_hdf5.py --dataset bigvul --max-nodes 3600 --workers 8
    uv run python scripts/convert_raw_to_hdf5.py --dataset titanvul --max-nodes 3400 --workers 8

HDF5 layout:
    <split>/func_<idx>/
        node_type  int16[N]   — NODE_TYPE_TO_IDX index (fast lookup)
        node_line  int32[N]   — line number (-1 if unknown)
        node_code  str[N]     — raw code string (variable-length)
        nodes_json str        — JSON of pruned node dicts (all fields needed by features.py)
        edges_json str        — JSON of edge dicts {src, dst, label}
        edge_src   int32[E]   — source node (0-indexed within graph)
        edge_dst   int32[E]   — dest node
        edge_type  int16[E]   — EDGE_TYPE_TO_IDX index
        attrs: label, cwe, raw_func, flaw_lines (JSON), language, row_id
    attrs (root): cwe_vocab (JSON string from cwe_vocab.json if present)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import h5py
except ImportError:
    raise SystemExit("h5py not installed. Run: uv add h5py")

from gnn_vuln.data.cpg.parser import parse_cpg
from gnn_vuln.data.cpg.constants import EDGE_TYPE_TO_IDX
from gnn_vuln.data.node_embedder import NODE_TYPE_TO_IDX

UNKNOWN_NODE = NODE_TYPE_TO_IDX["UNKNOWN"]
UNKNOWN_EDGE = len(EDGE_TYPE_TO_IDX)

STR_DT = h5py.special_dtype(vlen=str)

# Node fields needed by features.py / build_from_parsed.
# Both camelCase (MegaVul JSON) and UPPER_CASE (Joern GraphML) variants kept —
# GraphML parser only normalises labelV/code/lineNumber; rest stay as raw attr.name.
_NODE_KEEP = frozenset({
    "id", "labelV", "code",
    "lineNumber",            "LINE_NUMBER_END",   "lineNumberEnd",
    "isExternal",            "IS_EXTERNAL",
    "typeFullName",          "TYPE_FULL_NAME",
    "controlStructureType",  "CONTROL_STRUCTURE_TYPE",
    "evaluationStrategy",    "EVALUATION_STRATEGY",
    "argumentIndex",         "ARGUMENT_INDEX",
    "dispatchType",          "DISPATCH_TYPE",
    "methodFullName",        "METHOD_FULL_NAME",
    "isVariadic",            "IS_VARIADIC",
})


def _prune_node(n: dict) -> dict:
    return {k: v for k, v in n.items() if k in _NODE_KEEP}


def _encode_graph(cpg: dict) -> tuple:
    """Extract arrays and JSON strings from parsed CPG dict."""
    nodes = cpg["nodes"]
    id_map: dict[str, int] = {str(n.get("id", i)): i for i, n in enumerate(nodes)}

    node_type = np.array(
        [NODE_TYPE_TO_IDX.get(str(n.get("labelV", "")), UNKNOWN_NODE) for n in nodes],
        dtype=np.int16,
    )
    node_line = np.array(
        [int(n.get("lineNumber", -1) or -1) for n in nodes],
        dtype=np.int32,
    )
    node_code = np.array(
        [str(n.get("code", "")) for n in nodes],
        dtype=object,
    )

    edges = cpg.get("edges", [])
    src_list, dst_list, et_list, pruned_edges = [], [], [], []
    for e in edges:
        s = id_map.get(str(e.get("src", "")), -1)
        d = id_map.get(str(e.get("dst", "")), -1)
        if s < 0 or d < 0:
            continue
        et = EDGE_TYPE_TO_IDX.get(str(e.get("label", "")), UNKNOWN_EDGE)
        src_list.append(s); dst_list.append(d); et_list.append(et)
        pruned_edges.append({"src": e.get("src"), "dst": e.get("dst"), "label": e.get("label", "")})

    edge_src  = np.array(src_list, dtype=np.int32)
    edge_dst  = np.array(dst_list, dtype=np.int32)
    edge_type = np.array(et_list,  dtype=np.int16)

    nodes_json = json.dumps([_prune_node(n) for n in nodes], separators=(",", ":"))
    edges_json = json.dumps(pruned_edges, separators=(",", ":"))

    return node_type, node_line, node_code, edge_src, edge_dst, edge_type, nodes_json, edges_json


# ---------------------------------------------------------------------------
# Worker (runs in subprocess — parse + encode only, no HDF5 writes)
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> Optional[tuple]:
    """Parse one graph file and return encoded arrays + metadata, or None if skipped."""
    gf_str, max_nodes = args
    gf = Path(gf_str)
    meta_f = gf.parent / (gf.stem + ".meta.json")
    meta: dict = {}
    if meta_f.exists():
        try:
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        cpg = parse_cpg(gf, max_nodes=max_nodes)
    except Exception as exc:
        return None, str(exc)  # corrupted/malformed — signal error
    if cpg is None:
        return None, None

    encoded = _encode_graph(cpg)
    attrs = {
        "label":      int(meta.get("class_id", 0)),
        "cwe":        str(meta.get("cwe", "")),
        "raw_func":   str(meta.get("raw_func", ""))[:4096],
        "flaw_lines": json.dumps(meta.get("flaw_lines") or []),
        "language":   str(meta.get("language", "")),
        "row_id":     int(meta.get("id", -1)),
    }
    return gf.stem, encoded, attrs  # success: (key, encoded, attrs)


def _write_group(fg, encoded: tuple, attrs: dict) -> None:
    """Write encoded arrays + attrs into an open h5py group."""
    node_type, node_line, node_code, edge_src, edge_dst, edge_type, nodes_json, edges_json = encoded
    fg.create_dataset("node_type",  data=node_type, compression="gzip")
    fg.create_dataset("node_line",  data=node_line, compression="gzip")
    fg.create_dataset("node_code",  data=node_code.astype(STR_DT))
    fg.create_dataset("nodes_json", data=nodes_json)
    fg.create_dataset("edges_json", data=edges_json)
    if len(edge_src):
        fg.create_dataset("edge_src",  data=edge_src,  compression="gzip")
        fg.create_dataset("edge_dst",  data=edge_dst,  compression="gzip")
        fg.create_dataset("edge_type", data=edge_type, compression="gzip")
    else:
        fg.create_dataset("edge_src",  data=np.array([], dtype=np.int32))
        fg.create_dataset("edge_dst",  data=np.array([], dtype=np.int32))
        fg.create_dataset("edge_type", data=np.array([], dtype=np.int16))
    for k, v in attrs.items():
        fg.attrs[k] = v


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(dataset: str, max_nodes: int, out_path: Path, workers: int) -> None:
    raw_dir = Path("data/raw") / dataset
    if not raw_dir.exists():
        raise SystemExit(f"Raw dir not found: {raw_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = skipped = errors = 0

    with h5py.File(out_path, "w") as hf:
        vocab_path = raw_dir / "cwe_vocab.json"
        if vocab_path.exists():
            hf.attrs["cwe_vocab"] = vocab_path.read_text(encoding="utf-8")

        for split in ("benign", "vulnerable"):
            split_dir = raw_dir / split
            if not split_dir.exists():
                continue

            graph_files = sorted(
                str(f) for f in split_dir.iterdir()
                if f.suffix in (".json", ".xml", ".graphml")
                and ".meta." not in f.name
            )
            print(f"[{split}] {len(graph_files)} files — {workers} workers")
            grp = hf.require_group(split)

            work = [(gf, max_nodes) for gf in graph_files]

            def _handle(result, i):
                nonlocal total, skipped, errors
                key = result[0]
                if key is None:
                    # (None, None) = too large/empty; (None, str) = parse error
                    if result[1] is not None:
                        errors += 1
                    else:
                        skipped += 1
                else:
                    _, encoded, attrs = result
                    _write_group(grp.create_group(key), encoded, attrs)
                    total += 1
                if (i + 1) % 1000 == 0:
                    print(f"  {i+1}/{len(work)}  written={total} skipped={skipped} errors={errors}", flush=True)

            if workers > 1:
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=workers) as pool:
                    for i, result in enumerate(pool.imap_unordered(_worker, work, chunksize=32)):
                        _handle(result, i)
            else:
                for i, args in enumerate(work):
                    _handle(_worker(args), i)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nDone: {total} written, {skipped} skipped (>{max_nodes} nodes), {errors} parse errors")
    print(f"Output: {out_path}  ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw CPG files to HDF5.")
    parser.add_argument("--dataset",   required=True, help="Dataset name (megavul/bigvul/titanvul)")
    parser.add_argument("--max-nodes", type=int, default=5000, help="Skip graphs larger than this")
    parser.add_argument("--out",       default=None, help="Output .hdf5 path (default: data/graphs/<dataset>.hdf5)")
    parser.add_argument("--workers",   type=int, default=max(1, os.cpu_count() - 1),
                        help="Parallel parse workers (default: nCPU-1)")
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path("data/graphs") / f"{args.dataset}.hdf5"
    print(f"Workers: {args.workers} (CPUs: {os.cpu_count()})")
    convert(args.dataset, args.max_nodes, out, args.workers)


if __name__ == "__main__":
    main()
