"""Patch flaw_line_mask in already-built datasets: drop the METHOD-node flag.

Background
----------
The old flaw_line_mask marked any node whose [lineNumber, lineNumberEnd] range
contained a patch line. In MegaVul CPGs only the METHOD node has a lineNumberEnd
(spanning the whole function), so it marked line 1 (the signature) as a flaw in
~90% of vulnerable functions and inflated localization Top-1. Every other node
lacks lineNumberEnd, so its range already collapses to an exact line match.

Zeroing the METHOD node's flag therefore turns the old mask into the correct
exact-line mask (a node is a flaw iff its own source line is a patch line),
without a full rebuild that would re-embed every node with UniXcoder. This is
identical to a from-scratch rebuild with the corrected builder (verified).

Handles both storage formats:
  - lazy   : a directory of per-graph .pt files (``*_graphs/``)
  - inmemory: a monolithic .pt holding (data, slices) or a single Data

Usage
-----
    uv run python scripts/patch_flaw_mask.py --data <dir_or_pt> --dry-run
    uv run python scripts/patch_flaw_mask.py --data data/processed/<ds>_graphs
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnn_vuln.data.node_embedder import NODE_TYPE_TO_IDX  # noqa: E402
from gnn_vuln.data.cpg.features import _NUM_NODE_TYPES  # noqa: E402

METHOD_IDX = NODE_TYPE_TO_IDX["METHOD"]


def _method_mask(x: torch.Tensor) -> torch.Tensor:
    """Boolean node mask selecting METHOD nodes from the type one-hot block."""
    th = x[:, :_NUM_NODE_TYPES]
    return (th.argmax(dim=1) == METHOD_IDX) & (th.sum(dim=1) > 0)


def _patch_data(data) -> int:
    """Zero flaw_line_mask at METHOD nodes. Returns count of flags removed."""
    if not hasattr(data, "flaw_line_mask") or data.flaw_line_mask is None:
        return 0
    if not hasattr(data, "x") or data.x is None:
        return 0
    meth = _method_mask(data.x)
    removed = int((data.flaw_line_mask.bool() & meth).sum())
    if removed:
        fm = data.flaw_line_mask.clone()
        fm[meth] = 0
        data.flaw_line_mask = fm
    return removed


def patch_lazy_dir(dir_path: Path, dry_run: bool) -> None:
    files = sorted(
        f for f in glob.glob(os.path.join(dir_path, "*.pt"))
        if not os.path.basename(f).startswith("_")
    )
    if not files:
        raise SystemExit(f"No .pt graph files in {dir_path}")
    total = len(files)
    changed = removed_total = had = has_after = became_empty = 0
    for fp in files:
        g = torch.load(fp, weights_only=False)
        before = int(g.flaw_line_mask.sum()) if getattr(g, "flaw_line_mask", None) is not None else 0
        removed = _patch_data(g)
        after = int(g.flaw_line_mask.sum()) if getattr(g, "flaw_line_mask", None) is not None else 0
        if before > 0:
            had += 1
            has_after += 1 if after > 0 else 0
            became_empty += 1 if after == 0 else 0
        if removed:
            removed_total += removed
            changed += 1
            if not dry_run:
                torch.save(g, fp)
    print(f"[lazy] {dir_path}")
    print(f"  graphs total            : {total}")
    print(f"  graphs changed          : {changed}")
    print(f"  METHOD flags removed    : {removed_total}")
    print(f"  had flaw (before)       : {had}")
    print(f"  have flaw (after)       : {has_after}")
    print(f"  empty after (no node)   : {became_empty}")
    print("  DRY RUN — nothing written" if dry_run else "  written in place")


def patch_inmemory_pt(pt_path: Path, dry_run: bool) -> None:
    obj = torch.load(pt_path, weights_only=False)
    if isinstance(obj, tuple) and len(obj) == 2:
        data, slices = obj
        removed = _patch_data(data)
        out = (data, slices)
    else:
        data = obj
        removed = _patch_data(data)
        out = data
    print(f"[inmemory] {pt_path}")
    print(f"  METHOD flags removed    : {removed}")
    if removed and not dry_run:
        torch.save(out, pt_path)
        print("  written in place")
    else:
        print("  DRY RUN — nothing written" if dry_run else "  no change")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="lazy _graphs dir OR monolithic .pt")
    ap.add_argument("--dry-run", action="store_true", help="report only, do not write")
    args = ap.parse_args()

    p = Path(args.data)
    if not p.exists():
        raise SystemExit(f"Not found: {p}")
    print(f"METHOD type index = {METHOD_IDX}, num node types = {_NUM_NODE_TYPES}")
    if p.is_dir():
        patch_lazy_dir(p, args.dry_run)
    else:
        patch_inmemory_pt(p, args.dry_run)


if __name__ == "__main__":
    main()
