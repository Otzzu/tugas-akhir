"""
losver_patch_compat.py — make LOSVER's run_*.py import-compatible with the project's
modern transformers (>=4.48), so it runs in the pod env without downgrading (transformers
4.19 + modern torch breaks on the removed `torch._six`).

Removed-from-transformers symbols LOSVER imports:
  - AdamW         -> use torch.optim.AdamW
  - WEIGHTS_NAME  -> "pytorch_model.bin" (or import if still available)

Edits ONLY the `from transformers import ( ... )` block (drops AdamW/WEIGHTS_NAME) and
prepends shims. Usage calls `AdamW(...)` / `WEIGHTS_NAME` then resolve to the shims.
Idempotent (re-running is a no-op once shims are present).

Run (pod, from project root):
    python scripts/losver_patch_compat.py --files src/losver/classification/run_line_CWE.py ...
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

SHIM = (
    "# --- losver_patch_compat: transformers>=4.48 shims ---\n"
    "from torch.optim import AdamW  # transformers removed AdamW\n"
    "try:\n"
    "    from transformers import WEIGHTS_NAME\n"
    "except Exception:\n"
    "    WEIGHTS_NAME = 'pytorch_model.bin'\n"
    "# --- end shim ---\n"
)
MARKER = "losver_patch_compat"


def _strip_in_import_block(block: str) -> str:
    for sym in ("AdamW", "WEIGHTS_NAME"):
        block = re.sub(rf"\b{sym}\s*,\s*", "", block)   # "Sym, "
        block = re.sub(rf",\s*{sym}\b", "", block)        # ", Sym"
    return block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    a = ap.parse_args()

    for fp in a.files:
        p = Path(fp)
        if not p.exists():
            print(f"  skip (missing): {fp}")
            continue
        src = p.read_text(encoding="utf-8")
        if MARKER in src:
            print(f"  already patched: {fp}")
            continue
        # edit only the parenthesized `from transformers import ( ... )` block
        src2 = re.sub(
            r"from transformers import \([^)]*\)",
            lambda m: _strip_in_import_block(m.group(0)),
            src,
            flags=re.DOTALL,
        )
        # single-fold: LOSVER runs 5-fold CV (main("1")..main("5")); we have ONE split,
        # so drop folds 2-5 (run only fold 1). No-op if the file has no such calls.
        src2 = re.sub(r'(?m)^[ \t]*main\(\s*"[2-5]"\s*\)[ \t]*\n', "", src2)
        # insert shim AFTER any `from __future__` imports (which must stay first)
        lines = src2.split("\n")
        fut = [i for i, l in enumerate(lines) if l.lstrip().startswith("from __future__")]
        pos = (max(fut) + 1) if fut else 0
        lines.insert(pos, SHIM.rstrip("\n"))
        p.write_text("\n".join(lines), encoding="utf-8")
        print(f"  patched: {fp}")


if __name__ == "__main__":
    main()
