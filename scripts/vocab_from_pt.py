"""Regenerate cwe_vocab.json from a built dataset — class_names in the .pt is the same
information, ordered by id. No raw data, no rebuild.

    uv run python scripts/vocab_from_pt.py \
        --pt data/processed/<name>_meta.pt --out data/raw/megavul/cwe_vocab.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=Path, required=True, help="<name>_meta.pt (lazy) or <name>.pt")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    meta = torch.load(args.pt, map_location="cpu", weights_only=False)
    names = meta.get("class_names")
    if not names:
        raise SystemExit(f"{args.pt} has no class_names (binary dataset?)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({c: i for i, c in enumerate(names)}, indent=2), encoding="utf-8")
    print(f"{len(names)} classes -> {args.out}")


if __name__ == "__main__":
    main()
