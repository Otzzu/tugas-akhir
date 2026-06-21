"""Thin shim — implementation moved into the gnn_vuln library.

The dataset CPG-generation pipeline now lives in the package so it can be reused
by any consumer (research CLI, the API ingestion service, etc.) without depending
on this scripts/ directory.

Back-compat (unchanged):
    uv run python scripts/prepare_dataset.py --input ... --format ... --out-dir ...
Preferred:
    python -m gnn_vuln.data.prepare --input ... --format ... --out-dir ...
"""
import sys
from pathlib import Path

# Allow running as a bare script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_vuln.data.prepare import main

if __name__ == "__main__":
    main()
