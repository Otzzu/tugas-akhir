"""Thin shim — implementation moved into the gnn_vuln library.

The CPG -> .pt dataset build now lives in the package so any consumer (research
CLI, the API ingestion service, etc.) can reuse it.

Back-compat (unchanged):
    uv run python scripts/process_dataset.py --config configs/...yaml
Preferred:
    python -m gnn_vuln.data.build_pt --config configs/...yaml
"""
import sys
from pathlib import Path

# Allow running as a bare script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_vuln.data.build_pt import main

if __name__ == "__main__":
    main()
