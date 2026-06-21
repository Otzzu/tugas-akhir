"""Export the FastAPI OpenAPI schema to openapi.json for offline review.

View it with the live docs at http://localhost:8000/docs, or open openapi.json with any
Swagger/OpenAPI viewer (e.g. the VS Code OpenAPI extension, or https://editor.swagger.io).

Run (from the API service venv):
    cd API && uv run python export_openapi.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Put the repo root on the path so the `API` package resolves when this file is run
# as a script. gnn_vuln is NOT needed here — app import is lazy (no torch on import).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from API.main import app

HERE = Path(__file__).resolve().parent
spec = app.openapi()
(HERE / "openapi.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
print(f"wrote {HERE / 'openapi.json'}")
print(f"paths: {len(spec.get('paths', {}))}")
