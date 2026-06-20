"""
Inference service: raw code string -> Joern CPG -> model -> structured result.

Reuses gnn_vuln.inference.VulnPredictor (works for all archs) + Joern runner. Predictors
are loaded once per model_id and cached. The DB graph cache skips Joern on repeat code.
"""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from gnn_vuln.inference import VulnPredictor
from gnn_vuln.data.joern_runner import process_function

from API.core.config import settings
from API.services import registry, graph_cache

_predictors: dict[str, VulnPredictor] = {}
_lock = threading.Lock()


def get_predictor(model_id: str) -> VulnPredictor:
    with _lock:
        if model_id not in _predictors:
            m = registry.get_model(model_id)
            ckpt = registry.abspath(m["checkpoint"])
            cfg = registry.abspath(m["config"])
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint missing for '{model_id}': {ckpt}")
            if not cfg.exists():
                raise FileNotFoundError(f"Config missing for '{model_id}': {cfg}")
            predictor = VulnPredictor.from_checkpoint(str(ckpt), str(cfg), device=settings.DEVICE)
            names = m.get("class_names")
            if names:
                predictor.class_names = names
            _predictors[model_id] = predictor
        return _predictors[model_id]


def _predict_text(predictor: VulnPredictor, fmt: str, content: str, top_k: int | None):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / f"cpg.{fmt}"
        p.write_text(content, encoding="utf-8")
        return predictor.predict_from_file(str(p), max_nodes=settings.MAX_NODES, top_k_lines=top_k)


def infer_codes(model_id: str, codes: list[str], top_k_lines: int | None = None) -> list[dict]:
    """One result dict per input (ok flag + error). Cache-first, Joern on miss."""
    predictor = get_predictor(model_id)
    joern = Path(settings.JOERN_CLI)
    results: list[dict] = []
    for i, code in enumerate(codes):
        try:
            h = graph_cache.code_hash(code)
            cached = graph_cache.get(h)
            if cached is not None:
                fmt, content = cached
                r = _predict_text(predictor, fmt, content, top_k_lines)
                hit = True
            else:
                with tempfile.TemporaryDirectory() as td:
                    cpg = process_function(code, i, Path(td), joern_cli_dir=joern)
                    if cpg is None:
                        results.append({"index": i, "ok": False,
                                        "error": "Joern produced no CPG (parse failed or empty function)"})
                        continue
                    fmt = Path(cpg).suffix.lstrip(".")
                    content = Path(cpg).read_text(encoding="utf-8", errors="replace")
                    graph_cache.put(h, fmt, content)
                    r = predictor.predict_from_file(str(cpg), max_nodes=settings.MAX_NODES,
                                                    top_k_lines=top_k_lines)
                hit = False
            if r is None:
                results.append({"index": i, "ok": False,
                                "error": f"graph empty or exceeds max_nodes ({settings.MAX_NODES})"})
                continue
            results.append({"index": i, "ok": True, "cached": hit, **r})
        except Exception as e:  # noqa: BLE001 — surface per-function failure, keep batch going
            results.append({"index": i, "ok": False, "error": f"{type(e).__name__}: {e}"})
    return results
