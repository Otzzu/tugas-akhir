"""
Inference service: raw code string -> Joern CPG -> model -> structured result.

Reuses gnn_vuln.inference.VulnPredictor (works for all archs) + Joern runner. Predictors
are loaded once per model_id and cached. The DB graph cache skips Joern on repeat code.
"""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from API.core.config import settings
from API.core.database import SessionLocal
from API.models.tables import InferenceResult
from API.services import registry, graph_cache


def _store_result(model_id: str, code_h: str, r: dict) -> None:
    """Persist a prediction (history/audit). code_hash links to the cached graph.
    Stores the pre-head function embedding (drift detection / similarity search)."""
    emb = r.get("cls_embedding")
    with SessionLocal() as s:
        s.add(InferenceResult(
            model_id=model_id, code_hash=code_h,
            prediction=r.get("prediction"), class_id=r.get("class_id"),
            is_vulnerable=r.get("is_vulnerable"), confidence=r.get("confidence"),
            suspicious_lines=r.get("suspicious_lines", []),
            cls_embedding=emb,
            cls_embedding_dim=(len(emb) if emb is not None else None),
        ))
        s.commit()

_predictors: dict[str, VulnPredictor] = {}
_lock = threading.Lock()


def _materialize_checkpoint(m: dict) -> Path:
    """Ensure the model checkpoint is on local disk. Object storage (checkpoints bucket) is
    the source of truth; the local path is a cache — pulled on demand if absent (so a fresh
    worker on another node can load a model it never trained). Mirrors dataset materialize."""
    ckpt = registry.abspath(m["checkpoint"])
    if ckpt.exists():
        return ckpt
    uri = m.get("storage_uri") or ""
    if uri.startswith("s3://"):
        bucket, _, key = uri[5:].partition("/")
    else:
        bucket, key = settings.S3_BUCKET_CHECKPOINTS, f"{m['id']}.pt"
    from API.services import storage
    blob = storage.get_bytes(bucket, key)
    if blob is None:
        raise FileNotFoundError(
            f"Checkpoint for '{m['id']}' missing locally ({ckpt}) and not in object "
            f"storage ({bucket}/{key}).")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(blob)
    return ckpt


def get_predictor(model_id: str) -> VulnPredictor:
    from gnn_vuln.inference import VulnPredictor
    with _lock:
        if model_id not in _predictors:
            m = registry.get_model(model_id)
            ckpt = _materialize_checkpoint(m)         # local cache; pulls from object storage if absent
            cfg = registry.materialize_config(m)      # from DB snapshot (no repo files needed)
            predictor = VulnPredictor.from_checkpoint(str(ckpt), str(cfg), device=settings.DEVICE)
            names = m.get("class_names")
            if names:
                predictor.class_names = names
            _predictors[model_id] = predictor
        return _predictors[model_id]


def list_results(model_id: str | None = None, limit: int = 50) -> list[dict]:
    """Recent persisted predictions (newest first), optionally filtered by model."""
    from sqlalchemy import select
    with SessionLocal() as s:
        q = select(InferenceResult).order_by(InferenceResult.id.desc()).limit(limit)
        if model_id:
            q = select(InferenceResult).where(InferenceResult.model_id == model_id)\
                .order_by(InferenceResult.id.desc()).limit(limit)
        return [r.to_dict() for r in s.scalars(q).all()]


def _attach_source_lines(code: str, result: dict) -> None:
    """Attach per-line source: `code` = the single line, `statement` = the full
    statement (spans continuation lines of a multi-line statement). Line numbers
    are 1-indexed. The statement extends until a line ends in ; { } or : (capped)."""
    lines = code.splitlines()
    for sl in result.get("suspicious_lines", []):
        ln = sl.get("line")
        if isinstance(ln, int) and 1 <= ln <= len(lines):
            sl["code"] = lines[ln - 1]
            parts = [lines[ln - 1]]
            i = ln
            while i < len(lines) and not parts[-1].rstrip().endswith((";", "{", "}", ":")):
                parts.append(lines[i])
                i += 1
                if i - ln >= 15:
                    break
            sl["statement"] = "\n".join(parts).strip()


def _predict_text(predictor: VulnPredictor, fmt: str, content: str, top_k: int | None):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / f"cpg.{fmt}"
        p.write_text(content, encoding="utf-8")
        return predictor.predict_from_file(str(p), max_nodes=settings.MAX_NODES, top_k_lines=top_k)


def embed_codes(model_id: str, codes: list[str]) -> list[dict]:
    """Final pre-head function embedding per input — no classification output.
    For similarity search / drift detection. Reuses the inference pipeline (the embedding
    is the vector captured just before the output head) and persists it like a prediction."""
    out = infer_codes(model_id, codes, top_k_lines=None, keep_embedding=True)
    return [{"index": r.get("index"), "ok": r.get("ok"),
             "cls_embedding": r.get("cls_embedding"),
             "cls_embedding_dim": r.get("cls_embedding_dim"),
             "error": r.get("error")} for r in out]


def infer_codes(model_id: str, codes: list[str], top_k_lines: int | None = None,
                keep_embedding: bool = False) -> list[dict]:
    """One result dict per input (ok flag + error). Cache-first, Joern on miss.
    keep_embedding=True returns the pre-head embedding in the result (used by /embed);
    the default pops it (kept out of the /inference HTTP response, only its dim returned)."""
    from gnn_vuln.data.joern_runner import process_function
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
            _attach_source_lines(code, r)
            _store_result(model_id, h, r)
            # The embedding is persisted (drift / search). /inference returns only its
            # dim; /embed (keep_embedding=True) returns the vector itself.
            emb = r.get("cls_embedding")
            r["cls_embedding_dim"] = len(emb) if emb is not None else None
            if not keep_embedding:
                r.pop("cls_embedding", None)
            results.append({"index": i, "ok": True, "cached": hit, **r})
        except Exception as e:  # noqa: BLE001 — surface per-function failure, keep batch going
            results.append({"index": i, "ok": False, "error": f"{type(e).__name__}: {e}"})
    return results
