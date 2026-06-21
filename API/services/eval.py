"""Model evaluation over stored inference history.

`evaluate()` is the GENERAL, label-free evaluation of a deployed model from what the API
already records per inference (prediction, confidence, pre-head embedding): how much it's
been used, its prediction mix, its confidence, and — as one nested signal — drift.

Extensible: more eval facets (e.g. supervised metrics when labels are supplied, calibration,
per-class coverage) can be added to the report without changing the endpoint.

Drift signal (label-free covariate/prior shift, recent window vs the prior window):
  - **PSI** (Population Stability Index) on the predicted-class distribution — the standard
    monitoring metric (<0.1 stable, 0.1-0.25 moderate, >0.25 significant).
  - **confidence drop** — mean softmax confidence recent vs reference.
  - **embedding centroid shift** — cosine distance between the mean pre-head embedding.
"""
from __future__ import annotations

import math

from sqlalchemy import select

from API.core.database import SessionLocal
from API.models.tables import InferenceResult

_PSI_SIGNIFICANT = 0.25
_CONF_DROP = 0.10


def _psi(ref_counts: dict[int, int], cur_counts: dict[int, int]) -> float:
    classes = set(ref_counts) | set(cur_counts)
    r_tot = sum(ref_counts.values()) or 1
    c_tot = sum(cur_counts.values()) or 1
    psi = 0.0
    for c in classes:
        r = (ref_counts.get(c, 0) + 1) / (r_tot + len(classes))
        u = (cur_counts.get(c, 0) + 1) / (c_tot + len(classes))
        psi += (u - r) * math.log(u / r)
    return psi


def _centroid(rows) -> list[float] | None:
    vecs = [r.cls_embedding for r in rows if r.cls_embedding]
    if not vecs:
        return None
    dim = len(vecs[0])
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]


def _cosine_distance(a, b) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return None
    return 1.0 - dot / (na * nb)


def _drift_signal(rows: list) -> dict | None:
    """Recent half vs older half of `rows` (newest-first). None if too few samples."""
    n = len(rows)
    if n < 4:
        return None
    cur, ref = rows[: n // 2], rows[n // 2:]

    def counts(rs):
        d: dict[int, int] = {}
        for r in rs:
            if r.class_id is not None:
                d[r.class_id] = d.get(r.class_id, 0) + 1
        return d

    def mean_conf(rs):
        cs = [r.confidence for r in rs if r.confidence is not None]
        return sum(cs) / len(cs) if cs else None

    psi = _psi(counts(ref), counts(cur))
    cur_conf, ref_conf = mean_conf(cur), mean_conf(ref)
    conf_delta = (cur_conf - ref_conf) if (cur_conf is not None and ref_conf is not None) else None
    centroid_shift = _cosine_distance(_centroid(cur), _centroid(ref))
    drift = bool(psi >= _PSI_SIGNIFICANT or (conf_delta is not None and conf_delta <= -_CONF_DROP))
    return {
        "n_reference": len(ref), "n_current": len(cur),
        "class_psi": round(psi, 4),
        "psi_interpretation": ("significant" if psi >= _PSI_SIGNIFICANT
                               else "moderate" if psi >= 0.1 else "stable"),
        "confidence_reference": round(ref_conf, 4) if ref_conf is not None else None,
        "confidence_current": round(cur_conf, 4) if cur_conf is not None else None,
        "confidence_delta": round(conf_delta, 4) if conf_delta is not None else None,
        "embedding_centroid_cosine_shift": round(centroid_shift, 4) if centroid_shift is not None else None,
        "drift": drift,
        "message": "drift detected — consider relearn" if drift else "no significant drift",
    }


def evaluate(model_id: str, window: int = 200) -> dict:
    """General label-free evaluation of a model over its recent inference history:
    usage count, prediction mix, mean confidence, and a nested drift signal."""
    with SessionLocal() as s:
        rows = list(s.scalars(
            select(InferenceResult)
            .where(InferenceResult.model_id == model_id)
            .order_by(InferenceResult.id.desc())
            .limit(2 * window)
        ))
    confs = [r.confidence for r in rows if r.confidence is not None]
    dist: dict[str, int] = {}
    for r in rows:
        k = r.prediction if r.prediction is not None else str(r.class_id)
        dist[k] = dist.get(k, 0) + 1
    return {
        "model_id": model_id,
        "n_history": len(rows),
        "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
        "prediction_distribution": dist,
        "drift": _drift_signal(rows),
    }
