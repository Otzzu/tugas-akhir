"""Evaluation / monitoring endpoint — general, label-free model evaluation."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from API.services import registry
from API.services import eval as eval_service
from API.schemas.inference import EvalReport

router = APIRouter(tags=["eval"])


@router.get("/eval", response_model=EvalReport)
def evaluate(
    model_id: str,
    window: int = Query(
        200,
        description=(
            "Comparison-window size for the drift signal. The endpoint pulls the most "
            "recent 2×window stored predictions and splits them into the newest `window` "
            "(current) vs the `window` before it (reference); drift compares the two. "
            "Older predictions are ignored. Smaller = more sensitive but noisier; larger "
            "= more stable but slower to react. The general stats also cover these 2×window "
            "rows, so n_history is capped at 2×window, not the lifetime total."
        ),
    ),
) -> EvalReport:
    """General label-free evaluation of a model over its recent inference history.

    No ground-truth labels needed — it summarizes the predictions the model already made
    (stored per inference): usage count (`n_history`), prediction mix
    (`prediction_distribution`), `mean_confidence`, and a nested **drift** signal.

    Drift compares the newest `window` predictions (current) against the `window` before
    them (reference) using PSI on the class distribution + mean-confidence drop + pre-head
    embedding-centroid cosine shift. `drift.drift = true` is the trigger to relearn.
    Extensible to more eval facets later."""
    if model_id not in registry.load_models():
        raise HTTPException(404, f"Unknown model_id '{model_id}'")
    return EvalReport(**eval_service.evaluate(model_id, window))
