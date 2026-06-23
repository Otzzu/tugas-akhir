"""Inference endpoint: code -> CWE + suspicious lines."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from API.services import registry
from API.services import inference as inference_service
from API.schemas import (
    InferenceRequest, InferenceResponse, FunctionResult,
    EmbedRequest, EmbedResponse, EmbedResult,
)

router = APIRouter(tags=["inference"])


@router.post("/inference", response_model=InferenceResponse)
def inference(req: InferenceRequest) -> InferenceResponse:
    if req.model_id not in registry.load_models():
        raise HTTPException(404, f"Unknown model_id '{req.model_id}'")
    if not req.codes:
        raise HTTPException(422, "codes must be a non-empty list")
    try:
        raw = inference_service.infer_codes(
            req.model_id, req.codes, req.config.top_k_lines if req.config else None)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    return InferenceResponse(model_id=req.model_id,
                             results=[FunctionResult(**r) for r in raw])


@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    """Return the final pre-head function embedding per input — no classification output.
    For similarity search and drift detection. Runs the same CPG + model pass as
    /inference but yields the representation fed to the output head, not the prediction."""
    if req.model_id not in registry.load_models():
        raise HTTPException(404, f"Unknown model_id '{req.model_id}'")
    if not req.codes:
        raise HTTPException(422, "codes must be a non-empty list")
    try:
        raw = inference_service.embed_codes(req.model_id, req.codes)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    return EmbedResponse(model_id=req.model_id,
                         results=[EmbedResult(**r) for r in raw])


@router.get("/inference/history")
def inference_history(model_id: str | None = None, limit: int = 50) -> list[dict]:
    """Recent persisted predictions (newest first)."""
    return inference_service.list_results(model_id, limit)
