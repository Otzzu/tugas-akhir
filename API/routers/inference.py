"""Inference endpoint: code -> CWE + suspicious lines."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from API.services import registry
from API.services import inference as inference_service
from API.schemas import InferenceRequest, InferenceResponse, FunctionResult

router = APIRouter(tags=["inference"])


@router.post("/inference", response_model=InferenceResponse)
def inference(req: InferenceRequest) -> InferenceResponse:
    if req.model_id not in registry.load_models():
        raise HTTPException(404, f"Unknown model_id '{req.model_id}'")
    if not req.codes:
        raise HTTPException(422, "codes must be a non-empty list")
    try:
        raw = inference_service.infer_codes(req.model_id, req.codes, req.top_k_lines)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    return InferenceResponse(model_id=req.model_id,
                             results=[FunctionResult(**r) for r in raw])
