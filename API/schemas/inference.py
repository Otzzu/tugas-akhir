"""Inference request/response schemas."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    model_id: str = Field(..., description="Registered model id, e.g. 'n48', 'o1', 'seqgnn'")
    codes: list[str] = Field(..., description="List of C/C++/Java function source strings")
    top_k_lines: Optional[int] = Field(
        None, description="Return only the top-k suspicious lines per function (all if null)"
    )


class SuspiciousLine(BaseModel):
    line: int
    score: float                                  # vulnerability score [0,1], higher = worse
    predicted_cwe: Optional[str] = None           # only for per-line multiclass heads
    class_probabilities: Optional[dict[str, float]] = None


class FunctionResult(BaseModel):
    index: int
    ok: bool
    error: Optional[str] = None
    cached: Optional[bool] = None                 # True if the CPG came from the graph cache
    prediction: Optional[str] = None
    class_id: Optional[int] = None
    is_vulnerable: Optional[bool] = None
    confidence: Optional[float] = None
    class_probabilities: Optional[dict[str, float]] = None
    suspicious_lines: list[SuspiciousLine] = []


class InferenceResponse(BaseModel):
    model_id: str
    results: list[FunctionResult]
