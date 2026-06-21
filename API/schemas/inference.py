"""Inference request/response schemas."""
from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    model_id: str = Field(..., description="Registered model id, e.g. 'graph_based', 'hybrid_graph_lm', 'sequential'")
    codes: Annotated[list[Annotated[str, Field(max_length=200000)]], Field(min_length=1, max_length=64)] = Field(
        ..., description="List of C/C++/Java function source strings"
    )
    top_k_lines: Optional[int] = Field(
        None, description="Return only the top-k suspicious lines per function (all if null)"
    )


class SuspiciousLine(BaseModel):
    line: int
    score: float                                  # vulnerability score [0,1], higher = worse
    code: Optional[str] = None                    # the source text of that line
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
    cls_embedding_dim: Optional[int] = Field(
        None, description="Dimensionality of the stored pre-head function embedding "
                          "(full vector persisted for drift detection / similarity search, "
                          "not returned in this response)"
    )


class InferenceResponse(BaseModel):
    model_id: str
    results: list[FunctionResult]


class EmbedRequest(BaseModel):
    model_id: str = Field(..., description="Registered model id whose embedding space to use")
    codes: Annotated[list[Annotated[str, Field(max_length=200000)]], Field(min_length=1, max_length=64)] = Field(
        ..., description="List of C/C++/Java function source strings"
    )


class EmbedResult(BaseModel):
    index: int
    ok: bool
    error: Optional[str] = None
    cls_embedding: Optional[list[float]] = Field(
        None, description="Final pre-head function representation (the vector fed to the "
                          "output head). For similarity search / drift detection."
    )
    cls_embedding_dim: Optional[int] = None


class EmbedResponse(BaseModel):
    model_id: str
    results: list[EmbedResult]


class DriftSignal(BaseModel):
    """Label-free drift signal: the newest `window` predictions (current) vs the `window`
    before them (reference)."""
    drift: bool = Field(..., description="True if a significant shift is detected (PSI>=0.25 or confidence drop>=0.10)")
    message: str
    n_reference: int = Field(..., description="predictions in the reference (older) window")
    n_current: int = Field(..., description="predictions in the current (newest) window")
    class_psi: float = Field(..., description="Population Stability Index on the predicted-class distribution")
    psi_interpretation: str = Field(..., description="stable (<0.1) | moderate (0.1-0.25) | significant (>=0.25)")
    confidence_reference: Optional[float] = None
    confidence_current: Optional[float] = None
    confidence_delta: Optional[float] = Field(None, description="current minus reference mean confidence (negative = model less sure)")
    embedding_centroid_cosine_shift: Optional[float] = Field(None, description="cosine distance between the mean pre-head embedding of each window")


class EvalReport(BaseModel):
    """General label-free evaluation of a model over its recent inference history."""
    model_id: str
    n_history: int = Field(..., description="predictions analysed (capped at 2×window, not the lifetime total)")
    mean_confidence: Optional[float] = None
    prediction_distribution: dict[str, int] = Field({}, description="count per predicted class over the analysed rows")
    drift: Optional[DriftSignal] = Field(None, description="null until there are >=4 stored predictions")
