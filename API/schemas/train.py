"""Train-from-scratch request/response schemas."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from API.schemas.relearn import SplitSpec


class TrainRequest(BaseModel):
    """Train a brand-new model from scratch. The architecture + train hyperparameters come from
    `config_id`; the data from `dataset_ids`. (Continuing an existing model is `/relearn`.)"""
    config_id: str = Field(
        ..., description="Config to instantiate — kind=model (architecture + train hyperparameters) "
        "or kind=full. Guarded: a kind=data config is rejected.")
    dataset_ids: list[str] = Field(
        ..., min_length=1, description="Dataset id(s) to train on. More than one are joined first.")
    epochs: Optional[int] = Field(None, description="Override training epochs")
    run_name: Optional[str] = Field(None, description="Optional human label for the job")
    split: Optional[SplitSpec] = Field(
        None, description="Optional split control (same modes as /relearn). Omit = default 80/10/10 seed 42.")


class TrainJob(BaseModel):
    job_id: str
    status: str                                   # queued | running | done | failed
    dataset_ids: list[str]
    config_id: Optional[str] = None
    config_path: Optional[str] = None
    log_path: Optional[str] = None
    result_model_id: Optional[str] = None
    message: Optional[str] = None
    metrics: Optional[dict] = Field(
        None, description="Compact eval summary on the test split (function-level classification + "
        "localization). Full artifacts are stored as model_artifacts, same as /relearn.")
