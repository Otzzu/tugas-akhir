"""Train-from-scratch request/response schemas."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from API.schemas.relearn import RunConfig


class TrainConfig(RunConfig):
    """Run-config overrides for /train — RunConfig (epochs, split) plus an optional architecture
    override applied on top of config_id."""
    model_type: Optional[str] = Field(
        None,
        description="Override the architecture from config_id (e.g. lmgat_codebert, lmgat_seqgnn). "
        "Omit = use the architecture in config_id.")


class TrainRequest(BaseModel):
    """Train a brand-new model from scratch. The architecture + train hyperparameters come from
    `config_id`; the data from `dataset_ids`. (Continuing an existing model is `/relearn`.)"""
    config_id: str = Field(
        ..., description="Registered config to instantiate the model (architecture + train "
        "hyperparameters). The dataset(s) provide the data.")
    dataset_ids: list[str] = Field(
        ..., min_length=1, description="Dataset id(s) to train on. More than one are joined first.")
    run_name: Optional[str] = Field(None, description="Optional human label for the job")
    config: Optional[TrainConfig] = Field(
        None,
        description="Config overrides on config_id (epochs, split, model_type). Omit = use config_id as-is.")


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
