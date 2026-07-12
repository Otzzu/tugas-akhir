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
    val_dataset_id: Optional[str] = Field(
        None,
        description="Registered dataset to use as the VALIDATION set (role dataset). When set, "
        "training uses 100% of dataset_ids and early-stops on this dataset instead of an "
        "internal split. Label space must match.")
    test_dataset_id: Optional[str] = Field(
        None,
        description="Registered dataset to use as the TEST set (e.g. a fixed golden benchmark "
        "reused across model versions). Requires val_dataset_id. Label space must match.")
    run_name: Optional[str] = Field(None, description="Optional human label for the job")
    config: Optional[TrainConfig] = Field(
        None,
        description="Config overrides on config_id (epochs, split, model_type). Omit = use config_id as-is.")


class TrainJob(BaseModel):
    job_id: str
    status: str                                   # queued | running | done | failed
    dataset_ids: list[str]
    config_id: Optional[str] = None
    val_dataset_id: Optional[str] = None
    test_dataset_id: Optional[str] = None
    config_path: Optional[str] = None
    log_path: Optional[str] = None
    result_model_id: Optional[str] = None
    message: Optional[str] = None
    metrics: Optional[dict] = Field(
        None, description="Compact eval summary on the test split (function-level classification + "
        "localization). Full artifacts are stored as model_artifacts, same as /relearn.",
        json_schema_extra={"examples": [{
            "function_level": {"f1_macro": 0.474, "f1_weighted": 0.480, "accuracy": 0.474,
                               "auc_roc_macro_ovr": 0.891, "confidence_mean": 0.52,
                               "num_test_samples": 1073},
            "localization": {"ifa_mean": 12.31, "top_1_accuracy": 0.260, "top_5_accuracy": 0.589,
                             "recall_at_5pct_loc": 0.089, "recall_at_20pct_loc": 0.289},
        }]})
