"""Relearn request/response schemas."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RelearnMethod(str, Enum):
    er = "ER"
    ewc = "EWC"
    ewc_er = "EWC-ER"
    finetune = "finetune"
    retrain = "retrain"


class SplitSpec(BaseModel):
    # Mode B — library generates the split from ratios (test = 1 - train - val):
    train_ratio: Optional[float] = None
    val_ratio: Optional[float] = None
    seed: Optional[int] = None
    # Mode A — caller supplies the exact split, keyed on parquet_id (overrides ratios):
    train: Optional[list[int]] = None
    val: Optional[list[int]] = None
    test: Optional[list[int]] = None


class RelearnRequest(BaseModel):
    method: RelearnMethod
    dataset_ids: list[str] = Field(
        ..., description="Task-B dataset id(s). More than one are joined before training."
    )
    base_model_id: Optional[str] = Field(
        None,
        description="Required for ER/EWC/EWC-ER/finetune — model to continue from. Its dataset, "
        "config and checkpoint are read from the registry. Ignored for 'retrain'.",
    )
    epochs: Optional[int] = Field(None, description="Override training epochs")
    run_name: Optional[str] = Field(None, description="Optional human label for the job")
    split: Optional[SplitSpec] = Field(
        None,
        description="Optional split control. Mode A: explicit {train,val,test} parquet_id lists. "
        "Mode B: {train_ratio,val_ratio,seed} for the library to split. Omit = default 80/10/10 seed 42.")


class RelearnJob(BaseModel):
    job_id: str
    status: str                                   # queued | running | done | failed
    method: str
    dataset_ids: list[str]
    base_model_id: Optional[str] = None
    config_path: Optional[str] = None
    log_path: Optional[str] = None
    result_model_id: Optional[str] = None
    message: Optional[str] = None
    metrics: Optional[dict] = Field(
        None, description="Compact eval summary on the task-B test split (function-level "
        "classification + localization). The full metrics_summary, the training_summary, and the "
        "realized train/val/test split are stored as model_artifacts (kinds: metrics, "
        "training_summary, train_split).")
