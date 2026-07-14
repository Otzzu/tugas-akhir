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


class SplitSpec(BaseModel):
    """Split control, two modes. Mode B (ratios): the library splits with a seeded
    shuffle. Mode A (explicit lists): the caller pins EXACTLY which rows go where,
    keyed on each row's `id` from the dataset's raw rows (parquet_id) — overrides
    the ratios. Lists don't have to cover every row; uncovered rows are unused."""
    # Mode B — library generates the split from ratios (test = 1 - train - val):
    train_ratio: Optional[float] = Field(None, description="Mode B: train fraction, e.g. 0.8")
    val_ratio: Optional[float] = Field(None, description="Mode B: val fraction, e.g. 0.1")
    seed: Optional[int] = Field(None, description="Mode B: shuffle seed, e.g. 42")
    # Mode A — caller supplies the exact split, keyed on parquet_id (overrides ratios):
    train: Optional[list[int]] = Field(None, description="Mode A: row ids for TRAIN")
    val: Optional[list[int]] = Field(None, description="Mode A: row ids for VAL")
    test: Optional[list[int]] = Field(None, description="Mode A: row ids for TEST")

    model_config = {"json_schema_extra": {"examples": [
        {"train_ratio": 0.8, "val_ratio": 0.1, "seed": 42},
        {"train": [12154, 40412, 47121], "val": [6065, 88098], "test": [251, 9932]},
    ]}}


class RunConfig(BaseModel):
    """Config overrides applied on top of the inherited config for ONE run — kept separate from
    the job spec (method, datasets, base_model). Omit any field to keep the inherited value."""
    epochs: Optional[int] = Field(None, description="Override training epochs")
    split: Optional[SplitSpec] = Field(
        None,
        description="Optional split control. Mode A: explicit {train,val,test} parquet_id lists. "
        "Mode B: {train_ratio,val_ratio,seed} for the library to split. Omit = the base config's "
        "split, which for the production models is 90/10/0 — no test holdout, val drives early "
        "stopping (the honest test in service is the next dataset that arrives).")


class RelearnRequest(BaseModel):
    method: RelearnMethod
    dataset_ids: list[str] = Field(
        ..., description="Task-B dataset id(s). More than one are joined before training."
    )
    base_model_id: Optional[str] = Field(
        None,
        description="Required — the model to continue from. Its dataset, config and checkpoint "
        "are read from the registry. To train a fresh model instead, use POST /train.",
    )
    val_dataset_id: Optional[str] = Field(
        None,
        description="Registered dataset to use as the VALIDATION set. When set, training uses "
        "100% of dataset_ids and early-stops on this one instead of an internal split. Label "
        "space must match.")
    test_dataset_id: Optional[str] = Field(
        None,
        description="Registered dataset to use as the TEST set — e.g. a fixed benchmark, so the "
        "number is comparable across model versions. Independent of val_dataset_id: it also works "
        "with a ratio split. Omit it (and leave no test remainder) and the model is registered "
        "WITHOUT metrics. Label space must match.")
    run_name: Optional[str] = Field(None, description="Optional human label for the job")
    config: Optional[RunConfig] = Field(
        None,
        description="Config overrides on the base model's config for this run (epochs, split). "
        "Everything else here is job spec, not config. Omit = use the base config as-is.")


class RelearnJob(BaseModel):
    job_id: str
    status: str                                   # queued | running | done | failed
    method: str
    dataset_ids: list[str]
    base_model_id: Optional[str] = None
    val_dataset_id: Optional[str] = None
    test_dataset_id: Optional[str] = None
    config_path: Optional[str] = None
    log_path: Optional[str] = None
    result_model_id: Optional[str] = None
    message: Optional[str] = None
    metrics: Optional[dict] = Field(
        None, description="Compact eval summary on the task-B test split (function-level "
        "classification + localization). The full metrics_summary, the training_summary, and the "
        "realized train/val/test split are stored as model_artifacts (kinds: metrics, "
        "training_summary, train_split).",
        json_schema_extra={"examples": [{
            "function_level": {"f1_macro": 0.474, "f1_weighted": 0.480, "accuracy": 0.474,
                               "auc_roc_macro_ovr": 0.891, "confidence_mean": 0.52,
                               "num_test_samples": 1073},
            "localization": {"ifa_mean": 12.31, "top_1_accuracy": 0.260, "top_5_accuracy": 0.589,
                             "recall_at_5pct_loc": 0.089, "recall_at_20pct_loc": 0.289},
        }]})
