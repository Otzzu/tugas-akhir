"""Dataset-ingestion request/response schemas."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DatasetRow(BaseModel):
    id: Optional[str | int] = Field(
        None, description="Caller's row id (provenance, any string/int). Auto-generated "
        "from the row position when absent. Not used by the build itself.")
    cve_id: Optional[str] = Field(None, description="CVE id, e.g. 'CVE-2019-12111' (provenance)")
    code: str = Field(..., description="Function source (the 'before' code)")
    cwe: Optional[str] = Field(None, description="CWE id, e.g. 'CWE-787' (multiclass label)")
    label: Optional[int] = Field(None, description="Explicit class id (alternative to cwe; 0 = benign)")
    func_after: Optional[str] = Field(
        None,
        description="Patched function; flaw lines derived from the diff. Required on a "
        "vulnerable row unless flaw_lines is given.")
    flaw_lines: Optional[list[int]] = Field(
        None, min_length=1,
        description="Manually annotated 1-indexed vulnerable lines of `code`. Required on a "
        "vulnerable row unless func_after is given. When both are present flaw_lines wins "
        "and func_after is kept as the fallback. Omit the field when absent; an empty list "
        "is not a valid annotation. Ignored on benign rows.")
    language: Optional[str] = Field(None, description="Language name (default C)")


class DataConfigOverride(BaseModel):
    """Data-build config — frozen onto the dataset. All overridable; defaults are
    production-sensible (small val split, no test held out — test = forgetting eval
    on old data + drift monitoring)."""
    mode: str = Field("multiclass", description="multiclass | binary")
    val_fraction: float = Field(0.1, ge=0.0, lt=1.0, description="0 = 100% train (fixed epochs, no early stop)")
    test_fraction: float = Field(0.0, ge=0.0, lt=1.0, description="Held-out test fraction (prod default 0)")
    max_nodes: int = Field(2500, description="Skip CPGs larger than this")
    top_cwe: int = Field(0, description="Keep only top-N CWE classes (0 = all)")
    max_per_class: int = Field(0, description="Balance: max samples per class (0 = unlimited)")
    resample_seed: int = Field(42, description="Seed for sampling / splits")


class DatasetIngestRequest(BaseModel):
    """Provide EXACTLY ONE of `rows` (ingest from raw functions) or `dataset_ids`
    (merge >=2 already-registered datasets into a new one)."""
    name: str = Field(..., description="Human-readable dataset name / label")
    rows: Optional[list[DatasetRow]] = Field(None, min_length=1, max_length=5000, description="Raw functions to build the dataset from (ingest path)")
    dataset_ids: Optional[list[str]] = Field(None, min_length=2, description="Registered dataset ids to MERGE into a new dataset (merge path)")
    dedup: bool = Field(True, description="Drop duplicate functions when merging")
    data_config_id: Optional[str] = Field(
        None,
        description="Base data-build config to reuse (prior). The `config` object below overrides "
        "its fields. Omit to use `config` alone.")
    config: DataConfigOverride = Field(
        default_factory=DataConfigOverride,
        description="Data-build params — inline config when data_config_id is absent, or field "
        "overrides on top of data_config_id when it is set.")


class DatasetJob(BaseModel):
    job_id: str
    status: str                                  # queued | running | done | failed
    name: str
    dataset_id: Optional[str] = None             # set when status == done
    raw_id: Optional[str] = Field(
        None, description="raw_datasets row backing this dataset (set when done); "
        "download the rows via GET /datasets/{dataset_id}/raw")
    num_rows: Optional[int] = None
    message: Optional[str] = None
