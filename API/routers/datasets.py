"""Dataset endpoints: async ingestion (raw data -> registered dataset) + job status.

The endpoint is intentionally thin — it enqueues a Celery job and polls its DB row.
The build pipeline lives in the gnn_vuln library (driven by API.tasks)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from API.services import datasets as ds_service, registry
from API.schemas import DatasetIngestRequest, DatasetJob

router = APIRouter(tags=["datasets"])

_JOB_KEYS = ("job_id", "status", "name", "dataset_id", "num_rows", "message")


def _job(d: dict) -> DatasetJob:
    return DatasetJob(**{k: d.get(k) for k in _JOB_KEYS})


@router.post("/datasets", response_model=DatasetJob, status_code=202)
def ingest_dataset(req: DatasetIngestRequest) -> DatasetJob:
    """Build a training-ready dataset. Provide EXACTLY ONE of `rows` (ingest from raw
    functions) or `dataset_ids` (merge >=2 registered datasets). Returns a queued job;
    poll GET /datasets/jobs/{job_id}. When done, `dataset_id` is usable in /relearn."""
    has_rows = bool(req.rows)
    has_ids = bool(req.dataset_ids)
    if has_rows == has_ids:
        raise HTTPException(422, "Provide exactly one of 'rows' or 'dataset_ids'")
    if has_ids:
        for did in req.dataset_ids:
            try:
                registry.get_dataset(did)
            except KeyError:
                raise HTTPException(422, f"Unknown dataset_id '{did}'")
        return _job(ds_service.create_merge_job(req))
    return _job(ds_service.create_ingest_job(req))


@router.get("/datasets/jobs")
def list_dataset_jobs(limit: int = 50) -> list[dict]:
    return ds_service.list_jobs(limit)


@router.get("/datasets/jobs/{job_id}", response_model=DatasetJob)
def dataset_job_status(job_id: str) -> DatasetJob:
    job = ds_service.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job_id '{job_id}'")
    return _job(job)
