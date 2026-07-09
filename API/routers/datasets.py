"""Dataset endpoints: async ingestion (raw data -> registered dataset) + job status.

The endpoint is intentionally thin — it enqueues a Celery job and polls its DB row.
The build pipeline lives in the gnn_vuln library (driven by API.tasks)."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

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
    try:
        return _job(ds_service.create_ingest_job(req))
    except ds_service.UploadParseError as e:
        raise HTTPException(422, str(e))


@router.post("/datasets/upload", response_model=DatasetJob, status_code=202)
async def upload_dataset(
    file: UploadFile = File(..., description=".json (array of rows, or object with a 'rows' array) or .jsonl (one row per line)"),
    name: str | None = Form(None, description="Dataset name; falls back to the file's 'name', then the filename"),
    data_config_id: str | None = Form(None, description="Base data-build config to reuse (prior)"),
    config: str | None = Form(None, description="DataConfigOverride as a JSON object string; overrides the file's 'config'"),
) -> DatasetJob:
    """Ingest a dataset from an uploaded file. Same pipeline as POST /datasets with `rows`,
    but the row cap is API_MAX_UPLOAD_ROWS instead of the inline 5000. Returns a queued job;
    poll GET /datasets/jobs/{job_id}."""
    try:
        req = ds_service.parse_upload(await file.read(), file.filename or "", name, data_config_id, config)
    except ds_service.UploadParseError as e:
        raise HTTPException(422, str(e))
    if req.data_config_id:
        try:
            registry.get_config(req.data_config_id)
        except KeyError:
            raise HTTPException(422, f"Unknown data_config_id '{req.data_config_id}'")
    try:
        return _job(ds_service.create_ingest_job(req))
    except ds_service.UploadParseError as e:
        raise HTTPException(422, str(e))


@router.get("/datasets/jobs")
def list_dataset_jobs(limit: int = 50) -> list[dict]:
    return ds_service.list_jobs(limit)


@router.get("/datasets/jobs/{job_id}", response_model=DatasetJob)
def dataset_job_status(job_id: str) -> DatasetJob:
    job = ds_service.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job_id '{job_id}'")
    return _job(job)
