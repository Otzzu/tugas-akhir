"""Relearn endpoints: submit job, poll status, list jobs."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from API.services import registry
from API.services import relearn as relearn_service
from API.schemas import RelearnRequest, RelearnJob

router = APIRouter(tags=["relearn"])


@router.post("/relearn", response_model=RelearnJob)
def relearn(req: RelearnRequest) -> RelearnJob:
    method = req.method.value
    if method in relearn_service._REQUIRES_BASE and not req.base_model_id:
        raise HTTPException(422, f"method '{method}' requires base_model_id")
    for d in req.dataset_ids:
        if d not in registry.load_datasets():
            raise HTTPException(404, f"Unknown dataset_id '{d}'")
    if req.base_model_id and req.base_model_id not in registry.load_models():
        raise HTTPException(404, f"Unknown base_model_id '{req.base_model_id}'")
    try:
        job = relearn_service.submit_relearn(
            method, req.dataset_ids, req.base_model_id, req.epochs, req.run_name,
            split=req.split.model_dump(exclude_none=True) if req.split else None)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise HTTPException(422, str(e))
    return RelearnJob(**job)


@router.get("/relearn/{job_id}", response_model=RelearnJob)
def relearn_status(job_id: str) -> RelearnJob:
    job = relearn_service.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job_id '{job_id}'")
    return RelearnJob(**job)


@router.get("/relearn")
def relearn_list() -> list[dict]:
    return relearn_service.list_jobs()
