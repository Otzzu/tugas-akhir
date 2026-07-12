"""Relearn endpoints: submit job, poll status, list jobs."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from API.services import registry
from API.services import relearn as relearn_service
from API.schemas import RelearnRequest, RelearnJob

router = APIRouter(tags=["relearn"])


@router.post("/relearn", response_model=RelearnJob)
def relearn(req: RelearnRequest) -> RelearnJob:
    """Submit a continual-learning job (async).

    Materializes the task-B dataset(s), trains a new model via the gnn_vuln library, evaluates
    it on the task-B test split (metrics-only), and registers the result. The library runs in
    API mode (`GNN_VULN_API_MODE`): no research artifacts — per-sample CSVs, ROC/confusion/PR
    plots, training log/curves — touch the worker disk. The metrics, the realized
    train/val/test split, and the training summary are persisted as per-model `model_artifacts`
    (DB + object storage), then the local results dir is deleted. Poll `GET /relearn/{job_id}`.
    """
    method = req.method.value
    if method in relearn_service._REQUIRES_BASE and not req.base_model_id:
        raise HTTPException(422, f"method '{method}' requires base_model_id")
    for d in req.dataset_ids:
        if d not in registry.load_datasets():
            raise HTTPException(404, f"Unknown dataset_id '{d}'")
    for role_id in (req.val_dataset_id, req.test_dataset_id):
        if role_id and role_id not in registry.load_datasets():
            raise HTTPException(404, f"Unknown dataset_id '{role_id}'")
    if req.base_model_id and req.base_model_id not in registry.load_models():
        raise HTTPException(404, f"Unknown base_model_id '{req.base_model_id}'")
    try:
        cfg = req.config
        job = relearn_service.submit_relearn(
            method, req.dataset_ids, req.base_model_id,
            cfg.epochs if cfg else None, req.run_name,
            split=cfg.split.model_dump(exclude_none=True) if (cfg and cfg.split) else None,
            val_dataset_id=req.val_dataset_id, test_dataset_id=req.test_dataset_id)
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
