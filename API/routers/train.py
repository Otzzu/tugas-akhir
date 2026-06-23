"""Train-from-scratch endpoint: submit job, poll status, list jobs."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from API.services import registry
from API.services import relearn as relearn_service
from API.schemas import TrainRequest, TrainJob

router = APIRouter(tags=["train"])


@router.post("/train", response_model=TrainJob)
def train(req: TrainRequest) -> TrainJob:
    """Train a NEW model from scratch (async).

    Architecture + train hyperparameters come from `config_id`; data from `dataset_ids`. Runs on
    the same worker pipeline as /relearn but with fresh weights — no EWC
    importance, no replay buffer. Registers the trained model and stores its metrics + split +
    training summary as model_artifacts. Poll `GET /train/{job_id}`. To CONTINUE an existing
    model instead of starting fresh, use `/relearn`.
    """
    for d in req.dataset_ids:
        if d not in registry.load_datasets():
            raise HTTPException(404, f"Unknown dataset_id '{d}'")
    try:
        registry.get_config(req.config_id)             # ensure the config exists (KeyError -> 422)
        cfg = req.config
        job = relearn_service.submit_train(
            req.config_id, req.dataset_ids,
            cfg.epochs if cfg else None, req.run_name,
            split=cfg.split.model_dump(exclude_none=True) if (cfg and cfg.split) else None)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise HTTPException(422, str(e))
    return TrainJob(**job)


@router.get("/train/{job_id}", response_model=TrainJob)
def train_status(job_id: str) -> TrainJob:
    job = relearn_service.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job_id '{job_id}'")
    return TrainJob(**job)


@router.get("/train")
def train_list() -> list[dict]:
    return relearn_service.list_jobs()
