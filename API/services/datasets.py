"""Dataset-ingestion service — thin orchestration around the Celery task.

The endpoint stays oblivious to implementation: it calls create_ingest_job (which
persists the raw rows + a job row and enqueues the Celery task) and polls get_job.
The heavy raw->CPG->.pt build lives in the gnn_vuln library, driven by API.tasks.
"""
from __future__ import annotations

import re
import uuid

from API.core.config import settings
from API.core.database import SessionLocal
from API.models.tables import DatasetJobRecord


def _slug(name: str) -> str:
    base = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower() or "dataset"
    return base


def _rows_to_parquet(rows: list, path) -> int:
    """Normalize ingest rows into the BigVul-style columns the library loader reads:
    func_before / func_after / 'CWE ID' / vul / language."""
    import pandas as pd
    recs = []
    for r in rows:
        cwe = (r.cwe or "").strip()
        vul = 1 if (cwe or (r.label is not None and r.label > 0)) else 0
        recs.append({
            "func_before": r.code,
            "func_after": r.func_after or "",
            "CWE ID": cwe,
            "vul": vul,
            "language": r.language or "C",
        })
    df = pd.DataFrame(recs)
    df.to_parquet(path)
    return len(df)


def create_ingest_job(req) -> dict:
    """Persist input rows + a job row, enqueue the Celery task, return the job dict."""
    job_id = uuid.uuid4().hex[:16]
    dataset_id = f"{_slug(req.name)}_{job_id[:8]}"
    job_dir = settings.JOBS_DIR / "datasets" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / "input.parquet"
    n_rows = _rows_to_parquet(req.rows, input_path)

    data_config = {**req.data_config.model_dump(), "name": req.name,
                   "dataset_id": dataset_id, "input_path": str(input_path),
                   "job_dir": str(job_dir)}

    with SessionLocal() as s:
        s.add(DatasetJobRecord(job_id=job_id, status="queued", name=req.name,
                               data_config=data_config, num_rows=n_rows,
                               log_path=str(job_dir / "ingest.log")))
        s.commit()

    # enqueue on the Celery worker (import here to avoid a hard dependency at API import time)
    from API.tasks import ingest_dataset
    ingest_dataset.delay(job_id)

    return get_job(job_id)


def create_merge_job(req) -> dict:
    """Persist a merge job row, enqueue the Celery merge task, return the job dict.
    Merges >=2 already-registered datasets into a new one (no raw rows)."""
    job_id = uuid.uuid4().hex[:16]
    dataset_id = f"{_slug(req.name)}_{job_id[:8]}"
    job_dir = settings.JOBS_DIR / "datasets" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    data_config = {**req.data_config.model_dump(), "name": req.name,
                   "dataset_id": dataset_id, "dataset_ids": list(req.dataset_ids),
                   "dedup": req.dedup, "job_dir": str(job_dir)}

    with SessionLocal() as s:
        s.add(DatasetJobRecord(job_id=job_id, status="queued", name=req.name,
                               data_config=data_config, num_rows=0,
                               log_path=str(job_dir / "merge.log")))
        s.commit()

    from API.tasks import merge_datasets
    merge_datasets.delay(job_id)

    return get_job(job_id)


def get_job(job_id: str) -> dict | None:
    with SessionLocal() as s:
        j = s.get(DatasetJobRecord, job_id)
        return j.to_dict() if j else None


def list_jobs(limit: int = 50) -> list[dict]:
    from sqlalchemy import select
    with SessionLocal() as s:
        q = select(DatasetJobRecord).order_by(DatasetJobRecord.created_at.desc()).limit(limit)
        return [j.to_dict() for j in s.scalars(q).all()]
