"""Celery application — async job queue for long-running ingestion / relearn.

The worker is a SEPARATE process (can live on a GPU box, scaled independently of
the web server):

    celery -A API.celery_app worker --loglevel=info --concurrency=1

Broker + result backend = Redis (settings.CELERY_BROKER_URL / CELERY_RESULT_BACKEND).
Tasks are defined in API.tasks; `include` registers them with the app.

Separation of concerns: tasks here ORCHESTRATE (job state, storage, registration)
and call the gnn_vuln LIBRARY for the actual work — the library knows nothing about
Celery, jobs, S3, or the API.
"""
from __future__ import annotations

from celery import Celery

from API.core.config import settings

celery_app = Celery(
    "vuln_api",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["API.tasks"],
)

celery_app.conf.update(
    task_track_started=True,          # report STARTED, not just PENDING -> SUCCESS
    task_acks_late=True,              # re-deliver if a worker dies mid-task
    worker_prefetch_multiplier=1,     # one heavy job at a time per worker
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=86_400,            # keep task results 1 day
)
