"""
FastAPI app — vulnerability detection inference + continual-learning relearn.

Run (dev, from repo root):
    PYTHONPATH=src uvicorn API.main:app --host 0.0.0.0 --port 8000
Run (prod): docker compose up (see API/docker-compose.yml).

Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from API.core.config import settings
from API.core.database import init_db
from API.core.observability import (
    logging_middleware, setup_logging, unhandled_exception_handler,
)
from API.routers import meta, inference, relearn, train, datasets, eval as eval_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    init_db()                       # create tables + seed on first boot
    yield


app = FastAPI(
    title="GNN Vulnerability Detection API",
    description="Inference (code -> CWE + vulnerable lines) and continual-learning relearn.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.middleware("http")(logging_middleware)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(meta.router)
app.include_router(inference.router)
app.include_router(relearn.router)
app.include_router(train.router)
app.include_router(datasets.router)
app.include_router(eval_router.router)
