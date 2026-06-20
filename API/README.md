# Vulnerability Detection API

Production FastAPI service over the trained GNN models.

1. **Inference** — send raw function source; the API builds a Joern CPG (cached in the DB),
   embeds nodes, runs the model, returns predicted CWE + suspicious lines with scores.
2. **Relearn** — launch a continual-learning job (5 methods) over the training pipeline.

## Structure

```
API/
  core/        config.py (settings/env), database.py (engine, session, init)
  models/      tables.py — SQLAlchemy ORM (models, datasets, relearn_jobs, graph_cache)
  schemas/     pydantic request/response (inference, relearn)
  services/    registry, seed, graph_cache, inference, relearn (business logic)
  routers/     meta (/health /models /datasets), inference, relearn
  seeds/       models.json, datasets.json (DB seed on first boot)
  main.py      app wiring (lifespan init_db + seed, CORS, routers)
  Dockerfile, docker-compose.yml, requirements.txt
```

## Run (dev, uv)

```bash
# install the API extra into the project venv (uv)
uv sync --extra api                  # or: uv pip install -e ".[api]"

# run (PYTHONPATH=src so the gnn_vuln package is importable)
PYTHONPATH=src uv run uvicorn API.main:app --host 0.0.0.0 --port 8000
```

Host requirements: project deps (torch, torch-geometric, transformers), **Joern** installed
(set `JOERN_CLI`, default `C:/joern/joern-cli`). GPU optional (`API_DEVICE=cpu|cuda`).
Docs at `http://localhost:8000/docs`.

## Run (prod, Docker + PostgreSQL)

```bash
docker compose -f API/docker-compose.yml up --build
```
Brings up PostgreSQL + the API (Joern + JDK baked into the image, deps installed with uv).
Checkpoints + `data/` are mounted from the host. Env: `DATABASE_URL`, `JOERN_CLI`,
`API_DEVICE`, `API_CORS_ORIGINS`, `API_MAX_NODES`.

## Database

SQLAlchemy. `DATABASE_URL` selects the backend — `sqlite:///API/app.db` (dev default) or
`postgresql+psycopg://user:pass@host/db` (prod). Tables: **models**, **datasets**,
**relearn_jobs**, **graph_cache** (created + seeded automatically on first boot).

Register real checkpoints by editing `API/seeds/models.json` (replace the `REPLACE_WITH_*`
placeholders for `o1`/`seqgnn`) before first boot, or POST to the DB via the registry. Models
trained via `/relearn` are inserted automatically.

## Endpoints

### `POST /inference`
```json
{ "model_id": "n48", "codes": ["void f(char*s){ char b[8]; strcpy(b, s); }"], "top_k_lines": 5 }
```
Per function: `prediction` (CWE), `confidence`, `class_probabilities`,
`suspicious_lines: [{ "line": 14, "score": 0.92 }]`, `cached` (graph-cache hit), and
`ok`/`error` so one bad function doesn't fail the batch.

### `POST /relearn`
```json
{ "method": "EWC-ER", "dataset_ids": ["relearn"], "base_model_id": "n48", "epochs": 100 }
```
- `method`: `ER` | `EWC` | `EWC-ER` | `finetune` | `retrain`.
- `dataset_ids`: task-B dataset(s); multiple → joined before training.
- `base_model_id`: required for `ER`/`EWC`/`EWC-ER`/`finetune` (gives arch/config/start-weights
  + the dataset for the replay buffer and EWC importance). `retrain` = from scratch.

Returns a job `{job_id, status, ...}`; runs in the background (EWC importance then training),
then registers the resulting checkpoint as a new model id.

### `GET /relearn/{job_id}` status · `GET /relearn` list · `GET /models` `GET /datasets` `GET /health`

| method | start weights | EWC penalty | replay |
|---|---|---|---|
| finetune | base checkpoint | off | no |
| EWC | base checkpoint | weight 1000 | no |
| ER | base checkpoint | off | base-dataset buffer |
| EWC-ER | base checkpoint | weight 1000 | base-dataset buffer |
| retrain | random init | off | no |

CIL (task-B adds new classes) is automatic — `num_classes` grows and the base head loads
expandably (26→N).
