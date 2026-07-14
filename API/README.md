# Vulnerability Detection API

Production FastAPI service over three trained GNN/LM vulnerability-detection models. It
classifies a C/C++/Java function by CWE, localizes the suspicious lines, exposes the
pre-head function embedding, ingests new datasets, retrains continually, and monitors
for drift.

The model code is a separate installable **library** (`gnn_vuln`); this service depends on
it as a package — it never reaches into the model source tree. See
[The `gnn_vuln` library](#the-gnn_vuln-library) below.

> **Full endpoint reference (request/response schemas, examples) lives in
> [`openapi.json`](openapi.json)** — view it at the live docs `http://localhost:8000/docs`,
> or open `openapi.json` with any OpenAPI viewer (e.g. the VS Code OpenAPI extension). This
> README covers architecture and the bits the spec can't: how the pieces fit, the storage
> model, and the library API.

---

## Capabilities

| Endpoint          | What it does                                                                            |
| ----------------- | --------------------------------------------------------------------------------------- |
| `POST /inference` | function source → CWE + confidence + suspicious lines (per-line scores)                 |
| `POST /embed`     | function source → final pre-head embedding (similarity search / drift), no output head  |
| `POST /datasets`  | raw rows → Joern CPG → `.pt`, **or** merge existing `dataset_ids` → new dataset, in object storage |
| `GET /datasets/{id}/raw` | download the exact raw rows (JSON, upload format) the dataset's `.pt` was built from |
| `POST /relearn`   | continual-learning job — CONTINUE a base model (finetune / EWC / ER / EWC-ER) → new model |
| `POST /train`     | train a NEW model from scratch — `config_id` + `dataset_ids` → new model |
| `GET /eval`       | general label-free model eval over stored predictions (usage, mix, confidence + drift)  |
| `GET …`           | `/health` `/models` `/datasets` `/configs` `/inference/history` + job status            |

---

## System architecture

```mermaid
flowchart LR
    client([Client])

    subgraph svc[API service]
      api[FastAPI app]
      wgpu[worker-gpu<br/>concurrency 1 · GPU]
      wcpu[worker-cpu<br/>concurrency 2 · CPU]
    end

    subgraph broker[Redis broker]
      qr[[queue: relearn]]
      qi[[queue: ingest]]
      qm[[queue: merge]]
    end

    pg[(PostgreSQL<br/>pgvector)]
    minio[(MinIO / S3<br/>object store)]
    joern[[Joern CLI<br/>+ JDK 21]]
    lib[[gnn_vuln library<br/>model + trainer]]

    client -->|sync · /inference /embed /eval| api
    client -->|async · /datasets /relearn| api

    api -->|metadata, pointers| pg
    api -->|CPG, datasets, checkpoints| minio
    api -->|sync: Joern + model forward| joern
    api --> lib

    api -->|run_relearn| qr
    api -->|ingest_dataset| qi
    api -->|merge_datasets| qm

    qr --> wgpu
    qi --> wgpu
    qm --> wcpu

    wgpu -->|train / ingest subprocess| lib
    wgpu --> joern
    wgpu --> minio
    wgpu --> pg
    wcpu -->|merge subprocess| lib
    wcpu --> minio
    wcpu --> pg
```

- **FastAPI app** — request/response. Synchronous, fast paths (inference, embed, eval) run
  inline; long jobs (dataset ingestion, relearn training) are **enqueued to Celery** and
  return a job id immediately.
- **Celery workers** — separate processes for the heavy work, split by **per-task queue**
  (broker = Redis):
  - `worker-gpu` (`-Q relearn,ingest`, concurrency 1) — GPU-bound tasks (training subprocess,
    ingest node-embedding). One at a time → they serialize on a single GPU (no OOM).
  - `worker-cpu` (`-Q merge`, concurrency 2) — CPU-only `.pt` merge, runs in parallel and is
    never blocked by a long train.
  Routing lives in `celery_app.py` (`task_routes`). A second GPU box later = give `ingest` its
  own worker with `-Q ingest`, no code change.
- **PostgreSQL** (pgvector image) — metadata + pointers only (never blobs). pgvector is ready
  for ANN search over the stored embeddings.
- **MinIO/S3** — object storage for the big binaries (CPG blobs, dataset `.pt` bundles, model checkpoints).
- **Joern + JDK 21** — Code Property Graph generation, baked into the image.
- **`gnn_vuln`** — installed library: the 3 model architectures, the CPG→`.pt` pipeline, the
  trainer, and the `VulnPredictor` inference wrapper.

### The three models

All three share the CodeBERT/UniXcoder node featurization + a GNN; they differ in how the
function-level LM is fused. Detail in `docs/bab-3.md`; model ids match those names.

| `model_id`        | Architecture (bab-3)        | arch flag        | LM fusion                            |
| ----------------- | --------------------------- | ---------------- | ------------------------------------ |
| `graph_based`     | Arsitektur Berbasis Graph   | `lmgat_codebert` | GNN-only (frozen LM node features)   |
| `hybrid_graph_lm` | Arsitektur Hibrida Graph–LM | `lmgat_codebert` | GNN ⊕ live function-LM (late fusion) |
| `sequential`      | Arsitektur Sekuensial       | `lmgat_seqgnn`   | localize→classify two-stage GNN      |

### Request flows

- **Inference / Embed** — `code → graph_cache lookup (sha256) → [miss] Joern CPG → node
embed + GNN/LM forward → prediction + pre-head embedding`. The embedding is captured by an
  eval-only forward-pre-hook on the output head and persisted (drift/search); `/inference`
  returns only its dim, `/embed` returns the vector.
- **Ingest** (`POST /datasets` with `rows`, async) — `rows → parquet → gnn_vuln.data.prepare
(Joern CPG + cwe_vocab) → data-build config → gnn_vuln.data.build_pt (.pt) → tar(.pt + vocab)
→ MinIO → register dataset_id (+ immutable data_config_id)`. Poll `GET /datasets/jobs/{id}`.
  Pass `data_config_id` to reuse a registered config as the base (prior); the `config` object then
  overrides its fields. Omit `data_config_id` to use `config` alone.
  The exact raw rows are also persisted to MinIO as `{raw_id}.json` (canonical
  upload-format JSON, row ids materialized) and registered as a first-class
  `raw_datasets` row with a content-addressed id (`raw_` + sha256 — identical uploads
  dedupe to one row, and one raw can back many datasets built with different configs).
  Download via `GET /datasets/{id}/raw`. Rows may carry optional provenance fields `id`
  (auto-generated from the row position when absent) and `cve_id`; a vulnerable row may
  supply `flaw_lines`, `func_after`, or both (flaw_lines wins, func_after is the
  fallback). Lineage is queryable in the DB: `models.dataset_id → datasets.raw_id →
  raw_datasets.storage_uri`, merges via `datasets.source_dataset_ids`.
- **Combining datasets** — three ways: (A) send all rows in one ingest; (B) `POST /datasets`
  with `dataset_ids: [X, Y]` → materialize each `.pt` → `gnn_vuln.data.merge` (concat graphs +
  unify the label space + remap + best-effort dedup) → a NEW reusable `dataset_id`; (C) `POST
  /relearn` with multiple `dataset_ids` → the same `.pt`-level merge inline at train time. All
  three merge at the `.pt` level (no raw CPG, no re-embedding). A merged dataset has no raw
  of its own — it registers `source_dataset_ids`, and each source keeps its own `raw_id`.
- **Train from scratch** (`POST /train`, async) — a NEW model with fresh weights from
  `config_id` + `dataset_ids`. `config.model_type` optionally overrides the architecture from
  `config_id` (else the config's architecture is used). Shares the relearn worker pipeline
  (`build_config` with `method=retrain` → `gnn_vuln.train` → evaluate → register), just with no
  base checkpoint, no EWC importance, no replay. Poll `GET /train/{id}`. Continuing an existing
  model is `/relearn`, not this.
- **Role datasets (val/test)** — both `/train` and `/relearn` accept optional `val_dataset_id`
  and `test_dataset_id` (SageMaker channel style): training then uses 100% of `dataset_ids`,
  early-stops on the val dataset, and reports test on the test dataset — e.g. a fixed golden
  benchmark dataset reused across model versions so metrics are comparable. Validated:
  featurization must match; label space must match exactly on `/train`, and on `/relearn` must
  be a SUBSET of the model's final class space (so a 26-class benchmark is valid against a
  36-class CIL model — target_vocab remaps it). Each role dataset is loaded with its own build
  identity (`source_{val,test}_params`), so CIL task data with different filters still finds
  the benchmark's `.pt`. Omit both for the internal val split (prod default: no test holdout).
  The ids are recorded on the job and the resulting model (lineage).
- **Relearn** (`POST /relearn`, async — CONTINUES a base model; for a fresh model use `/train`) —
  `materialize task-A+B datasets to local disk →
build merged config → [EWC] importance pass → gnn_vuln.train → register new model →
gnn_vuln.evaluate (task-B test, metrics-only) → capture metrics + split + training summary →
drop the local results dir`. Poll `GET /relearn/{id}`: returns `result_model_id` + a compact
  `metrics` summary (function-level + localization); the full `metrics_summary.json`, the
  `training_summary.json`, the EWC importance, and the realized train/val/test split are stored
  as per-model `model_artifacts`.
  - **No research artifacts on the worker** — train + eval run with `GNN_VULN_API_MODE=1`, so
    the bulky/per-sample research outputs (`predictions.csv`, `localization_scores.csv`,
    `embeddings.npz`, ROC / confusion / PR plots, `training_log.csv`, `training_curves.png`)
    are **never written**;
    only the small JSON handoffs (`metrics_summary`, `split`, `training_summary`) are produced,
    harvested to DB + object storage, then the local `results/<run>` dir is deleted. Object
    storage + DB are the source of truth; local disk keeps only the dataset cache + checkpoint.
  - **Request body** separates the **config** part from the job spec. Run-config overrides
    (`epochs`, `split`) live under a `config` object; `method`, `dataset_ids`, `base_model_id`,
    `run_name` stay top-level. `/train` and `/inference` follow the same shape (`config` =
    `{epochs, split}` for train, `{top_k_lines}` for inference). Omit `config` to inherit as-is.
  - **Split control** (optional `config.split`) — two modes, both recorded in the immutable
    config + a `train_split` artifact:
    - **Mode B (ratios)** — `{train_ratio, val_ratio, seed}`; the library splits (test =
      1 − train − val). E.g. `{0.9, 0.1, 42}` = prod 90/10/0 (no test → no eval metrics);
      omit `config.split` = default 80/10/10 seed 42 (report).
    - **Mode A (explicit)** — `{train, val, test}` parquet-id lists; written to a `split_file`
      that overrides ratios (bring-your-own split, e.g. to match a baseline or reproduce a run).
    Metrics need a test split + flaw-line GT to be complete; a 90/10/0 (test-empty) model is
    intentionally not evaluated.
- **Eval** (`GET /eval`) — general label-free report over stored predictions: usage count,
  prediction mix, mean confidence, plus a nested drift signal (PSI on class distribution +
  mean-confidence delta + embedding-centroid cosine shift). The drift signal _triggers_ a
  relearn.

---

## Layout

```
API/
  core/        config.py (settings/env), database.py (engine, session, init+seed)
  models/      tables.py — ORM: configs, models, model_artifacts, datasets, relearn_jobs,
               dataset_jobs, graph_cache, inference_results
  schemas/     pydantic request/response (inference+embed+drift, relearn, dataset)
  services/    registry, seed, storage (object store), graph_cache,
               inference (predict+embed), relearn (+materialize), datasets, eval (general+drift)
  routers/     meta, inference (+/embed), relearn, datasets, eval (/eval)
  configs/     data_build.yaml + graph_based/hybrid_graph_lm/sequential.yaml — bundled seed
               configs, snapshotted to the DB at register (image bakes no repo configs/ tree)
  seeds/       models.json, datasets.json — DB seed on first boot
  celery_app.py, tasks.py   — Celery app + async tasks (ingest_dataset, merge_datasets, run_relearn)
  main.py      app wiring (lifespan init_db + seed, CORS, routers)
  Dockerfile, docker-compose.yml, docker-compose.gpu.yml, requirements.txt, pyproject.toml
  export_openapi.py → openapi.json
```

---

## Run — dev (Docker, recommended)

The stack (Postgres + Redis + MinIO + API + worker-gpu + worker-cpu + Adminer) comes up with
one command. API source is bind-mounted, so editing `API/*.py` + `docker compose restart api
worker-gpu worker-cpu` applies without a rebuild (only `src/gnn_vuln` changes need `build`,
because that goes into the installed package).

```bash
docker compose -f API/docker-compose.yml up -d --build
curl http://localhost:8000/health
```

UIs: API docs `:8000/docs` · MinIO console `:9001` (minioadmin/minioadmin) ·
Adminer `:8080` (server `db`, user/pass/db `vuln`/`vuln`/`vulndb`).

### GPU (mimic prod)

```bash
docker compose -f API/docker-compose.yml -f API/docker-compose.gpu.yml up -d --build
```

The overlay builds the image with CUDA torch (cu124), reserves the host GPU for api +
worker-gpu (worker-cpu stays `API_DEVICE=cpu`, no GPU), and sets `API_DEVICE=cuda`. Needs the
NVIDIA Container Toolkit (`docker info` shows the `nvidia` runtime).

### Run the API standalone (its own venv)

The API is its own uv project depending on `gnn-vuln` as a package:

```bash
cd API && uv sync          # builds/installs gnn-vuln from .. + web deps into API/.venv
uv run uvicorn API.main:app --host 0.0.0.0 --port 8000
```

Env: `DATABASE_URL`, `JOERN_CLI`, `GNN_VULN_ROOT` (data/checkpoint root), `API_DEVICE`
(`cpu`|`cuda`), `API_CORS_ORIGINS`, `API_MAX_NODES`, `API_MODEL_CACHE_SIZE` (loaded predictors
kept in RAM/GPU, LRU, default 3), `STORAGE_BACKEND` (`fs`|`s3`) +
`S3_*` + `S3_BUCKET_GRAPHS`/`S3_BUCKET_DATASETS`/`S3_BUCKET_CHECKPOINTS`, `CELERY_BROKER_URL`,
`CELERY_RESULT_BACKEND`.

Seed models: `API/seeds/models.json` points each model id at `checkpoints/<id>.pt` (local
cache) + `s3://checkpoints/<id>.pt` (`storage_uri`); the config travels as an immutable DB
snapshot (bundled defaults in `API/configs/`), so no repo `configs/` tree is needed. Provide
each `.pt` either on local disk (`/app/checkpoints`) **or** in the `checkpoints` bucket —
inference materializes it on demand. Models trained via `/relearn` register themselves.

---

## Storage & data model — metadata in the DB, blobs in object storage

Best practice for ML services: keep large binaries **out** of the relational DB (5–20× pricier
per byte for blobs, 1 GB field cap, slow backups). The DB holds metadata + a **pointer**; the
blob lives in object storage.

- **Built CPG graphs** → blob in the `graphs` bucket (key `sha256(code).fmt`); `graph_cache`
  keeps only `code_hash → object_key`. Repeat requests skip Joern. Backend pluggable:
  `STORAGE_BACKEND=fs` (local folder, zero deps) | `s3` (MinIO/S3, same boto3 code).
- **Dataset `.pt` bundles** → `datasets` bucket; `datasets.params.storage_uri` records the
  pointer. Pulled to local disk at relearn time (SageMaker File-Mode pattern, cached by id).
- **Model checkpoints** → `checkpoints` bucket; `models.storage_uri` is the source of truth,
  `models.checkpoint` is the local cache path. A relearned model uploads its `.pt` after
  training; inference materializes it back to disk on demand — so a worker on another node
  can load a model it never trained (the container FS is ephemeral; only the buckets are
  durable + shared). Seed models carry `storage_uri` too, and their **config loads from the
  immutable DB snapshot** (not a repo file) — so the deployed image bakes no `configs/` tree.
- **Per-model artifacts** (EWC importance, eval metrics, realized split) → `model_artifacts`
  keeps a pointer (`model_id, kind, storage_uri`); the blob lives in the `checkpoints` bucket
  (`<id>.ewc_importance.pt`, `<id>.metrics_summary.json`, `<id>.train_split.json`). One row
  per `(model_id, kind)`.
- **Inference results** (prediction + confidence + suspicious lines + **pre-head embedding**)
  → `inference_results`. The embedding is small + structured, so it stays in the DB (pgvector
  ANN index at search time). Query `GET /inference/history`.

### Immutable, content-addressed versioning

Configs, datasets and models are **append-only — never overwritten**:

- **Configs** → `configs` table, id = `{name}@{sha256(content)[:10]}`. **One config per entity**
  covers the whole flow (data + model + train) — there is no kind taxonomy. Same content reuses
  the id (dedup); an edit mints a **new** id. So any reference always resolves to the exact
  content it was trained with — reproducibility without a separate snapshot.
- **One config, used by all flows** — a **model** stores its full config (`model.config_id`):
  `/inference` loads it, and `/relearn` uses it as the base, overriding the data section with the
  task-B dataset. A **dataset** stores its build config (`dataset.data_config_id` = source,
  filters, split, featurization), reused by `/datasets` (`data_config_id`) and by relearn to match
  the cached `.pt`. `/train` instantiates a model from any `config_id` + `dataset_ids`. The
  endpoints pull the section they need from a config (`config_section`), no kind gate.
- **Datasets / models** get a fresh id per ingest/run. A dataset links its `data_config_id`;
  a model links `config_id` + `base_model_id` + `method`.

Why no graph DB (Neo4j)? We only store/fetch the whole CPG by hash and render client-side — no
server-side traversal — so object storage + Postgres suffice.

### ERD

```mermaid
erDiagram
    datasets ||--o{ models : "trained on"
    configs ||--o{ models : "config_id"
    configs ||--o{ datasets : "data_config_id"
    models ||--o{ models : "base_model_id (relearn)"
    models ||--o{ relearn_jobs : "base / result"
    models ||--o{ inference_results : "model_id"
    graph_cache ||--o{ inference_results : "code_hash"
    graph_cache }o--|| object_store : "object_key -> graphs bucket"
    raw_datasets ||--o{ datasets : "raw_id (one raw, many builds)"
    raw_datasets }o--|| object_store : "storage_uri -> datasets bucket (raw rows json)"
    datasets }o--|| object_store : "storage_uri -> datasets bucket (.pt bundle)"
    datasets ||--o{ datasets : "source_dataset_ids (merge)"
    models }o--|| object_store : "storage_uri -> checkpoints bucket"

    configs {
      string id PK "name@hash"
      string name
      text content "canonical YAML"
      string content_hash
      datetime created_at
    }
    raw_datasets {
      string id PK "raw_ + sha256(content)[:12]"
      text storage_uri "-> datasets bucket, raw rows json"
      int num_rows
      int size_bytes
      string content_hash
      datetime created_at
    }
    datasets {
      string id PK
      string source
      string mode
      int num_classes
      string data_config_id FK
      string raw_id FK "-> raw_datasets"
      json source_dataset_ids "merge provenance -> datasets.id"
      json params "max_nodes, filters, storage_uri, ..."
      datetime created_at
    }
    models {
      string id PK
      string arch
      string config_id FK
      string checkpoint "local cache path"
      string storage_uri "-> checkpoints bucket"
      string dataset_id FK
      json dataset_ids "CUMULATIVE lineage: ancestors + this job (ER replay pool)"
      string val_dataset_id FK "role dataset (val)"
      string test_dataset_id FK "role dataset (test/benchmark)"
      int num_classes
      json class_names
      string base_model_id FK
      string method
    }
    relearn_jobs {
      string job_id PK
      string status
      string method
      json dataset_ids
      string base_model_id FK
      string val_dataset_id FK "role dataset (val)"
      string test_dataset_id FK "role dataset (test/benchmark)"
      string result_model_id FK
      string log_path
    }
    dataset_jobs {
      string job_id PK
      string status
      string dataset_id
      json data_config
      string log_path
    }
    graph_cache {
      string code_hash PK
      text object_key "-> graphs bucket"
    }
    inference_results {
      int id PK
      string model_id FK
      string code_hash FK
      string prediction
      float confidence
      json suspicious_lines
      json cls_embedding "pre-head vector"
      int cls_embedding_dim
    }
    object_store {
      string bucket
      string key
      bytes blob "CPG / dataset .pt (MinIO/S3 or fs)"
    }
```

### Relearn methods

| `method`   | start weights   | EWC penalty | replay buffer       |
| ---------- | --------------- | ----------- | ------------------- |
| `finetune` | base checkpoint | off         | no                  |
| `EWC`      | base checkpoint | weight 1000 | no                  |
| `ER`       | base checkpoint | off         | base-dataset buffer |
| `EWC-ER`   | base checkpoint | weight 1000 | base-dataset buffer |
| `retrain`  | random init     | off         | no                  |

`base_model_id` is required for everything except `retrain` (it supplies the architecture,
start weights, and the task-A dataset for the replay buffer + EWC importance). CIL (task-B adds
new classes) is automatic — `num_classes` grows and the base head loads expandably.

---

## The `gnn_vuln` library

The model code is an installable package (`pip install gnn-vuln`). The service uses only its
public surface — string in, dict out; the Joern CPG step is hidden inside the call:

```python
from gnn_vuln.inference import VulnPredictor

predictor = VulnPredictor.from_checkpoint("checkpoints/<run>/best_model.pt",
                                          "configs/<arch>/config.yaml", device="cuda")
result = predictor.predict_code("void f(char*s){char b[8];strcpy(b,s);}",
                                joern_cli="/opt/joern/joern-cli", top_k_lines=5)
# result: {prediction, confidence, class_probabilities, suspicious_lines, cls_embedding, ...}
```

**Full library reference** (every exported function/class, import paths, inputs, outputs,
result schema, the `python -m` data/training CLIs) → **[`src/gnn_vuln/README.md`](../src/gnn_vuln/README.md)**.
