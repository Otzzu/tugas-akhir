"""Celery tasks — async orchestration for long-running jobs.

Tasks ORCHESTRATE only: job state (DB), storage (MinIO/S3), registration. The
actual ML work is the gnn_vuln LIBRARY, invoked via its module entrypoints
(`python -m gnn_vuln.data.prepare`, `... .build_pt`). The library knows nothing
about Celery / the API.

Run the worker:  celery -A API.celery_app worker --loglevel=info --concurrency=1
"""
from __future__ import annotations

import copy
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import yaml

from API.celery_app import celery_app
from API.core.config import settings
from API.core.database import SessionLocal
from API.models.tables import DatasetJobRecord
from API.services import registry, storage

ROOT = settings.ROOT
# Self-contained DATA-BUILD config (featurization + neutral data defaults only —
# NO training hyperparameters). The per-request data overrides overlay on top.
_DATA_BUILD_TEMPLATE = Path(__file__).resolve().parent / "configs" / "data_build.yaml"


def _set_status(job_id: str, **fields) -> None:
    with SessionLocal() as s:
        j = s.get(DatasetJobRecord, job_id)
        if j is None:
            return
        for k, v in fields.items():
            setattr(j, k, v)
        s.commit()


def _run(cmd: list[str], log) -> None:
    log.write(f"\n$ {' '.join(cmd)}\n"); log.flush()
    subprocess.run(cmd, check=True, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)


def _log_tail(path, n: int = 12, maxlen: int = 600) -> str:
    """Last meaningful lines of a job log — folded into the job's `message` so the real
    subprocess error reaches the user via GET /datasets/jobs/{id}, no log-spelunking."""
    try:
        from pathlib import Path as _P
        lines = [ln for ln in _P(path).read_text(errors="replace").splitlines() if ln.strip()]
        return "\n".join(lines[-n:])[-maxlen:]
    except Exception:
        return ""


@celery_app.task(name="run_relearn", bind=True)
def run_relearn(self, job_id: str, train_cfg: str, importance_cfg: str | None, meta: dict) -> dict:
    """Execute a relearn training job (built by relearn.submit_relearn) off the web server."""
    from API.services import relearn
    relearn.execute_relearn(job_id, train_cfg, importance_cfg, meta)
    return {"job_id": job_id, "status": "done"}


@celery_app.task(name="merge_datasets", bind=True)
def merge_datasets(self, job_id: str) -> dict:
    """merge >=2 registered datasets -> unified .pt -> tar -> object storage -> registered dataset."""
    from API.services.relearn import materialize_dataset

    with SessionLocal() as s:
        rec = s.get(DatasetJobRecord, job_id)
        if rec is None:
            return {"job_id": job_id, "status": "failed", "message": "job row missing"}
        dc = dict(rec.data_config or {})
        log_path = rec.log_path

    job_dir = Path(dc["job_dir"])
    dataset_id = dc["dataset_id"]
    name = dc["name"]
    dataset_ids = dc["dataset_ids"]
    dedup = dc.get("dedup", True)
    out_source = registry._config_slug(dataset_id)  # stable slug for the merged .pt name

    data_root = settings.ROOT / "data"
    raw_dir = data_root / "raw"
    processed_dir = data_root / "processed"

    lp = log_path or (job_dir / "merge.log")
    log = open(lp, "w", encoding="utf-8")
    try:
        _set_status(job_id, status="running")

        # 1) stage each source dataset's .pt + vocab onto local disk, collect sources
        src_sources = []
        for did in dataset_ids:
            materialize_dataset(did)
            src_sources.append(registry.get_dataset(did)["source"])
        mode = registry.get_dataset(dataset_ids[0]).get("mode", "multiclass")
        max_nodes = dc.get("max_nodes", 2500)

        # 2) data-build config (same template as ingest; out source slug, root dirs)
        cfg = copy.deepcopy(yaml.safe_load(_DATA_BUILD_TEMPLATE.read_text()))
        d = cfg.setdefault("data", {})
        d.update({
            "source": out_source, "mode": mode,
            "raw_dir": str(raw_dir), "processed_dir": str(processed_dir),
            "max_nodes": max_nodes,
        })
        cfg.setdefault("train", {})["device"] = settings.DEVICE
        cfg_path = job_dir / "data_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        # 3) merge the staged .pt's via the library entrypoint
        merge = ["python", "-m", "gnn_vuln.data.merge", "--config", str(cfg_path),
                 "--sources", *src_sources, "--out-source", out_source]
        if dedup:
            merge.append("--dedup")
        _run(merge, log)

        # num_classes from the unified vocab the merge wrote
        vocab_path = raw_dir / out_source / "cwe_vocab.json"
        num_classes = len(json.loads(vocab_path.read_text())) if vocab_path.exists() else 2

        # 4) register the data-build config (content-addressed, immutable)
        cfg.setdefault("model", {})["num_classes"] = num_classes
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        data_config_id = registry.upsert_config(out_source, cfg)

        # 5) bundle the DATA ARTIFACTS only (this merge's .pt + label vocab) -> object storage.
        # processed_dir is the SHARED data root holding EVERY dataset's .pt (incl multi-GB
        # ones); tar ONLY the .pt this merge wrote — never the whole dir (gzip of GBs hangs).
        pts = sorted(processed_dir.glob(f"lm_dataset_{out_source}_*.pt"))
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for pt in pts:
                tar.add(pt, arcname=f"processed/{pt.name}")
            if vocab_path.exists():
                tar.add(vocab_path, arcname="cwe_vocab.json")
        key = f"{dataset_id}.tar.gz"
        uri = storage.put_bytes(settings.S3_BUCKET_DATASETS, key, buf.getvalue())
        log.write(f"\nuploaded merged dataset bundle -> {uri}\n"); log.flush()

        # 6) register the dataset. A merge has no raw rows of its own — provenance
        # lives in source_dataset_ids; each source dataset carries its own raw_id.
        registry.register_dataset(dataset_id, {
            "label": name, "source": out_source, "mode": mode, "num_classes": num_classes,
            "data_config_id": data_config_id,
            "storage_uri": uri, "storage": "inmemory",
            "source_dataset_ids": list(dataset_ids),
            "max_nodes": max_nodes, "top_cwe": dc.get("top_cwe", 0),
            "max_per_class": dc.get("max_per_class", 0), "resample_seed": dc.get("resample_seed", 42),
            "val_fraction": dc.get("val_fraction", 0.1), "test_fraction": dc.get("test_fraction", 0.0),
        })

        _set_status(job_id, status="done", dataset_id=dataset_id,
                    message=f"merged {len(dataset_ids)} datasets, {num_classes} classes, bundle at {uri}")
        return {"job_id": job_id, "status": "done", "dataset_id": dataset_id}

    except subprocess.CalledProcessError as e:
        log.flush()
        tail = _log_tail(lp)
        msg = f"merge step failed (exit {e.returncode})"
        _set_status(job_id, status="failed", message=f"{msg}:\n{tail}" if tail else f"{msg}; see log")
        return {"job_id": job_id, "status": "failed"}
    except Exception as e:  # noqa: BLE001
        _set_status(job_id, status="failed", message=f"{type(e).__name__}: {e}")
        return {"job_id": job_id, "status": "failed"}
    finally:
        log.close()


@celery_app.task(name="ingest_dataset", bind=True)
def ingest_dataset(self, job_id: str) -> dict:
    """raw rows -> Joern CPG -> .pt -> tar -> object storage -> registered dataset."""
    with SessionLocal() as s:
        rec = s.get(DatasetJobRecord, job_id)
        if rec is None:
            return {"job_id": job_id, "status": "failed", "message": "job row missing"}
        dc = dict(rec.data_config or {})
        log_path = rec.log_path
        n_rows = int(rec.num_rows or 0)

    job_dir = Path(dc["job_dir"])
    raw_dir = job_dir / "raw"
    processed_dir = job_dir / "processed"
    input_path = dc["input_path"]
    dataset_id = dc["dataset_id"]
    name = dc["name"]
    mode = dc.get("mode", "multiclass")
    source = registry._config_slug(dataset_id)  # stable slug for the .pt name

    lp = log_path or (job_dir / "ingest.log")
    log = open(lp, "w", encoding="utf-8")
    try:
        _set_status(job_id, status="running")

        # 1) raw -> CPG (+ cwe_vocab.json) via the library entrypoint. Format `api` = bigvul
        # schema plus per-row flaw_lines annotations; rows without one fall back to the diff.
        prep = ["python", "-m", "gnn_vuln.data.prepare",
                "--input", str(input_path), "--format", "api",
                "--out-dir", str(raw_dir), "--workers", "4",
                "--joern-cli", str(settings.JOERN_CLI)]
        if mode == "binary":
            prep.append("--binary")
        if dc.get("top_cwe", 0):
            prep += ["--top-cwe", str(dc["top_cwe"])]
        if dc.get("max_per_class", 0):
            prep += ["--sample-per-class", str(dc["max_per_class"])]
        _run(prep, log)

        # prepare nests CPGs under <raw_dir>/<format> ("bigvul"); the .pt builder
        # reads <raw_dir>/<source>. Rename so both agree on the dataset slug.
        prep_dir = raw_dir / "bigvul"
        src_dir = raw_dir / source
        if prep_dir.exists() and prep_dir != src_dir:
            if src_dir.exists():
                shutil.rmtree(src_dir)
            prep_dir.rename(src_dir)

        # num_classes from the vocab the prepare step wrote
        vocab_path = src_dir / "cwe_vocab.json"
        if mode == "binary":
            num_classes = 2
        elif vocab_path.exists():
            num_classes = len(json.loads(vocab_path.read_text()))
        else:
            num_classes = 2

        # 2) data-build config: the self-contained featurization template with the
        # per-request data overrides overlaid (job dirs, mode, sampling). No
        # training config is involved, so nothing training-specific can leak in.
        cfg = copy.deepcopy(yaml.safe_load(_DATA_BUILD_TEMPLATE.read_text()))
        d = cfg.setdefault("data", {})
        d.update({
            "source": source, "mode": mode,
            "raw_dir": str(raw_dir), "processed_dir": str(processed_dir),
            "max_nodes": dc.get("max_nodes", d.get("max_nodes", 2500)),
            "top_cwe": dc.get("top_cwe", 0),
            "max_per_class": dc.get("max_per_class", 0),
            "resample_seed": dc.get("resample_seed", 42),
        })
        cfg.setdefault("model", {})["num_classes"] = num_classes
        cfg.setdefault("train", {})["device"] = settings.DEVICE
        cfg_path = job_dir / "data_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        # register the data-build config as an immutable, content-addressed row so the
        # dataset references it by id. Editing it later mints a new id.
        data_config_id = registry.upsert_config(source, cfg)

        # 3) CPG -> .pt via the library entrypoint
        _run(["python", "-m", "gnn_vuln.data.build_pt", "--config", str(cfg_path), "--split", "train"], log)

        # 4) bundle the DATA ARTIFACTS only (.pt + label vocab) -> object storage.
        # The config is NOT bundled: it lives in the DB (datasets.params), which is
        # the single source of truth relearn reads. .pt = artifact, config = config.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            if processed_dir.exists():
                tar.add(processed_dir, arcname="processed")
            if vocab_path.exists():
                tar.add(vocab_path, arcname="cwe_vocab.json")
        key = f"{dataset_id}.tar.gz"
        uri = storage.put_bytes(settings.S3_BUCKET_DATASETS, key, buf.getvalue())
        log.write(f"\nuploaded dataset bundle -> {uri}\n"); log.flush()

        # 4b) persist the EXACT raw rows the .pt was built from (canonical upload
        # JSON, row ids materialized) and register them as a first-class raw_datasets
        # row. Content-addressed id ("raw_" + sha256[:12]) → identical rows re-uploaded
        # dedupe to the same row + object, and ONE raw can back MANY datasets (different
        # build configs). DB holds metadata + pointer only; the blob lives in S3.
        raw_id = None
        raw_path = dc.get("raw_path")
        if raw_path and Path(raw_path).exists():
            import hashlib
            raw_bytes = Path(raw_path).read_bytes()
            digest = hashlib.sha256(raw_bytes).hexdigest()
            raw_id = f"raw_{digest[:12]}"
            raw_uri = storage.put_bytes(settings.S3_BUCKET_DATASETS, f"{raw_id}.json", raw_bytes)
            registry.register_raw(raw_id, {
                "storage_uri": raw_uri, "num_rows": n_rows,
                "size_bytes": len(raw_bytes), "content_hash": digest,
            })
            log.write(f"raw rows registered as {raw_id} -> {raw_uri}\n"); log.flush()

        # 5) register the dataset (params hold the frozen data-config + storage pointers)
        registry.register_dataset(dataset_id, {
            "label": name, "source": source, "mode": mode, "num_classes": num_classes,
            "data_config_id": data_config_id,
            "storage_uri": uri, "storage": "inmemory", "raw_id": raw_id,
            "max_nodes": dc.get("max_nodes", 2500), "top_cwe": dc.get("top_cwe", 0),
            "max_per_class": dc.get("max_per_class", 0), "resample_seed": dc.get("resample_seed", 42),
            "val_fraction": dc.get("val_fraction", 0.1), "test_fraction": dc.get("test_fraction", 0.0),
        })

        _set_status(job_id, status="done", dataset_id=dataset_id, raw_id=raw_id,
                    message=f"{num_classes} classes, bundle at {uri}")
        return {"job_id": job_id, "status": "done", "dataset_id": dataset_id}

    except subprocess.CalledProcessError as e:
        log.flush()
        tail = _log_tail(lp)
        msg = f"pipeline step failed (exit {e.returncode})"
        _set_status(job_id, status="failed", message=f"{msg}:\n{tail}" if tail else f"{msg}; see log")
        return {"job_id": job_id, "status": "failed"}
    except Exception as e:  # noqa: BLE001
        _set_status(job_id, status="failed", message=f"{type(e).__name__}: {e}")
        return {"job_id": job_id, "status": "failed"}
    finally:
        log.close()
