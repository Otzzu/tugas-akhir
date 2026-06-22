"""
Relearn service: continual-learning training jobs over the existing pipeline.

Methods: finetune (base weights, no protection), EWC (EWC-DR penalty),
ER (experience replay), EWC-ER (both), retrain (fresh, no base).

For the 4 non-retrain methods base_model_id is required — its config (arch), checkpoint
(start weights) and training dataset (replay buffer + EWC importance source) come from the
registry. Multiple dataset_ids are joined before training.

A job writes a config under API/jobs/<job_id>/ and runs in a background thread:
  1. EWC importance pass on the base dataset (EWC / EWC-ER only, if cache missing)
  2. training (python -m gnn_vuln.train)
then registers the resulting checkpoint as a new model id.
"""
from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import tarfile
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from sqlalchemy import select

from API.core.config import settings
from API.core.database import SessionLocal
from API.models.tables import JobRecord
from API.services import registry, storage

ROOT = settings.ROOT
JOBS_DIR = settings.JOBS_DIR
DATA_ROOT = ROOT / "data"            # local materialization target (the trainer's root)
ENV: dict[str, str] = {}             # gnn_vuln is an installed package — no path injection needed
TRAIN_MODULE = "gnn_vuln.train"      # library trainer (installed package, single source of truth)
EVAL_MODULE = "gnn_vuln.evaluate"    # library evaluator — returns metrics_summary (cls + localization)
COMPUTE_IMPORTANCE_ON_FINISH = True  # eagerly compute EWC importance for the newly trained model
EVALUATE_ON_FINISH = True            # run the library evaluator on the new model (task-B test split)

_REQUIRES_BASE = {"ER", "EWC", "EWC-ER", "finetune"}
_JOB_FIELDS = ("status", "method", "dataset_ids", "base_model_id",
               "config_path", "log_path", "result_model_id", "message", "metrics")


# ── job persistence (DB) ────────────────────────────────────────────────────
def _save_job(job: dict) -> None:
    with SessionLocal() as s:
        j = s.get(JobRecord, job["job_id"]) or JobRecord(job_id=job["job_id"])
        for k in _JOB_FIELDS:
            if k in job:
                setattr(j, k, job[k])
        s.merge(j)
        s.commit()


def get_job(job_id: str) -> dict | None:
    with SessionLocal() as s:
        j = s.get(JobRecord, job_id)
        return j.to_dict() if j else None


def list_jobs() -> list[dict]:
    with SessionLocal() as s:
        return [j.to_dict() for j in s.scalars(select(JobRecord)).all()]


# ── dataset materialization (object store -> local disk) ────────────────────
def materialize_dataset(dataset_id: str) -> Path:
    """Stage a dataset bundle from object storage onto local disk (SageMaker File-Mode
    pattern: download once, then train from local). Cached by dataset_id — datasets are
    immutable, so the local copy never goes stale. Lays out the files exactly where the
    trainer looks: <DATA_ROOT>/raw/<source>/cwe_vocab.json + <DATA_ROOT>/processed/<pt>."""
    ds = registry.get_dataset(dataset_id)
    source = ds["source"]
    marker = DATA_ROOT / ".materialized" / dataset_id
    local_vocab = DATA_ROOT / "raw" / source / "cwe_vocab.json"
    # already staged (cached from a prior run) OR provided locally (seed/base datasets
    # like megavul that the user unzips into data/ by hand — no object-store bundle).
    if marker.exists() or local_vocab.exists():
        return DATA_ROOT
    key = f"{dataset_id}.tar.gz"
    blob = storage.get_bytes(settings.S3_BUCKET_DATASETS, key)
    if blob is None:
        raise FileNotFoundError(
            f"dataset '{dataset_id}' is neither in object storage "
            f"({settings.S3_BUCKET_DATASETS}/{key}) nor local at {local_vocab}. "
            f"Ingest it via POST /datasets, or unzip the prebuilt dataset into {DATA_ROOT}.")
    raw_dir = DATA_ROOT / "raw" / source
    proc_dir = DATA_ROOT / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            data = tar.extractfile(m).read()
            if Path(m.name).name == "cwe_vocab.json":
                (raw_dir / "cwe_vocab.json").write_bytes(data)
            elif m.name.startswith("processed/"):
                (proc_dir / Path(m.name).name).write_bytes(data)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(datetime.utcnow().isoformat())
    return DATA_ROOT


def _dataset_data_config(dataset_id: str) -> dict:
    """The frozen data-build config that produced this dataset's .pt (via data_config_id).
    Used so the relearn run reuses the EXACT featurization — same source/embedder/filters
    → same processed .pt name → the materialized cache loads instead of rebuilding."""
    ds = registry.get_dataset(dataset_id)
    cid = ds.get("data_config_id")
    if not cid:
        return {}
    try:
        return yaml.safe_load(registry.get_config(cid)["content"]) or {}
    except KeyError:
        return {}


# ── dataset join ────────────────────────────────────────────────────────────
def _join_datasets(dataset_ids: list[str]) -> tuple[str, dict]:
    """Single id → its source. Multiple → merge at the PROCESSED .pt level via the
    gnn_vuln.data.merge CLI (concatenate finished .pt + unify vocab; no raw CPG, no
    rebuild). Works for datasets ingested via /datasets whose bundle has only the .pt."""
    if len(dataset_ids) == 1:
        d = registry.get_dataset(dataset_ids[0])
        return d["source"], d

    metas = [registry.get_dataset(i) for i in dataset_ids]
    joined = "join_" + "_".join(dataset_ids)

    sources: list[str] = []
    for did, m in zip(dataset_ids, metas):
        materialize_dataset(did)          # stage each dataset's .pt + vocab locally
        sources.append(m["source"])

    # minimal data-build config for the merge — raw/processed pinned to DATA_ROOT,
    # embedder pulled from the first dataset's frozen build config so the merge reads
    # and writes .pt with matching featurization.
    cfg = copy.deepcopy(yaml.safe_load((ROOT / "API" / "configs" / "data_build.yaml").read_text()))
    fmodel = _dataset_data_config(dataset_ids[0]).get("model", {})
    cfg.setdefault("model", {}).update(
        {k: fmodel[k] for k in
         ("pretrained_lm", "func_lm", "func_lm_source", "add_func_tokens", "func_max_length")
         if k in fmodel})
    cfg["data"] = {**cfg.get("data", {}), "raw_dir": str(DATA_ROOT / "raw"),
                   "processed_dir": str(DATA_ROOT / "processed"), "source": joined}
    cfg_path = DATA_ROOT / "processed" / f"{joined}_merge.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    subprocess.run(
        ["python", "-m", "gnn_vuln.data.merge", "--config", str(cfg_path),
         "--sources", *sources, "--out-source", joined, "--dedup"],
        check=True, cwd=str(ROOT), env={**os.environ, **ENV})

    vocab = json.loads((DATA_ROOT / "raw" / joined / "cwe_vocab.json").read_text())
    base = copy.deepcopy(metas[0])
    base["source"] = joined
    base["num_classes"] = len(vocab)
    return joined, base


# ── config generation ───────────────────────────────────────────────────────
def build_config(method: str, dataset_ids: list[str], base_model_id: str | None,
                 epochs: int | None, job_dir: Path) -> tuple[Path, Path | None, dict]:
    source, data_block = _join_datasets(dataset_ids)

    if method in _REQUIRES_BASE:
        if not base_model_id:
            raise ValueError(f"method '{method}' requires base_model_id")
        base = registry.get_model(base_model_id)
        base_cfg = yaml.safe_load(registry.config_text(base))
        from API.services.inference import _materialize_checkpoint
        base_ckpt = _materialize_checkpoint(base)   # local cache; pulls from MinIO if absent
        base_ds = registry.get_dataset(base["dataset_id"])
        materialize_dataset(base["dataset_id"])   # task-A data on disk (replay buffer + EWC importance)
    else:  # retrain — fresh weights; optional template for arch
        if base_model_id:
            base = registry.get_model(base_model_id)
            base_cfg = yaml.safe_load(registry.config_text(base))
        else:
            base_cfg = yaml.safe_load((settings.API_DIR / "configs" / "graph_based.yaml").read_text())
        base_ckpt = None
        base_ds = None

    # stage datasets to local disk + reuse each dataset's FROZEN featurization config so
    # the cached processed .pt is found (identical source/embedder/filters → identical name,
    # no rebuild from raw needed).
    for did in dataset_ids:
        materialize_dataset(did)
    frozen = _dataset_data_config(dataset_ids[0])
    fdata, fmodel = frozen.get("data", {}), frozen.get("model", {})

    cfg = copy.deepcopy(base_cfg)
    cfg["data"] = {**cfg.get("data", {}), **{
        "source": source, "mode": fdata.get("mode", data_block.get("mode", "multiclass")),
        "raw_dir": str(DATA_ROOT / "raw"), "processed_dir": str(DATA_ROOT / "processed"),
        "max_nodes": fdata.get("max_nodes", data_block.get("max_nodes", 2500)),
        "top_cwe": fdata.get("top_cwe", data_block.get("top_cwe", 0)),
        "filter_top25_dangerous": fdata.get("filter_top25_dangerous",
                                            data_block.get("filter_top25_dangerous", False)),
        "filter_owasp": fdata.get("filter_owasp", False),
        "max_per_class": fdata.get("max_per_class", data_block.get("max_per_class", 0)),
        "resample_seed": fdata.get("resample_seed", data_block.get("resample_seed", 42)),
        "storage": fdata.get("storage", data_block.get("storage", "inmemory")),
        "ds_name_suffix": data_block.get("ds_name_suffix", ""),
    }}
    # reuse the dataset's embedder (featurization) so the .pt name matches the cached build
    for k in ("pretrained_lm", "func_lm", "add_func_tokens", "func_max_length", "func_lm_source"):
        if k in fmodel:
            cfg["model"][k] = fmodel[k]
    cfg["model"]["num_classes"] = data_block.get("num_classes", cfg["model"].get("num_classes", 26))
    # the installed wheel's default checkpoint/results/log dirs resolve to a bogus
    # site-packages path; pin them to the app root so the trainer writes where we read.
    t = cfg.setdefault("train", {})
    t["checkpoint_dir"] = str(ROOT / "checkpoints")
    t["results_dir"] = str(ROOT / "results")
    t["log_dir"] = str(ROOT / "logs")
    if epochs:
        t["epochs"] = epochs

    ckpt_dir = ROOT / "checkpoints" / f"api_base_{base_model_id}" if base_ckpt else None
    importance_cfg_path = None
    cfg.pop("ewc", None)
    cfg.pop("replay", None)

    if method != "retrain":
        cache = str(ckpt_dir / "ewc_importance.pt") if ckpt_dir else ""
        if base_ckpt:
            import shutil
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            staged = ckpt_dir / "best_model.pt"
            if not staged.exists():
                shutil.copy2(base_ckpt, staged)
            src_ckpt = str(staged)
        else:
            src_ckpt = ""
        weight = 1000.0 if method in ("EWC", "EWC-ER") else 0.0
        cfg["ewc"] = {"enabled": True, "weight": weight, "scope": "all",
                      "source_checkpoint": src_ckpt, "importance_cache": cache, "n_batches": 0}
        if method in ("ER", "EWC-ER"):
            cfg["replay"] = {"enabled": True, "source": base_ds["source"],
                             "ds_name_suffix": base_ds.get("ds_name_suffix", ""),
                             "buffer_per_class": 50, "weight": 1.0, "buffer_seed": 42,
                             "filter_top25_dangerous": base_ds.get("filter_top25_dangerous", False),
                             "max_per_class": base_ds.get("max_per_class", 0),
                             "top_cwe": base_ds.get("top_cwe", 0),
                             "resample_seed": base_ds.get("resample_seed", 42)}
        if method in ("EWC", "EWC-ER") and not Path(cache).exists():
            # durable read-back: a prior relearn may have stored importance for this base model
            art = registry.get_artifact(base_model_id, "ewc_importance")
            blob = (storage.get_bytes(settings.S3_BUCKET_CHECKPOINTS,
                                      f"{base_model_id}.ewc_importance.pt") if art else None)
            if blob is not None:
                Path(cache).parent.mkdir(parents=True, exist_ok=True)
                Path(cache).write_bytes(blob)   # cfg["ewc"]["importance_cache"] already points here
            else:  # not stored → compute before relearn (current behavior, the protection)
                imp = copy.deepcopy(cfg)
                imp["data"] = {**imp["data"], **{
                    "source": base_ds["source"],
                    "filter_top25_dangerous": base_ds.get("filter_top25_dangerous", False),
                    "max_per_class": base_ds.get("max_per_class", 0), "top_cwe": base_ds.get("top_cwe", 0),
                    "ds_name_suffix": base_ds.get("ds_name_suffix", "")}}
                imp["model"]["num_classes"] = base_ds.get("num_classes", cfg["model"]["num_classes"])
                imp["ewc"] = {**cfg["ewc"], "weight": 1000.0, "compute_only": True}
                imp.pop("replay", None)
                importance_cfg_path = job_dir / "importance.yaml"
                importance_cfg_path.write_text(yaml.safe_dump(imp, sort_keys=False))

    train_cfg_path = job_dir / "train.yaml"
    train_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    meta = {"source": source, "num_classes": cfg["model"]["num_classes"],
            "arch": cfg["model"]["architecture"],
            "base_model_id": base_model_id,
            "importance_cache": str(cache) if method != "retrain" else None}
    return train_cfg_path, importance_cfg_path, meta


# ── durable EWC-importance artifact ─────────────────────────────────────────
def _store_importance(model_id: str, cache_path: str | Path) -> None:
    """Upload a computed importance cache to object storage + register it for model_id."""
    data = Path(cache_path).read_bytes()
    uri = storage.put_bytes(settings.S3_BUCKET_CHECKPOINTS, f"{model_id}.ewc_importance.pt", data)
    registry.add_artifact(model_id, "ewc_importance", uri)


def _compute_and_store_importance(model_id: str, train_cfg_path: Path, checkpoint_path: Path,
                                  out_cache_path: Path, log_file) -> None:
    """Best-effort: compute EWC importance for a freshly trained model so a future relearn
    using it as base is instant. Never raises — eager importance is optional."""
    try:
        cfg = copy.deepcopy(yaml.safe_load(Path(train_cfg_path).read_text()))
        cfg["ewc"] = {"enabled": True, "weight": 1000.0, "scope": "all",
                      "source_checkpoint": str(checkpoint_path),
                      "importance_cache": str(out_cache_path), "n_batches": 0, "compute_only": True}
        cfg.pop("replay", None)
        out_cache_path.parent.mkdir(parents=True, exist_ok=True)
        imp_cfg = Path(train_cfg_path).parent / "importance_finish.yaml"
        imp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
        log_file.write(f"== EWC importance (eager, finish): {imp_cfg} ==\n"); log_file.flush()
        subprocess.run(["python", "-m", TRAIN_MODULE, "--config", str(imp_cfg)],
                       check=True, cwd=str(ROOT), env={**os.environ, **ENV},
                       stdout=log_file, stderr=subprocess.STDOUT)
        _store_importance(model_id, out_cache_path)
    except Exception as e:  # noqa: BLE001
        log_file.write(f"WARN eager importance failed for {model_id}: {type(e).__name__}: {e}\n")
        log_file.flush()


# ── evaluation metrics (task-B test split, via the library evaluator) ────────
def _compact_metrics(summary: dict) -> dict:
    """Headline numbers from metrics_summary for the job record (full blob stays in storage)."""
    fl = summary.get("function_level", {}) or {}
    loc = summary.get("localization", {}) or {}
    pick = lambda d, ks: {k: d[k] for k in ks if k in d}  # noqa: E731
    return {
        "function_level": pick(fl, ("f1_macro", "f1_weighted", "accuracy",
                                    "auc_roc_macro_ovr", "confidence_mean", "num_test_samples")),
        "localization": pick(loc, ("ifa_mean", "top_1_accuracy", "top_5_accuracy",
                                   "recall_at_5pct_loc", "recall_at_20pct_loc")),
    }


def _evaluate_and_store(model_id: str, train_cfg_path: Path, checkpoint_path: Path,
                        run_dir: Path, log_file) -> dict | None:
    """Best-effort: run the library evaluator on the new model (task-B test split), persist the
    full metrics_summary to object storage + a model_artifact, return a compact summary for the
    job record. Uses the library's official return path (gnn_vuln.evaluate). Never raises."""
    try:
        log_file.write(f"== evaluate (task-B test): {checkpoint_path} ==\n"); log_file.flush()
        subprocess.run(["python", "-m", EVAL_MODULE, "--checkpoint", str(checkpoint_path),
                        "--config", str(train_cfg_path)],
                       check=True, cwd=str(ROOT), env={**os.environ, **ENV},
                       stdout=log_file, stderr=subprocess.STDOUT)
        msj = ROOT / "results" / run_dir.name / "metrics_summary.json"
        if not msj.exists():
            log_file.write(f"WARN metrics_summary.json not found at {msj}\n"); log_file.flush()
            return None
        summary = json.loads(msj.read_text())
        uri = storage.put_bytes(settings.S3_BUCKET_CHECKPOINTS,
                                f"{model_id}.metrics_summary.json", msj.read_bytes())
        compact = _compact_metrics(summary)
        registry.add_artifact(model_id, "metrics", uri, meta=compact)
        return compact
    except Exception as e:  # noqa: BLE001
        log_file.write(f"WARN evaluate failed for {model_id}: {type(e).__name__}: {e}\n")
        log_file.flush()
        return None


# ── job runner (invoked by the Celery task in API.tasks) ────────────────────
def execute_relearn(job_id: str, train_cfg, importance_cfg, meta: dict) -> None:
    """Run the (optional EWC-importance pass +) training subprocess for a relearn job,
    then register the resulting checkpoint as a new model. Called by the Celery worker."""
    job = get_job(job_id)
    if job is None:
        return
    train_cfg = Path(train_cfg)
    importance_cfg = Path(importance_cfg) if importance_cfg else None
    log = Path(job["log_path"])
    try:
        job["status"] = "running"; _save_job(job)
        env = {**os.environ, **ENV}
        base_model_id = meta.get("base_model_id")
        importance_cache = meta.get("importance_cache")
        with open(log, "w", encoding="utf-8") as lf:
            if importance_cfg is not None:
                lf.write(f"== EWC importance: {importance_cfg} ==\n"); lf.flush()
                subprocess.run(["python", "-m", TRAIN_MODULE, "--config", str(importance_cfg)],
                               check=True, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
                # persist the just-computed fallback importance for the BASE model (best-effort)
                if base_model_id and importance_cache and Path(importance_cache).exists():
                    try:
                        _store_importance(base_model_id, importance_cache)
                    except Exception as e:  # noqa: BLE001
                        lf.write(f"WARN persist base importance failed: {type(e).__name__}: {e}\n")
                        lf.flush()
            lf.write(f"== train: {train_cfg} ==\n"); lf.flush()
            subprocess.run(["python", "-m", TRAIN_MODULE, "--config", str(train_cfg)],
                           check=True, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        ckpts = sorted((ROOT / "checkpoints").glob("*_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        run_dir = next((c for c in ckpts if not c.name.startswith("api_base_")), None)
        best = next(run_dir.glob("best_*.pt"), None) if run_dir else None
        if best:
            new_id = f"relearn_{job['method']}_{run_dir.name}"
            # label predictions with CWE names: derive class_names from the materialized vocab
            class_names: list = []
            vocab_path = DATA_ROOT / "raw" / meta["source"] / "cwe_vocab.json"
            if vocab_path.exists():
                vocab = json.loads(vocab_path.read_text())
                class_names = [""] * len(vocab)
                for cwe, idx in vocab.items():
                    if 0 <= idx < len(class_names):
                        class_names[idx] = cwe
            # object storage is the source of truth for the checkpoint (so a worker on
            # another node can load it); the local path stays as a cache.
            ckpt_uri = storage.put_bytes(settings.S3_BUCKET_CHECKPOINTS, f"{new_id}.pt", best.read_bytes())
            registry.register_model(new_id, {
                "label": f"relearn {job['method']} ({', '.join(job['dataset_ids'])})",
                "arch": meta["arch"], "config": str(train_cfg.relative_to(ROOT)),
                "checkpoint": str(best.relative_to(ROOT)), "storage_uri": ckpt_uri,
                "dataset_id": job["dataset_ids"][0], "num_classes": meta["num_classes"],
                "class_names": class_names,
                "base_model_id": job.get("base_model_id"), "method": job["method"]})
            job["result_model_id"] = new_id
            # eager: compute + store importance for the NEW model so a future relearn is instant
            if COMPUTE_IMPORTANCE_ON_FINISH:
                out_cache = ROOT / "checkpoints" / f"api_base_{new_id}" / "ewc_importance.pt"
                with open(log, "a", encoding="utf-8") as lf:
                    _compute_and_store_importance(new_id, train_cfg, best, out_cache, lf)
            # capture eval metrics (task-B test) via the library evaluator → DB + storage
            if EVALUATE_ON_FINISH:
                with open(log, "a", encoding="utf-8") as lf:
                    job["metrics"] = _evaluate_and_store(new_id, train_cfg, best, run_dir, lf)
        job["status"] = "done"
    except subprocess.CalledProcessError as e:
        job["status"] = "failed"; job["message"] = f"training failed (exit {e.returncode}); see log"
    except Exception as e:  # noqa: BLE001
        job["status"] = "failed"; job["message"] = f"{type(e).__name__}: {e}"
    _save_job(job)


def submit_relearn(method: str, dataset_ids: list[str], base_model_id: str | None,
                   epochs: int | None, run_name: str | None) -> dict:
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job = {"job_id": job_id, "status": "queued", "method": method,
           "dataset_ids": dataset_ids, "base_model_id": base_model_id,
           "config_path": None, "log_path": str(job_dir / "run.log"),
           "result_model_id": None, "message": run_name}
    train_cfg, importance_cfg, meta = build_config(method, dataset_ids, base_model_id, epochs, job_dir)
    job["config_path"] = str(train_cfg.relative_to(ROOT))
    _save_job(job)
    # run on the Celery worker (off the web server), consistent with dataset ingestion
    from API.tasks import run_relearn
    run_relearn.delay(job_id, str(train_cfg), str(importance_cfg) if importance_cfg else None, meta)
    return job
