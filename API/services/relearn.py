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
import json
import os
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from sqlalchemy import select

from API.core.config import settings
from API.core.database import SessionLocal
from API.models.tables import JobRecord
from API.services import registry

ROOT = settings.ROOT
JOBS_DIR = settings.JOBS_DIR
ENV = {"PYTHONPATH": "src"}

_REQUIRES_BASE = {"ER", "EWC", "EWC-ER", "finetune"}
_JOB_FIELDS = ("status", "method", "dataset_ids", "base_model_id",
               "config_path", "log_path", "result_model_id", "message")


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


# ── dataset join ────────────────────────────────────────────────────────────
def _join_datasets(dataset_ids: list[str]) -> tuple[str, dict]:
    """Single id → its source. Multiple → merge raw CPG dirs into a combined source."""
    if len(dataset_ids) == 1:
        d = registry.get_dataset(dataset_ids[0])
        return d["source"], d

    import shutil
    metas = [registry.get_dataset(i) for i in dataset_ids]
    joined = "join_" + "_".join(dataset_ids)
    raw_root = ROOT / "data" / "raw"
    dst = raw_root / joined
    if not dst.exists():
        (dst / "benign").mkdir(parents=True, exist_ok=True)
        (dst / "vulnerable").mkdir(parents=True, exist_ok=True)
        vocab: dict[str, int] = {}
        for m in metas:
            src = raw_root / m["source"]
            if not src.exists():
                raise FileNotFoundError(
                    f"Cannot join: raw CPG dir missing for '{m['source']}' ({src})."
                )
            for sub in ("benign", "vulnerable"):
                for f in (src / sub).glob("*"):
                    tgt = dst / sub / f"{m['source']}__{f.name}"
                    if not tgt.exists():
                        shutil.copy2(f, tgt)
            vf = src / "cwe_vocab.json"
            if vf.exists():
                for k, v in json.loads(vf.read_text()).items():
                    vocab.setdefault(k, v)
        (dst / "cwe_vocab.json").write_text(json.dumps(vocab, indent=2))
    base = copy.deepcopy(metas[0])
    base["source"] = joined
    base["num_classes"] = max(m.get("num_classes", 0) for m in metas)
    return joined, base


# ── config generation ───────────────────────────────────────────────────────
def build_config(method: str, dataset_ids: list[str], base_model_id: str | None,
                 epochs: int | None, job_dir: Path) -> tuple[Path, Path | None, dict]:
    source, data_block = _join_datasets(dataset_ids)

    if method in _REQUIRES_BASE:
        if not base_model_id:
            raise ValueError(f"method '{method}' requires base_model_id")
        base = registry.get_model(base_model_id)
        base_cfg = yaml.safe_load(registry.abspath(base["config"]).read_text())
        base_ckpt = registry.abspath(base["checkpoint"])
        base_ds = registry.get_dataset(base["dataset_id"])
    else:  # retrain — fresh weights; optional template for arch
        if base_model_id:
            base = registry.get_model(base_model_id)
            base_cfg = yaml.safe_load(registry.abspath(base["config"]).read_text())
        else:
            base_cfg = yaml.safe_load((ROOT / "configs/ablation/gnn_only/N48_a1_l1_jknet.yaml").read_text())
        base_ckpt = None
        base_ds = None

    cfg = copy.deepcopy(base_cfg)
    cfg["data"] = {**cfg.get("data", {}), **{
        "source": source, "mode": data_block.get("mode", "multiclass"),
        "max_nodes": data_block.get("max_nodes", 2500),
        "top_cwe": data_block.get("top_cwe", 0),
        "filter_top25_dangerous": data_block.get("filter_top25_dangerous", False),
        "max_per_class": data_block.get("max_per_class", 0),
        "resample_seed": data_block.get("resample_seed", 42),
        "storage": data_block.get("storage", "lazy"),
        "ds_name_suffix": data_block.get("ds_name_suffix", ""),
    }}
    cfg["model"]["num_classes"] = data_block.get("num_classes", cfg["model"].get("num_classes", 26))
    if epochs:
        cfg.setdefault("train", {})["epochs"] = epochs

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
            "arch": cfg["model"]["architecture"]}
    return train_cfg_path, importance_cfg_path, meta


# ── job runner ──────────────────────────────────────────────────────────────
def _run_job(job: dict, train_cfg: Path, importance_cfg: Path | None, meta: dict) -> None:
    log = Path(job["log_path"])
    try:
        job["status"] = "running"; _save_job(job)
        env = {**os.environ, **ENV}
        with open(log, "w", encoding="utf-8") as lf:
            if importance_cfg is not None:
                lf.write(f"== EWC importance: {importance_cfg} ==\n"); lf.flush()
                subprocess.run(["python", "-m", "gnn_vuln.train", "--config", str(importance_cfg)],
                               check=True, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
            lf.write(f"== train: {train_cfg} ==\n"); lf.flush()
            subprocess.run(["python", "-m", "gnn_vuln.train", "--config", str(train_cfg)],
                           check=True, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        ckpts = sorted((ROOT / "checkpoints").glob("*_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        run_dir = next((c for c in ckpts if not c.name.startswith("api_base_")), None)
        best = next(run_dir.glob("best_*.pt"), None) if run_dir else None
        if best:
            new_id = f"relearn_{job['method']}_{run_dir.name}"
            registry.register_model(new_id, {
                "label": f"relearn {job['method']} ({', '.join(job['dataset_ids'])})",
                "arch": meta["arch"], "config": str(train_cfg.relative_to(ROOT)),
                "checkpoint": str(best.relative_to(ROOT)),
                "dataset_id": job["dataset_ids"][0], "num_classes": meta["num_classes"],
                "base_model_id": job.get("base_model_id"), "method": job["method"]})
            job["result_model_id"] = new_id
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
    threading.Thread(target=_run_job, args=(job, train_cfg, importance_cfg, meta), daemon=True).start()
    return job
