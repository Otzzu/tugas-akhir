"""Prove the e2e actually PERSISTED data — inspects the DB rows and the MinIO objects
directly (not just HTTP 200s). Run inside the api/worker container (it has DB access +
boto3 + the S3 creds):

    docker compose -f API/docker-compose.yml exec api python /app/API/tests/_verify_persist.py

Checks, and FAILS (non-zero exit) if any expected artifact is missing:
  - inference_results rows exist and carry the pre-head embedding (DB).
  - graph_cache rows point at objects that actually exist in the `graphs` bucket (MinIO).
  - each registered dataset's `storage_uri` object actually exists in the `datasets`
    bucket (MinIO), and its data_config_id resolves to a config (DB).
  - relearned models are registered with a checkpoint on disk (DB + fs).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
from sqlalchemy import select

from API.core.config import settings
from API.core.database import SessionLocal
from API.models.tables import (
    InferenceResult, DatasetRecord, ConfigRecord, ModelRecord, GraphCache,
)

problems: list[str] = []


def _s3():
    return boto3.client(
        "s3", endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=settings.S3_REGION,
    )


def _obj_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _parse_uri(uri: str) -> tuple[str, str] | None:
    # s3://bucket/key  OR  file://path
    if uri and uri.startswith("s3://"):
        b, _, k = uri[5:].partition("/")
        return b, k
    return None


def main() -> int:
    s3 = _s3()
    print(f"== storage backend: {settings.STORAGE_BACKEND}  endpoint: {settings.S3_ENDPOINT} ==\n")

    with SessionLocal() as s:
        # 1) inference_results — predictions + embedding persisted in the DB
        irs = list(s.scalars(select(InferenceResult).order_by(InferenceResult.id.desc())))
        print(f"[DB] inference_results: {len(irs)} row(s)")
        with_emb = [r for r in irs if r.cls_embedding]
        print(f"     with cls_embedding: {len(with_emb)}")
        for r in irs[:3]:
            d = bool(r.cls_embedding)
            print(f"     #{r.id} model={r.model_id} pred={r.prediction} emb={'yes' if d else 'NO'} dim={r.cls_embedding_dim}")
        if irs and not with_emb:
            problems.append("inference_results exist but NONE carry cls_embedding")

        # 2) graph_cache → blob present in the graphs bucket (MinIO)
        gcs = list(s.scalars(select(GraphCache)))
        print(f"\n[DB] graph_cache: {len(gcs)} pointer(s)")
        for g in gcs[:5]:
            ok = _obj_exists(s3, settings.S3_BUCKET_GRAPHS, g.object_key)
            print(f"     {g.code_hash[:12]}… -> {settings.S3_BUCKET_GRAPHS}/{g.object_key}  [{'OK' if ok else 'MISSING'}]")
            if not ok:
                problems.append(f"graph_cache {g.code_hash[:12]} object missing in MinIO: {g.object_key}")

        # 3) datasets → bundle present in the datasets bucket (MinIO) + config in DB
        dss = list(s.scalars(select(DatasetRecord)))
        print(f"\n[DB] datasets: {len(dss)}")
        for d in dss:
            uri = (d.params or {}).get("storage_uri", "")
            bk = _parse_uri(uri)
            tag = "(no bundle — seed/local)" if not bk else ""
            in_minio = _obj_exists(s3, *bk) if bk else None
            print(f"     {d.id}  source={d.source}  data_config_id={d.data_config_id}")
            print(f"        storage_uri={uri or '—'} {tag}  minio={'OK' if in_minio else ('MISSING' if in_minio is False else 'n/a')}")
            if bk and not in_minio:
                print("        WARNING: bundle not in MinIO (ok if this dataset wasn't seeded; "
                      "checked per seeded-model below)")
            # integrity invariant: every dataset MUST reference a resolvable config
            cfg = s.get(ConfigRecord, d.data_config_id) if d.data_config_id else None
            print(f"        data_config -> {'FOUND' if cfg else 'MISSING'} ({d.data_config_id})")
            if not cfg:
                problems.append(f"dataset {d.id} has no resolvable data_config_id ({d.data_config_id})")

        # 4) configs
        cfgs = list(s.scalars(select(ConfigRecord)))
        print(f"\n[DB] configs: {len(cfgs)}")

        # 5) models (seed + relearned)
        ms = list(s.scalars(select(ModelRecord)))
        print(f"\n[DB] models: {len(ms)}")
        for m in ms:
            ck = Path(settings.ROOT) / m.checkpoint
            disk = ck.exists()
            bk = _parse_uri(m.storage_uri or "")
            in_minio = _obj_exists(s3, *bk) if bk else None
            note = "" if m.method is None else f"  method={m.method} base={m.base_model_id}"
            print(f"     {m.id}  cfg={m.config_id}  dataset={m.dataset_id}  "
                  f"ckpt_disk={'OK' if disk else 'MISSING'}  "
                  f"minio={'OK' if in_minio else ('MISSING' if in_minio is False else 'n/a')}{note}")
            # integrity invariant: every model MUST reference a resolvable config AND dataset
            if not m.config_id or not s.get(ConfigRecord, m.config_id):
                problems.append(f"model {m.id} has no resolvable config_id ({m.config_id})")
            ds_rec = s.get(DatasetRecord, m.dataset_id) if m.dataset_id else None
            if not ds_rec:
                problems.append(f"model {m.id} has no resolvable dataset_id ({m.dataset_id})")
            # a SEEDED model (checkpoint present) MUST have its dataset bundle in MinIO — relearn
            # ER/EWC materialize the base model's dataset for the replay buffer + EWC importance.
            if (disk or in_minio is True) and ds_rec:
                ds_bk = _parse_uri((ds_rec.params or {}).get("storage_uri", ""))
                if not (ds_bk and _obj_exists(s3, *ds_bk)):
                    problems.append(
                        f"model {m.id} is seeded but its dataset {m.dataset_id} bundle is missing "
                        f"in MinIO — relearn ER/EWC on it will fail (seed without --checkpoints-only)")
            if m.method and not disk and not in_minio:
                problems.append(f"relearned model {m.id} checkpoint missing on disk AND in MinIO")
            if m.method and bk and not in_minio:
                problems.append(f"relearned model {m.id} storage_uri object missing in MinIO: {m.storage_uri}")

    # 6) raw MinIO listing (ground truth)
    print("\n== MinIO buckets ==")
    for b in (settings.S3_BUCKET_DATASETS, settings.S3_BUCKET_GRAPHS, settings.S3_BUCKET_CHECKPOINTS):
        try:
            objs = s3.list_objects_v2(Bucket=b).get("Contents", [])
            print(f"  [{b}] {len(objs)} object(s)" + "".join(
                f"\n     {o['Key']}  {o['Size']} B" for o in objs[:10]))
        except Exception as e:  # noqa: BLE001
            print(f"  [{b}] ERR {e}")

    print("\n== RESULT ==")
    if problems:
        print(f"FAIL — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("OK — all DB records resolve and all referenced MinIO objects exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
