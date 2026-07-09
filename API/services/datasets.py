"""Dataset-ingestion service — thin orchestration around the Celery task.

The endpoint stays oblivious to implementation: it calls create_ingest_job (which
persists the raw rows + a job row and enqueues the Celery task) and polls get_job.
The heavy raw->CPG->.pt build lives in the gnn_vuln library, driven by API.tasks.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Optional

from API.core.config import settings
from API.core.database import SessionLocal
from API.models.tables import DatasetJobRecord
from API.services import registry


def _resolve_data_config(req) -> dict:
    """Build params for the dataset, layered: DataConfigOverride defaults, then the data_config_id
    config's data section (if given) as the base, then the EXPLICITLY-set inline data_config
    fields on top. So data_config_id is the template and inline data_config overrides it field by
    field. Returns a DataConfigOverride-shaped dict so the rest of the pipeline is unchanged."""
    from API.schemas.dataset import DataConfigOverride
    dc = DataConfigOverride().model_dump()
    if getattr(req, "data_config_id", None):
        sec = registry.config_section(req.data_config_id, "data")   # base from the referenced config
        for k in dc:
            if k in sec:
                dc[k] = sec[k]
    dc.update(req.config.model_dump(exclude_unset=True))             # explicit inline overrides win
    return dc


def _slug(name: str) -> str:
    base = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower() or "dataset"
    return base


_MIN_LIB_FLAW_LINES = (0, 1, 10)    # gnn-vuln release with --flaw-lines-column


def _lib_supports_flaw_lines() -> bool:
    """Before 0.1.10 the loader had no --flaw-lines-column opt-in and silently fell back to
    the func_after diff, so an old library must reject a manually annotated row loudly."""
    from importlib.metadata import version
    try:
        parts = tuple(int(p) for p in version("gnn-vuln").split(".")[:3])
    except Exception:
        return False
    return parts >= _MIN_LIB_FLAW_LINES


def _clean_flaw_lines(code: str, flaw_lines: list[int], row_idx: int) -> list[int]:
    n = len(code.splitlines())
    want = sorted({int(i) for i in flaw_lines})
    bad = [i for i in want if i < 1 or i > n]
    if bad:
        raise UploadParseError(f"row {row_idx}: flaw_lines {bad} outside 1..{n}")
    return want


def _diff_flaw_lines(before: str, after: str) -> list[int]:
    """Mirror of gnn_vuln.data.prepare._diff_flaw_lines. Needed when a file mixes annotated
    and patched rows: the flaw_lines column then covers every row, so the func_after rows
    must be diffed here rather than by the loader."""
    import difflib
    if not before or not after or before == after:
        return []
    b, a = before.splitlines(keepends=True), after.splitlines(keepends=True)
    out: list[int] = []
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, b, a).get_opcodes():
        if tag in ("replace", "delete"):
            out.extend(range(i1 + 1, i2 + 1))
    return out


def _rows_to_parquet(rows: list, path) -> int:
    """Normalize ingest rows into the BigVul-style columns the library loader reads:
    func_before / func_after / 'CWE ID' / vul / language. A vulnerable row localizes itself
    with exactly one of `flaw_lines` (manual annotation) or `func_after` (patch to diff)."""
    import pandas as pd
    recs = []
    any_manual = False
    for i, r in enumerate(rows):
        cwe = (r.cwe or "").strip()
        vul = 1 if (cwe or (r.label is not None and r.label > 0)) else 0
        manual = getattr(r, "flaw_lines", None)
        has_after = bool(r.func_after)
        if manual is not None and has_after:
            raise UploadParseError(f"row {i}: give either func_after or flaw_lines, not both")
        if vul and manual is None and not has_after:
            raise UploadParseError(f"row {i}: a vulnerable row needs func_after or flaw_lines")
        flaw = _clean_flaw_lines(r.code, manual, i) if (manual and vul) else []
        any_manual |= bool(flaw)
        recs.append({
            "func_before": r.code,
            "func_after": r.func_after or "",
            "flaw_lines": flaw,
            "CWE ID": cwe,
            "vul": vul,
            "language": r.language or "C",
        })
    if not any_manual:
        for rec in recs:                    # no annotation anywhere -> plain diff path
            rec.pop("flaw_lines")
    else:
        if not _lib_supports_flaw_lines():
            raise UploadParseError(
                "'flaw_lines' needs gnn-vuln >= "
                f"{'.'.join(map(str, _MIN_LIB_FLAW_LINES))}; supply func_after instead")
        # the column is authoritative once present, so fill it for the patched rows too
        for rec in recs:
            if rec["vul"] and rec["func_after"] and not rec["flaw_lines"]:
                rec["flaw_lines"] = _diff_flaw_lines(rec["func_before"], rec["func_after"])
    df = pd.DataFrame(recs)
    df.to_parquet(path)
    return len(df)


class UploadParseError(ValueError):
    """Malformed dataset file — carries a human-readable reason for a 422."""


@dataclass
class FileIngest:
    """What create_ingest_job needs, sourced from an uploaded file rather than a JSON body.
    Bypasses DatasetIngestRequest's 5000-row inline cap (files get MAX_UPLOAD_ROWS instead)."""
    name: str
    rows: list
    data_config_id: Optional[str]
    config: object                      # DataConfigOverride


def _rows_from_payload(raw: bytes, filename: str) -> list[dict]:
    """A dataset file is a list of rows, serialized either as a JSON array (.json) or as
    one JSON object per line (.jsonl). Dataset-level fields live on the request, not in
    the file, so there is exactly one way to express each of them."""
    text = raw.decode("utf-8-sig")
    if filename.lower().endswith(".jsonl"):
        rows = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise UploadParseError(f"line {i} is not valid JSON: {e.msg}")
        return rows
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise UploadParseError(f"not valid JSON: {e.msg} (line {e.lineno})")
    if not isinstance(doc, list):
        raise UploadParseError(f"expected a JSON array of rows, got {type(doc).__name__}")
    return doc


def parse_upload(raw: bytes, filename: str, name: str | None,
                 data_config_id: str | None, config_json: str | None) -> FileIngest:
    """Validate an uploaded dataset file into a FileIngest."""
    from API.schemas.dataset import DataConfigOverride, DatasetRow

    if len(raw) > settings.MAX_UPLOAD_BYTES:
        raise UploadParseError(f"file exceeds {settings.MAX_UPLOAD_BYTES} bytes")
    if not filename.lower().endswith((".json", ".jsonl")):
        raise UploadParseError("file must be .json or .jsonl")

    raw_rows = _rows_from_payload(raw, filename)
    if not raw_rows:
        raise UploadParseError("file contains no rows")
    if len(raw_rows) > settings.MAX_UPLOAD_ROWS:
        raise UploadParseError(f"{len(raw_rows)} rows exceeds the {settings.MAX_UPLOAD_ROWS} limit")

    rows = []
    for i, r in enumerate(raw_rows):
        if not isinstance(r, dict):
            raise UploadParseError(f"row {i}: expected an object, got {type(r).__name__}")
        try:
            rows.append(DatasetRow(**r))
        except Exception as e:
            raise UploadParseError(f"row {i}: {e}")

    cfg_src = {}
    if config_json:
        try:
            cfg_src = json.loads(config_json)
        except json.JSONDecodeError as e:
            raise UploadParseError(f"'config' form field is not valid JSON: {e.msg}")
    if not isinstance(cfg_src, dict):
        raise UploadParseError("'config' must be a JSON object")
    try:
        config = DataConfigOverride(**cfg_src)
    except Exception as e:
        raise UploadParseError(f"'config': {e}")

    return FileIngest(name=name or filename.rsplit(".", 1)[0], rows=rows,
                      data_config_id=data_config_id, config=config)


def create_ingest_job(req) -> dict:
    """Persist input rows + a job row, enqueue the Celery task, return the job dict.
    Accepts a DatasetIngestRequest (inline rows) or a FileIngest (uploaded file)."""
    job_id = uuid.uuid4().hex[:16]
    dataset_id = f"{_slug(req.name)}_{job_id[:8]}"
    job_dir = settings.JOBS_DIR / "datasets" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / "input.parquet"
    n_rows = _rows_to_parquet(req.rows, input_path)

    data_config = {**_resolve_data_config(req), "name": req.name,
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

    data_config = {**_resolve_data_config(req), "name": req.name,
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
