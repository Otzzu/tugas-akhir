"""
seed_from_drive.py — fetch the trained seed models + datasets from public Google Drive and
push them into the API's object storage (MinIO), so the API serves them with ZERO local
dependency. Checkpoints + dataset bundles become the single source of truth in MinIO;
`materialize_dataset` / `_materialize_checkpoint` download them on demand into the worker cache.

Per model it uploads:
  - checkpoint zip (checkpoints/<run>/best_*.pt)  -> s3://checkpoints/<model_id>.pt
  - lazy dataset bundle (cloud format: <name>_meta.pt + <name>_graphs/, no vocab) is repackaged
    into the API bundle (cwe_vocab.json at top + processed/<name>_meta.pt + processed/<name>_graphs/)
    -> s3://datasets/<dataset_id>.tar.gz   (subdirs preserved; cwe_vocab generated from class_names)

MinIO endpoint/creds come from env (defaults match docker-compose, reachable from the host):
  S3_ENDPOINT=http://localhost:9000  S3_ACCESS_KEY=minioadmin  S3_SECRET_KEY=minioadmin

Usage (host, from project root, stack up):
  uv run --with gdown --with boto3 python scripts/seed_from_drive.py --models all
  uv run --with gdown --with boto3 python scripts/seed_from_drive.py --models graph_based,sequential
  uv run --with gdown --with boto3 python scripts/seed_from_drive.py --models all --checkpoints-only
"""
from __future__ import annotations
import argparse, json, os, tarfile, tempfile, zipfile
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "API" / "seeds"

ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:9000")
KEY      = os.environ.get("S3_ACCESS_KEY", "minioadmin")
SECRET   = os.environ.get("S3_SECRET_KEY", "minioadmin")
BUCKET_CKPT = os.environ.get("S3_BUCKET_CHECKPOINTS", "checkpoints")
BUCKET_DATA = os.environ.get("S3_BUCKET_DATASETS", "datasets")

# model_id -> public Drive file ids + the API dataset_id its dataset registers as.
# graph_based + sequential share one dataset (ml1024 GNN-only); hybrid uses ml5120 (live LM).
DRIVE = {
    "graph_based":     {"ckpt": "1zM7YQnIhNO01vh_tmCZlDr3O7JZ0AMky", "data": "1iOSIyE6nBzSh9iDq3mCXUpiVRKZPCXnS", "dataset_id": "megavul_26"},
    "hybrid_graph_lm": {"ckpt": "1_zkdNfydWa7KHOwdMhdzS93U_EldysIo", "data": "1qFX4L7EK0TGEgHDJ5zv7q4UimdJ_2sz7", "dataset_id": "megavul_26_lm"},
    "sequential":      {"ckpt": "1yvUXUjCgMfHtH0KTUye7J1VFIqPRlkpl", "data": "1iOSIyE6nBzSh9iDq3mCXUpiVRKZPCXnS", "dataset_id": "megavul_26"},
}


def _gdown(file_id: str, out: Path) -> None:
    try:
        import gdown
    except ImportError:
        raise SystemExit("gdown missing. Run with:  uv run --with gdown --with boto3 python scripts/seed_from_drive.py ...")
    gdown.download(id=file_id, output=str(out), quiet=False)
    if not out.exists() or out.stat().st_size == 0:
        raise SystemExit(f"download failed (empty) for id={file_id}")


def _unpack(archive: Path, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(into)
    else:
        raise SystemExit(f"unknown archive type: {archive}")


def _s3():
    try:
        import boto3
        from botocore.client import Config as BotoConfig
    except ImportError:
        raise SystemExit("boto3 missing. Run with:  uv run --with gdown --with boto3 python scripts/seed_from_drive.py ...")
    s3 = boto3.client("s3", endpoint_url=ENDPOINT, aws_access_key_id=KEY,
                      aws_secret_access_key=SECRET, config=BotoConfig(signature_version="s3v4"))
    have = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
    for b in (BUCKET_CKPT, BUCKET_DATA):
        if b not in have:
            s3.create_bucket(Bucket=b)
    return s3


def seed_checkpoint(s3, mid: str, file_id: str, tmp: Path) -> None:
    arc = tmp / f"{mid}_ckpt.bin"
    _gdown(file_id, arc)
    ex = tmp / f"{mid}_ckpt_x"
    _unpack(arc, ex)
    best = next(ex.rglob("best_*.pt"), None) or next(ex.rglob("*.pt"), None)
    if best is None:
        raise SystemExit(f"no best_*.pt inside checkpoint archive for {mid}")
    s3.upload_file(str(best), BUCKET_CKPT, f"{mid}.pt")
    print(f"  checkpoint -> s3://{BUCKET_CKPT}/{mid}.pt  (from {best.name})")


def seed_dataset(s3, dataset_id: str, class_names: list[str], file_id: str,
                 tmp: Path, done: set) -> None:
    if dataset_id in done:
        print(f"  dataset    -> s3://{BUCKET_DATA}/{dataset_id}.tar.gz already uploaded this run")
        return
    arc = tmp / f"data_{dataset_id}.bin"
    _gdown(file_id, arc)
    # cloud bundle = top-level <name>_meta.pt + <name>_graphs/ (no vocab). Extract into
    # stage/processed/ so they sit under processed/, then add the generated vocab at top.
    stage = tmp / f"stage_{dataset_id}"
    (stage / "processed").mkdir(parents=True, exist_ok=True)
    _unpack(arc, stage / "processed")
    (stage / "cwe_vocab.json").write_text(
        json.dumps({c: i for i, c in enumerate(class_names)}, indent=2), encoding="utf-8")
    # repackage into the API materialize bundle: cwe_vocab.json (top) + processed/<...>
    bundle = tmp / f"{dataset_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as t:
        t.add(stage / "cwe_vocab.json", arcname="cwe_vocab.json")
        t.add(stage / "processed", arcname="processed")
    s3.upload_file(str(bundle), BUCKET_DATA, f"{dataset_id}.tar.gz")
    print(f"  dataset    -> s3://{BUCKET_DATA}/{dataset_id}.tar.gz  ({bundle.stat().st_size/1e6:.1f} MB)")
    done.add(dataset_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed trained models + datasets from public Drive into MinIO")
    ap.add_argument("--models", default="all",
                    help="'all' or comma-separated: graph_based,hybrid_graph_lm,sequential")
    ap.add_argument("--checkpoints-only", action="store_true",
                    help="upload only checkpoints (inference-ready), skip the dataset bundles")
    args = ap.parse_args()

    models_meta = json.loads((SEEDS / "models.json").read_text())
    sel = list(DRIVE) if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in sel if m not in DRIVE]
    if bad:
        raise SystemExit(f"unknown model(s): {bad}. choose from {list(DRIVE)}")

    s3 = _s3()
    print(f"MinIO: {ENDPOINT}  buckets: {BUCKET_CKPT}, {BUCKET_DATA}")
    done: set = set()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for mid in sel:
            d = DRIVE[mid]
            print(f"\n== {mid} (dataset_id={d['dataset_id']}) ==")
            seed_checkpoint(s3, mid, d["ckpt"], tmp)
            if not args.checkpoints_only:
                seed_dataset(s3, d["dataset_id"], models_meta[mid]["class_names"], d["data"], tmp, done)

    print("\nDone. Object storage is the source of truth. Bring up the API; seed_if_empty "
          "registers from API/seeds/*.json and inference/relearn materialize from MinIO (no local data).")


if __name__ == "__main__":
    main()
