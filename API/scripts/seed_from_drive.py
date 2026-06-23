"""
seed_from_drive.py — populate the API's object storage (MinIO) with the trained seed models
and their datasets, so the API serves them with ZERO local dependency.

Datasets are pulled as the API-format bundles already prepared on Drive by
scripts/patch_dataset_to_api.sh (cwe_vocab.json + processed/<files>, subdirs preserved) and
copied straight into MinIO — no repackaging here. Checkpoints come from the public Drive file
ids and are unzipped to best_*.pt.

  - checkpoint zip (checkpoints/<run>/best_*.pt)  -> s3://checkpoints/<model_id>.pt
  - API dataset bundle (gdrive api_datasets/<source>/<name>_api.tar.gz)  -> s3://datasets/<dataset_id>.tar.gz

MinIO endpoint/creds from env (defaults match docker-compose, reachable from the host):
  S3_ENDPOINT=http://localhost:9000  S3_ACCESS_KEY=minioadmin  S3_SECRET_KEY=minioadmin
Datasets need rclone on PATH with the gdrive-mesach remote configured.

Each model is ALWAYS seeded together with its dataset (relearn ER/EWC need the base model's
dataset, so a checkpoint without its dataset would be unusable). megavul_mini is also seeded.

Usage (host, from project root, stack up):
  uv run --with gdown --with boto3 python API/scripts/seed_from_drive.py --models all
  uv run --with gdown --with boto3 python API/scripts/seed_from_drive.py --models graph_based,sequential
"""
from __future__ import annotations
import argparse, os, subprocess, tarfile, tempfile, zipfile
from pathlib import Path

ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:9000")
KEY      = os.environ.get("S3_ACCESS_KEY", "minioadmin")
SECRET   = os.environ.get("S3_SECRET_KEY", "minioadmin")
BUCKET_CKPT = os.environ.get("S3_BUCKET_CHECKPOINTS", "checkpoints")
BUCKET_DATA = os.environ.get("S3_BUCKET_DATASETS", "datasets")
GDRIVE = "gdrive-mesach:tugas-akhir"

# model_id -> checkpoint Drive file id + API dataset bundle path on Drive + the dataset_id it
# registers as. graph_based + sequential share the ml1024 bundle; hybrid uses the ml5120 one.
DRIVE = {
    "graph_based": {
        "ckpt": "1zM7YQnIhNO01vh_tmCZlDr3O7JZ0AMky",
        "data": f"{GDRIVE}/api_datasets/megavul/lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42_api.tar.gz",
        "dataset_id": "megavul_26",
    },
    "hybrid_graph_lm": {
        "ckpt": "1_zkdNfydWa7KHOwdMhdzS93U_EldysIo",
        "data": f"{GDRIVE}/api_datasets/megavul/lm_dataset_megavul_multiclass_unixcoder-base_ft_ml5120_f40f2e964_s1600r42_api.tar.gz",
        "dataset_id": "megavul_26_lm",
    },
    "sequential": {
        "ckpt": "1yvUXUjCgMfHtH0KTUye7J1VFIqPRlkpl",
        "data": f"{GDRIVE}/api_datasets/megavul/lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42_api.tar.gz",
        "dataset_id": "megavul_26",
    },
}

# Dataset-only seeds (no model) — already API-format bundles on Drive, copied straight to MinIO.
# megavul_mini = small 26-class subsample for demos + the relearn task-A test path.
EXTRA_DATASETS = {
    "megavul_mini": f"{GDRIVE}/data/processed/megavul_mini/megavul_mini.tar.gz",
}


def _gdown(file_id: str, out: Path) -> None:
    try:
        import gdown
    except ImportError:
        raise SystemExit("gdown missing. Run with:  uv run --with gdown --with boto3 python API/scripts/seed_from_drive.py ...")
    gdown.download(id=file_id, output=str(out), quiet=False)
    if not out.exists() or out.stat().st_size == 0:
        raise SystemExit(f"download failed (empty) for id={file_id}")


def _s3():
    try:
        import boto3
        from botocore.client import Config as BotoConfig
    except ImportError:
        raise SystemExit("boto3 missing. Run with:  uv run --with gdown --with boto3 python API/scripts/seed_from_drive.py ...")
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
    ex.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(arc):
        with zipfile.ZipFile(arc) as z: z.extractall(ex)
    elif tarfile.is_tarfile(arc):
        with tarfile.open(arc) as t: t.extractall(ex)
    else:
        raise SystemExit(f"checkpoint archive for {mid} is neither zip nor tar")
    best = next(ex.rglob("best_*.pt"), None) or next(ex.rglob("*.pt"), None)
    if best is None:
        raise SystemExit(f"no best_*.pt inside checkpoint archive for {mid}")
    s3.upload_file(str(best), BUCKET_CKPT, f"{mid}.pt")
    print(f"  checkpoint -> s3://{BUCKET_CKPT}/{mid}.pt  (from {best.name})")


def seed_dataset(s3, dataset_id: str, drive_path: str, tmp: Path, done: set) -> None:
    if dataset_id in done:
        print(f"  dataset    -> s3://{BUCKET_DATA}/{dataset_id}.tar.gz already uploaded this run")
        return
    # copy the ready-made API bundle from Drive, then push it to MinIO unchanged.
    subprocess.run(["rclone", "copy", drive_path, str(tmp), "--progress"], check=True)
    bundle = tmp / Path(drive_path).name
    if not bundle.exists():
        raise SystemExit(f"rclone did not produce {bundle} (check {drive_path})")
    s3.upload_file(str(bundle), BUCKET_DATA, f"{dataset_id}.tar.gz")
    print(f"  dataset    -> s3://{BUCKET_DATA}/{dataset_id}.tar.gz  ({bundle.stat().st_size/1e6:.1f} MB, from {Path(drive_path).name})")
    done.add(dataset_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed trained models + datasets from Drive into MinIO")
    ap.add_argument("--models", default="all",
                    help="'all' or comma-separated: graph_based,hybrid_graph_lm,sequential")
    args = ap.parse_args()

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
            # a model is always seeded WITH its dataset — relearn ER/EWC materialize the base
            # model's dataset, so seeding a checkpoint without its dataset would be unusable.
            seed_checkpoint(s3, mid, d["ckpt"], tmp)
            seed_dataset(s3, d["dataset_id"], d["data"], tmp, done)

        # small test/demo datasets — always seeded (cheap, small) so a single-model seed still
        # has megavul_mini for the /train + /relearn smoke tests.
        for did, path in EXTRA_DATASETS.items():
            print(f"\n== dataset {did} ==")
            seed_dataset(s3, did, path, tmp, done)

    print("\nDone. Object storage is the source of truth. Bring up the API; seed_if_empty "
          "registers from API/seeds/*.json and inference/relearn materialize from MinIO (no local data).")


if __name__ == "__main__":
    main()
