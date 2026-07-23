#!/usr/bin/env python3
"""rebundle_checkpoint_vocab.py — embed 26-class class_names into existing checkpoint zips.

Older checkpoints were saved WITHOUT class_names (index -> CWE mapping), so serving fell
back to cwe_vocab.json / ML-CLASS-i. gnn_vuln >= 0.1.16.post1 reads class_names straight
from the checkpoint. This script downloads each checkpoint zip from Drive, injects the
26-class vocab into every best_*.pt and last_*.pt, re-zips with the same internal layout,
and re-uploads in place (same filename -> same Drive file id).

Only torch + rclone are needed (the model is never instantiated — the checkpoint dict is
edited directly). Run on a pod with rclone gdrive-mesach configured.

  uv run python scripts/rebundle_checkpoint_vocab.py                 # 3 seed-42 26-class runs
  uv run python scripts/rebundle_checkpoint_vocab.py --runs 20260629_154445,20260629_155935
  uv run python scripts/rebundle_checkpoint_vocab.py --dry-run       # download+inject, no upload
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import torch

REMOTE = os.environ.get("CKPT_REMOTE", "gdrive-mesach:tugas-akhir/checkpoints")

# 26-class label space (index -> CWE), index 0 = benign. Same ordering the serving DB seeds
# (API/seeds/models.json) and the megavul cwe_vocab.json the models were trained on.
CLASS_NAMES_26 = [
    "benign", "CWE-787", "CWE-125", "CWE-476", "CWE-20", "CWE-416", "CWE-200", "CWE-120",
    "CWE-79", "CWE-22", "CWE-89", "CWE-770", "CWE-502", "CWE-122", "CWE-284", "CWE-863",
    "CWE-78", "CWE-862", "CWE-94", "CWE-918", "CWE-434", "CWE-352", "CWE-77", "CWE-121",
    "CWE-306", "CWE-639",
]

# seed 42 / 26-class final backbones (ABLATION_RESULTS.md "Raw per-seed nine — 26 kelas")
DEFAULT_RUNS = [
    "20260629_151930",  # graph  (N48)
    "20260630_101936",  # hibrida (O1)
    "20260630_183927",  # seq    (S1)
]


def rclone_out(*args: str) -> str:
    return subprocess.check_output(["rclone", *args], text=True)


def find_zip(run_id: str) -> str:
    """Return the single <run_id>_*_checkpoints.zip filename on the remote."""
    names = [
        n.strip()
        for n in rclone_out("lsf", REMOTE, "--include", f"{run_id}_*_checkpoints.zip").splitlines()
        if n.strip()
    ]
    if not names:
        raise SystemExit(f"[{run_id}] no matching zip in {REMOTE}")
    if len(names) > 1:
        raise SystemExit(f"[{run_id}] ambiguous, {len(names)} matches: {names}")
    return names[0]


def inject(pt_path: Path, class_names: list[str]) -> None:
    st = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = st.get("model_state_dict")
    if sd is None:
        raise SystemExit(f"{pt_path.name}: no model_state_dict")
    n = len(class_names)
    # a genuine n-class model must carry an n-output classifier weight — this guards against
    # injecting the 26-class vocab into a 25-class vuln-only checkpoint by mistake.
    if not any(
        hasattr(v, "ndim") and v.ndim >= 1 and v.shape[0] == n for v in sd.values()
    ):
        raise SystemExit(
            f"{pt_path.name}: no {n}-output head found — wrong label space, refusing to inject"
        )
    st["class_names"] = list(class_names)
    torch.save(st, pt_path)
    f1 = st.get("val_f1", "?")
    print(f"    injected {n} class_names into {pt_path.name} (val_f1={f1})")


def rezip(run_dir: Path, work: Path, out_zip: Path) -> None:
    """Zip work/checkpoints/<id>/ back with checkpoints/<id>/... arcnames."""
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(run_dir.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(work))


def process(run_id: str, dry_run: bool, class_names: list[str]) -> None:
    zip_name = find_zip(run_id)
    print(f"== {run_id}  ({zip_name}) ==")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        local_zip = tmp / zip_name
        subprocess.run(["rclone", "copyto", f"{REMOTE}/{zip_name}", str(local_zip)], check=True)

        work = tmp / "x"
        work.mkdir()
        with zipfile.ZipFile(local_zip) as z:
            z.extractall(work)

        run_dirs = [p for p in (work / "checkpoints").iterdir() if p.is_dir()]
        if len(run_dirs) != 1:
            raise SystemExit(f"[{run_id}] expected one run dir, got {[p.name for p in run_dirs]}")
        run_dir = run_dirs[0]

        pts = sorted(run_dir.glob("best_*.pt")) + sorted(run_dir.glob("last_*.pt"))
        if not pts:
            raise SystemExit(f"[{run_id}] no best_*.pt / last_*.pt in {run_dir.name}")
        for pt in pts:
            inject(pt, class_names)

        out_zip = tmp / f"rebundled_{zip_name}"
        rezip(run_dir, work, out_zip)

        if dry_run:
            print(f"    [dry-run] would upload -> {REMOTE}/{zip_name}  ({out_zip.stat().st_size/1e6:.1f} MB)")
            return
        subprocess.run(["rclone", "copyto", str(out_zip), f"{REMOTE}/{zip_name}"], check=True)
        print(f"    uploaded -> {REMOTE}/{zip_name}  (overwritten in place)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed 26-class class_names into checkpoint zips on Drive")
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS),
                    help="comma-separated run-id prefixes (default: seed-42 graph,hibrida,seq)")
    ap.add_argument("--vocab-json", default=None,
                    help="optional path to a cwe_vocab.json (name->idx) or a JSON list to override the baked 26-class vocab")
    ap.add_argument("--dry-run", action="store_true", help="download + inject locally, skip upload")
    args = ap.parse_args()

    class_names = CLASS_NAMES_26
    if args.vocab_json:
        obj = json.loads(Path(args.vocab_json).read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            class_names = [k for k, _ in sorted(obj.items(), key=lambda kv: kv[1])]
        elif isinstance(obj, list):
            class_names = obj
        else:
            raise SystemExit("--vocab-json must be a name->idx object or a JSON list")
    print(f"class_names ({len(class_names)}): {class_names}\n")

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    for run_id in runs:
        process(run_id, args.dry_run, class_names)
    print("\nDone.")


if __name__ == "__main__":
    main()
