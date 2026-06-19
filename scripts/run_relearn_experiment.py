"""
run_relearn_experiment.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
One-shot continual-learning (relearn) experiment driver. Runs every method,
evaluates each on task-A (MegaVul) and task-B (relearn) test sets, computes
forgetting, writes RELEARN_RESULTS.md, and uploads it to Drive.

Prerequisites on the pod (see configs/ablation/relearn/*.yaml headers):
  - data/raw/relearn/                       (restored CPG bundle: vulnerable + benign)
  - task-A 26-class N48 checkpoint           -> checkpoints/n48_taskA/best_model.pt
  - MegaVul 26-class .pt + relearn .pt build automatically on first use (GPU)

Run (cloud, Linux):
  PYTHONPATH=src python scripts/run_relearn_experiment.py
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CKPTS = ROOT / "checkpoints"
CFG = ROOT / "configs" / "ablation" / "relearn"
MEGAVUL_CFG = CFG / "N48_taskA_importance.yaml"        # data.source=megavul (task-A, 26-class)
RELEARN_CFG = CFG / "N48_relearn_naive.yaml"           # data.source=relearn (task-B, any relearn cfg)
TASKA_CKPT = CKPTS / "n48_taskA" / "best_model.pt"
DRIVE = "gdrive-mesach:tugas-akhir/results/"
OUT_MD = ROOT / "RELEARN_RESULTS.md"

METHODS = [
    ("Fine-tuning naif",              CFG / "N48_relearn_naive.yaml"),
    ("EWC-DR",                        CFG / "N48_relearn_ewc.yaml"),
    ("Experience replay",             CFG / "N48_relearn_replay.yaml"),
    ("EWC-DR dan experience replay",  CFG / "N48_relearn_ewc_replay.yaml"),
]

ENV = {**os.environ, "PYTHONPATH": "src"}

# ── Drive setup (used only with --setup) ────────────────────────────────────
DRIVE_ROOT = "gdrive-mesach:tugas-akhir"
RELEARN_BUNDLE = "relearn_bundle.tar.gz"               # at DRIVE_ROOT/ (CPG + parquet)
# Fill these with the exact archive names on Drive (leave "" to skip — provide the file yourself):
MEGAVUL_PT_ARCHIVE = ""   # e.g. "lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_..._s1600r42_lazy.tar.gz" (DRIVE_ROOT/)
TASKA_CKPT_ARCHIVE = ""   # e.g. "<run_id>_checkpoints.zip" at DRIVE_ROOT/checkpoints/  (unzips to checkpoints/<run_id>/best_*.pt)


def sh(args: list[str]) -> None:
    print("\n+ " + " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True, cwd=str(ROOT), env=ENV)


def _rclone(src: str, dst: str) -> None:
    sh(["rclone", "copy", src, dst, "--progress"])


def _extract(archive: Path, into: Path = ROOT) -> None:
    if archive.suffix == ".zip":
        sh(["unzip", "-o", "-q", str(archive), "-d", str(into)])
    else:  # .tar.gz / .tgz
        sh(["bash", "-c", f'tar -I "$(command -v pigz || echo gzip)" -xf "{archive}" -C "{into}"'])


def setup() -> None:
    """Download + extract prerequisites from Drive (idempotent — skips if present)."""
    # 1. relearn CPG bundle -> data/raw/relearn/ + parquet
    if not (ROOT / "data" / "raw" / "relearn").exists():
        _rclone(f"{DRIVE_ROOT}/{RELEARN_BUNDLE}", str(ROOT))
        _extract(ROOT / RELEARN_BUNDLE)
    else:
        print("relearn CPG already present, skip download.")
    # 2. MegaVul task-A .pt (importance + replay buffer)
    if MEGAVUL_PT_ARCHIVE and not list((ROOT / "data" / "processed").glob("lm_dataset_megavul_multiclass*")):
        _rclone(f"{DRIVE_ROOT}/{MEGAVUL_PT_ARCHIVE}", str(ROOT))
        _extract(ROOT / MEGAVUL_PT_ARCHIVE)
    elif not MEGAVUL_PT_ARCHIVE:
        print("WARN: MEGAVUL_PT_ARCHIVE not set — ensure megavul .pt is present for importance + replay.")
    # 3. task-A checkpoint
    if TASKA_CKPT_ARCHIVE and not TASKA_CKPT.exists():
        _rclone(f"{DRIVE_ROOT}/checkpoints/{TASKA_CKPT_ARCHIVE}", str(ROOT / "checkpoints"))
        _extract(ROOT / "checkpoints" / TASKA_CKPT_ARCHIVE, ROOT / "checkpoints")
        print(f"NOTE: ensure the extracted best_*.pt is at {TASKA_CKPT} (rename/symlink if needed).")
    elif not TASKA_CKPT_ARCHIVE:
        print(f"WARN: TASKA_CKPT_ARCHIVE not set — ensure {TASKA_CKPT} exists.")


def f1_macro(results_dir: Path) -> float:
    m = json.loads((results_dir / "metrics_summary.json").read_text())
    return float(m["function_level"]["f1_macro"])


def newest_train_dir(after: float) -> Path:
    """Newest results/<ts>_lmgat_codebert_multiclass dir from a TRAIN run (excludes eval dirs)."""
    best, bt = None, after
    for d in RESULTS.glob("*_lmgat_codebert_multiclass"):
        if d.name.startswith("rl_"):          # skip our eval output dirs
            continue
        ms = d / "metrics_summary.json"
        if ms.exists() and ms.stat().st_mtime > after and ms.stat().st_mtime >= bt:
            best, bt = d, ms.stat().st_mtime
    if best is None:
        raise RuntimeError(f"No new train results dir after t={after}")
    return best


def eval_ckpt(ckpt: Path, config: Path, tag: str) -> float:
    """Evaluate a checkpoint on the dataset defined by `config`; results -> results/<tag>."""
    ed = CKPTS / tag
    ed.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, ed / "best_model.pt")
    sh([sys.executable, "-m", "gnn_vuln.evaluate",
        "--checkpoint", ed / "best_model.pt", "--config", config])
    return f1_macro(RESULTS / tag)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run the relearn (domain-IL) continual learning experiment")
    ap.add_argument("--setup", action="store_true",
                    help="download + extract prerequisites from Drive before running")
    args = ap.parse_args()
    if args.setup:
        setup()

    if not TASKA_CKPT.exists():
        sys.exit(f"Missing task-A checkpoint: {TASKA_CKPT}")

    # 0. EWC importance on task-A (compute_only -> exits after saving cache)
    sh([sys.executable, "-m", "gnn_vuln.train", "--config", MEGAVUL_CFG])

    # Baseline (Sebelum pembaruan): task-A model evaluated on both test sets.
    taskA_before = eval_ckpt(TASKA_CKPT, MEGAVUL_CFG, "rl_taskA_before")
    taskB_before = eval_ckpt(TASKA_CKPT, RELEARN_CFG, "rl_taskB_before")
    rows = [("Sebelum pembaruan", taskA_before, taskB_before, None)]

    # Each method: train on task-B, then eval the trained model on task-A.
    for label, cfg in METHODS:
        t0 = time.time()
        sh([sys.executable, "-m", "gnn_vuln.train", "--config", cfg])
        rdir = newest_train_dir(t0)
        taskB = f1_macro(rdir)                                   # task-B test (from training)
        ckpt = next((CKPTS / rdir.name).glob("best_*.pt"))
        taskA = eval_ckpt(ckpt, MEGAVUL_CFG, f"rl_taskA_{rdir.name}")
        rows.append((label, taskA, taskB, taskA_before - taskA))  # forgetting = drop on task-A

    # Write RELEARN_RESULTS.md
    md = [
        "# Hasil Continual Learning (relearn task-B = BigVul + TitanVul)",
        "",
        "Task-A = MegaVul 26-kelas. Task-B = relearn 26-kelas. Macro F1 pada test masing-masing.",
        "Forgetting = Macro F1 task-A sebelum pembaruan dikurangi sesudah (makin kecil makin baik).",
        "",
        "| Metode | Macro F1 task-A | Macro F1 task-B | Forgetting ↓ |",
        "|---|---|---|---|",
    ]
    for label, ta, tb, fg in rows:
        fg_s = "—" if fg is None else f"{fg:+.3f}"
        md.append(f"| {label} | {ta:.3f} | {tb:.3f} | {fg_s} |")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + OUT_MD.read_text())

    # Upload
    subprocess.run(["rclone", "copy", str(OUT_MD), DRIVE, "--progress"], check=False)
    print(f"\nUploaded {OUT_MD.name} -> {DRIVE}")


if __name__ == "__main__":
    main()
