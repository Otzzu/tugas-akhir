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


def sh(args: list[str]) -> None:
    print("\n+ " + " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True, cwd=str(ROOT), env=ENV)


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
