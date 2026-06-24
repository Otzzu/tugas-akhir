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
  PYTHONPATH=src python scripts/run_relearn_experiment.py --setup   # download + extract from Drive, then run
  PYTHONPATH=src python scripts/run_relearn_experiment.py           # prerequisites already on the pod
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

# Trained method checkpoints already on Drive (run_ids from RELEARN_RESULTS.md) — used by
# --reeval to re-score the saved models with the current evaluate.py, no retraining.
REEVAL_CKPTS = [
    ("Fine-tuning naif",              "20260619_200234_lmgat_codebert_multiclass"),
    ("EWC-DR",                        "20260619_200708_lmgat_codebert_multiclass"),
    ("Experience replay",             "20260619_201757_lmgat_codebert_multiclass"),
    ("EWC-DR dan experience replay",  "20260619_202816_lmgat_codebert_multiclass"),
]

ENV = {**os.environ, "PYTHONPATH": "src"}

# ── Drive setup (used only with --setup) ────────────────────────────────────
DRIVE_ROOT = "gdrive-mesach:tugas-akhir"
RELEARN_BUNDLE = "relearn_bundle.tar.gz"               # at DRIVE_ROOT/ (CPG + parquet)
# Task-A = N48 26-class jknet (ABLATION_GNN_ONLY.md run 20260606_163818).
MEGAVUL_PT_DIR = "data/processed/megavul"              # Drive subdir holding the .pt tar
MEGAVUL_PT_ARCHIVE = "lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42_lazy_20260513_153956.tar.gz"
TASKA_CKPT_ARCHIVE = "20260606_163818_lmgat_codebert_multiclass_checkpoints.zip"   # DRIVE_ROOT/checkpoints/ -> checkpoints/<run_id>/best_*.pt
# Prebuilt relearn .pt (vocab-aligned) — fill after the first upload to skip rebuilding on the pod.
RELEARN_PT_DIR = "data/processed/relearn"
RELEARN_PT_ARCHIVE = "lm_dataset_relearn_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42.tar.gz"


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
    _align_relearn_vocab()   # canonical vocab BEFORE any .pt download so the guard keeps it
    # 2. MegaVul task-A .pt (importance + replay buffer) -> data/processed/
    proc = ROOT / "data" / "processed"
    if not list(proc.glob("lm_dataset_megavul_multiclass*")):
        proc.mkdir(parents=True, exist_ok=True)
        _rclone(f"{DRIVE_ROOT}/{MEGAVUL_PT_DIR}/{MEGAVUL_PT_ARCHIVE}", str(proc))
        _extract(proc / MEGAVUL_PT_ARCHIVE, proc)
    else:
        print("megavul .pt already present, skip download.")
    # 3. task-A checkpoint -> checkpoints/n48_taskA/best_model.pt
    if not TASKA_CKPT.exists():
        _rclone(f"{DRIVE_ROOT}/checkpoints/{TASKA_CKPT_ARCHIVE}", str(ROOT / "checkpoints"))
        _extract(ROOT / "checkpoints" / TASKA_CKPT_ARCHIVE, ROOT)   # inner path checkpoints/<run_id>/best_*.pt
        run_id = TASKA_CKPT_ARCHIVE.replace("_checkpoints.zip", "")
        src = next((ROOT / "checkpoints" / run_id).glob("best_*.pt"))
        TASKA_CKPT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, TASKA_CKPT)
        print(f"task-A ckpt {src.name} -> {TASKA_CKPT}")
    else:
        print("task-A ckpt already present, skip download.")
    # 4. prebuilt relearn .pt (optional) — skips the GPU rebuild on the pod
    if RELEARN_PT_ARCHIVE and not list(proc.glob("lm_dataset_relearn_multiclass*")):
        _rclone(f"{DRIVE_ROOT}/{RELEARN_PT_DIR}/{RELEARN_PT_ARCHIVE}", str(proc))
        _extract(proc / RELEARN_PT_ARCHIVE, proc)
    elif not RELEARN_PT_ARCHIVE:
        print("RELEARN_PT_ARCHIVE not set — relearn .pt will build from CPG on first use.")


def f1_macro(results_dir: Path) -> float:
    m = json.loads((results_dir / "metrics_summary.json").read_text())
    return float(m["function_level"]["f1_macro"])


def newest_train_dir(after: float) -> Path:
    """Newest results/<ts>_lmgat_codebert_multiclass dir from a TRAIN run (excludes eval dirs)."""
    best, bt = None, after
    for d in RESULTS.glob("*_lmgat_codebert_multiclass"):
        if d.name.startswith("rl_"):          # skip our eval output dirs
            continue
        ms = d / "training_summary.json"      # train writes this; metrics_summary.json comes from evaluate
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


def upload_ckpt(run_id: str) -> None:
    """Zip the full checkpoints/<run_id>/ dir (best + last) as <run_id>_checkpoints.zip and
    upload to Drive checkpoints/ — the standard checkpoint-zip convention (inner path
    checkpoints/<run_id>/), so it re-evaluates or resumes like any other model checkpoint."""
    z = f"{run_id}_checkpoints.zip"
    sh(["bash", "-c", f'cd "{ROOT}" && rm -f "{z}" && zip -q -r "{z}" checkpoints/{run_id}'])
    subprocess.run(["rclone", "copy", str(ROOT / z), f"{DRIVE_ROOT}/checkpoints/", "--progress"], check=False)
    (ROOT / z).unlink(missing_ok=True)
    print(f"Uploaded {z} -> {DRIVE_ROOT}/checkpoints/")


def _align_relearn_vocab() -> None:
    """Force the relearn dataset to use task-A's CWE->id map. Each dataset's
    cwe_vocab.json ranks CWEs by its OWN frequency; the loader reindexes top-25
    contiguously preserving that order, so relearn and megavul would get DIFFERENT
    label ids for the same CWE. The N48 checkpoint is locked to megavul's order, so
    cross-task eval is only valid when relearn adopts the canonical task-A vocab.
    Overwrites the vocab when it differs and clears any stale relearn .pt to rebuild."""
    canon = CFG / "taskA_cwe_vocab.json"
    dst = ROOT / "data" / "raw" / "relearn" / "cwe_vocab.json"
    if not canon.exists() or not dst.parent.exists():
        return
    if dst.exists() and json.loads(dst.read_text()) == json.loads(canon.read_text()):
        return
    dst.write_text(canon.read_text(), encoding="utf-8")
    cleared = 0
    for p in (ROOT / "data" / "processed").glob("lm_dataset_relearn_multiclass*"):
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        cleared += 1
    print(f"Aligned relearn vocab to task-A canonical; cleared {cleared} stale relearn .pt entries for rebuild.")


def _fetch_method_ckpt(run_id: str) -> None:
    """Download checkpoints/<run_id>_checkpoints.zip from Drive + extract (skip if already local)."""
    if list((CKPTS / run_id).glob("best_*.pt")):
        print(f"{run_id} ckpt present, skip download."); return
    z = f"{run_id}_checkpoints.zip"
    _rclone(f"{DRIVE_ROOT}/checkpoints/{z}", str(ROOT))
    _extract(ROOT / z, ROOT)            # inner path checkpoints/<run_id>/best_*.pt
    (ROOT / z).unlink(missing_ok=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run the relearn (domain-IL) continual learning experiment")
    ap.add_argument("--setup", action="store_true",
                    help="download + extract prerequisites from Drive before running")
    ap.add_argument("--reeval", action="store_true",
                    help="re-evaluate the saved method checkpoints (REEVAL_CKPTS) with the current "
                         "evaluate.py, no retraining — recomputes the table and rewrites the md")
    args = ap.parse_args()
    if args.setup or args.reeval:
        setup()
    _align_relearn_vocab()

    if not TASKA_CKPT.exists():
        sys.exit(f"Missing task-A checkpoint: {TASKA_CKPT}")

    # Baseline (Sebelum pembaruan): task-A model evaluated on both test sets.
    taskA_before = eval_ckpt(TASKA_CKPT, MEGAVUL_CFG, "rl_taskA_before")
    taskB_before = eval_ckpt(TASKA_CKPT, RELEARN_CFG, "rl_taskB_before")
    rows = [("Sebelum pembaruan", taskA_before, taskB_before, None)]
    ckpt_map = []   # (label, run_id) — uploaded ckpts, for re-eval without retraining

    if args.reeval:
        # Re-score the saved method checkpoints with the current evaluate.py (no retraining).
        for label, run_id in REEVAL_CKPTS:
            _fetch_method_ckpt(run_id)
            ckpt = next((CKPTS / run_id).glob("best_*.pt"))
            taskB = eval_ckpt(ckpt, RELEARN_CFG, f"rl_taskB_{run_id}")
            taskA = eval_ckpt(ckpt, MEGAVUL_CFG, f"rl_taskA_{run_id}")
            rows.append((label, taskA, taskB, taskA_before - taskA))
            ckpt_map.append((label, run_id))
    else:
        # 0. EWC importance on task-A (compute_only -> exits after saving cache)
        sh([sys.executable, "-m", "gnn_vuln.train", "--config", MEGAVUL_CFG])
        # Each method: train on task-B, then eval the trained model on task-A.
        for label, cfg in METHODS:
            t0 = time.time()
            sh([sys.executable, "-m", "gnn_vuln.train", "--config", cfg])
            rdir = newest_train_dir(t0)
            ckpt = next((CKPTS / rdir.name).glob("best_*.pt"))
            taskB = eval_ckpt(ckpt, RELEARN_CFG, f"rl_taskB_{rdir.name}")   # task-B test
            taskA = eval_ckpt(ckpt, MEGAVUL_CFG, f"rl_taskA_{rdir.name}")   # task-A test
            rows.append((label, taskA, taskB, taskA_before - taskA))  # forgetting = drop on task-A
            upload_ckpt(rdir.name)                                    # upload trained model for re-eval
            ckpt_map.append((label, rdir.name))

    # Write RELEARN_RESULTS.md
    md = [
        "# Hasil Continual Learning (relearn task-B = BigVul + TitanVul)",
        "",
        "Model: N48 (GNN-only, jknet, 26-kelas, checkpoint 20260606_163818, MegaVul Macro F1 0.525).",
        "Satu arsitektur N48 dipakai di semua baris. Setiap metode melanjutkan pelatihan model",
        "task-A yang sama pada task-B. Jenis: domain-incremental (26 kelas tetap, domain data berganti).",
        "",
        "Task-A = MegaVul 26-kelas. Task-B = relearn 26-kelas. Macro F1 pada test masing-masing.",
        "Sebelum pembaruan = model task-A apa adanya (tanpa retraining) dievaluasi pada kedua test.",
        "Forgetting = Macro F1 task-A sebelum pembaruan dikurangi sesudah (makin kecil makin baik).",
        "",
        "Urutan dan asal data:",
        "1. Task-A (MegaVul): N48 dilatih lebih dulu pada MegaVul top-25 CWE plus benign (26 kelas, maks 1600 per kelas, seed 42).",
        "2. Task-B (relearn): dibangun dari BigVul plus TitanVul, vuln top-25 dideduplikasi terhadap MegaVul dan antar keduanya, ditambah benign. Label dipetakan ke vocab kanonik task-A agar id kelas selaras.",
        "3. Pelatihan kontinual: model mulai dari bobot task-A, lalu dilanjutkan pada task-B. Urutan = MegaVul lebih dulu, baru relearn.",
        "4. Split test seed 42 (80/10/10); importance EWC-DR dan buffer replay diambil dari split train task-A tanpa kebocoran ke test.",
        "",
        "| Metode | Macro F1 task-A | Macro F1 task-B | Forgetting ↓ |",
        "|---|---|---|---|",
    ]
    for label, ta, tb, fg in rows:
        fg_s = "—" if fg is None else f"{fg:+.4f}"
        md.append(f"| {label} | {ta:.4f} | {tb:.4f} | {fg_s} |")
    md += [
        "",
        "Checkpoint terlatih (Drive checkpoints/, untuk re-evaluasi tanpa latih ulang):",
    ]
    for label, run_id in ckpt_map:
        md.append(f"- {label}: `{run_id}_checkpoints.zip` (config: lihat configs/ablation/relearn/)")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + OUT_MD.read_text())

    # Upload
    subprocess.run(["rclone", "copy", str(OUT_MD), DRIVE, "--progress"], check=False)
    print(f"\nUploaded {OUT_MD.name} -> {DRIVE}")


if __name__ == "__main__":
    main()
