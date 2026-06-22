"""
run_relearn_cil_experiment.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One-shot CLASS-INCREMENTAL (CIL) continual-learning driver. Same protocol as
run_relearn_experiment.py (domain-IL) but task-B introduces 10 NEW CWE classes
(megavul_cil, ids 26..35); the head expands 26→36. Runs every method, evaluates
each on task-A (MegaVul, 26 old classes) and task-B (10 new classes), computes
forgetting, writes RELEARN_CIL_RESULTS.md, uploads it to Drive.

Prereqs on the pod:
  - data/processed/lm_dataset_megavul_cil_*  (patched to labels 26..35 — patch_cil_labels.py)
  - data/processed/lm_dataset_megavul_multiclass_*  (task-A, for importance + replay buffer)
  - task-A 26-class N48 checkpoint → checkpoints/n48_taskA/best_model.pt

Run (cloud, Linux):
  PYTHONPATH=src python scripts/run_relearn_cil_experiment.py --setup
  PYTHONPATH=src python scripts/run_relearn_cil_experiment.py
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CKPTS = ROOT / "checkpoints"
RELEARN = ROOT / "configs" / "ablation" / "relearn"
CIL = RELEARN / "cil"
IMPORTANCE_CFG = RELEARN / "N48_taskA_importance.yaml"   # 26-class task-A importance (reused)
TASKA_EVAL26 = RELEARN / "N48_taskA_importance.yaml"     # 26-class megavul eval (baseline before)
TASKA_EVAL36 = CIL / "cil_taskA_eval.yaml"               # 36-class megavul eval (after)
TASKB_EVAL = CIL / "N48_cil_naive.yaml"                  # 36-class megavul_cil eval (task-B)
TASKA_CKPT = CKPTS / "n48_taskA" / "best_model.pt"
DRIVE = "gdrive-mesach:tugas-akhir/results/"
OUT_MD = ROOT / "RELEARN_CIL_RESULTS.md"

METHODS = [
    ("Fine-tuning naif",              CIL / "N48_cil_naive.yaml"),
    ("EWC-DR",                        CIL / "N48_cil_ewc.yaml"),
    ("Experience replay",             CIL / "N48_cil_replay.yaml"),
    ("EWC-DR dan experience replay",  CIL / "N48_cil_ewc_replay.yaml"),
]

ENV = {**os.environ, "PYTHONPATH": "src"}

# ── Drive setup (used only with --setup) ────────────────────────────────────
DRIVE_ROOT = "gdrive-mesach:tugas-akhir"
MEGAVUL_PT_DIR = "data/processed/megavul"
MEGAVUL_PT_ARCHIVE = "lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42_lazy_20260513_153956.tar.gz"
CIL_PT_DIR = "data/processed/relearn"   # both continual task-B datasets live under relearn/ on Drive
CIL_PT_ARCHIVE = "lm_dataset_megavul_cil_multiclass_unixcoder-base_ft_ml1024_lazy.tar.gz"
TASKA_CKPT_ARCHIVE = "20260606_163818_lmgat_codebert_multiclass_checkpoints.zip"


def sh(args: list[str]) -> None:
    print("\n+ " + " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True, cwd=str(ROOT), env=ENV)


def _rclone(src: str, dst: str) -> None:
    sh(["rclone", "copy", src, dst, "--progress"])


def _extract(archive: Path, into: Path = ROOT) -> None:
    if archive.suffix == ".zip":
        sh(["unzip", "-o", "-q", str(archive), "-d", str(into)])
    else:
        sh(["bash", "-c", f'tar -I "$(command -v pigz || echo gzip)" -xf "{archive}" -C "{into}"'])


def setup() -> None:
    """Download + extract prerequisites from Drive (idempotent)."""
    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    # 1. megavul task-A .pt (importance + replay buffer)
    if not list(proc.glob("lm_dataset_megavul_multiclass*")):
        _rclone(f"{DRIVE_ROOT}/{MEGAVUL_PT_DIR}/{MEGAVUL_PT_ARCHIVE}", str(proc))
        _extract(proc / MEGAVUL_PT_ARCHIVE, proc)
    else:
        print("megavul .pt already present, skip.")
    # 2. cil task-B .pt + label patch (26..35)
    if not list(proc.glob("lm_dataset_megavul_cil_multiclass*_meta.pt")):
        _rclone(f"{DRIVE_ROOT}/{CIL_PT_DIR}/{CIL_PT_ARCHIVE}", str(proc))
        _extract(proc / CIL_PT_ARCHIVE, proc)
    else:
        print("cil .pt already present, skip download.")
    sh([sys.executable, "scripts/patch_cil_labels.py"])   # idempotent — no-op if already 36-class
    # 2b. cil cwe_vocab.json — the dataset constructor requires it under data/raw/<source>
    # even when loading the prebuilt .pt (existence guard runs before the process-skip).
    cil_raw = ROOT / "data" / "raw" / "megavul_cil"
    cil_raw.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CIL / "megavul_cil_cwe_vocab.json", cil_raw / "cwe_vocab.json")
    print(f"placed cil cwe_vocab.json -> {cil_raw / 'cwe_vocab.json'}")
    # 3. task-A checkpoint → checkpoints/n48_taskA/best_model.pt
    if not TASKA_CKPT.exists():
        _rclone(f"{DRIVE_ROOT}/checkpoints/{TASKA_CKPT_ARCHIVE}", str(ROOT / "checkpoints"))
        _extract(ROOT / "checkpoints" / TASKA_CKPT_ARCHIVE, ROOT)
        run_id = TASKA_CKPT_ARCHIVE.replace("_checkpoints.zip", "")
        src = next((ROOT / "checkpoints" / run_id).glob("best_*.pt"))
        TASKA_CKPT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, TASKA_CKPT)
        print(f"task-A ckpt {src.name} -> {TASKA_CKPT}")
    else:
        print("task-A ckpt already present, skip.")


def f1_macro(results_dir: Path) -> float:
    m = json.loads((results_dir / "metrics_summary.json").read_text())
    return float(m["function_level"]["f1_macro"])


def accuracy_of(results_dir: Path) -> float:
    m = json.loads((results_dir / "metrics_summary.json").read_text())
    return float(m["function_level"]["accuracy"])


def combined_alast(tag_a: str, tag_b: str) -> tuple[float, float]:
    """A_last (paper EWC-DR §5.1.5): metrics over ALL seen classes after the final task.
    Merge task-A (old, 0..25) + task-B (new, 26..35) test predictions and compute
    macro-F1 + accuracy over the union (all 36). The final model is one model that holds
    both increments, so this is its overall performance on everything it should know."""
    import csv
    from sklearn.metrics import f1_score, accuracy_score
    yt, yp = [], []
    for tag in (tag_a, tag_b):
        with open(RESULTS / tag / "predictions.csv", newline="") as f:
            for row in csv.DictReader(f):
                yt.append(int(float(row["y_true"])))
                yp.append(int(float(row["y_pred"])))
    return (float(f1_score(yt, yp, average="macro", zero_division=0)),
            float(accuracy_score(yt, yp)))


def newest_train_dir(after: float) -> Path:
    best, bt = None, after
    for d in RESULTS.glob("*_lmgat_codebert_multiclass"):
        if d.name.startswith("rlcil_"):
            continue
        ms = d / "training_summary.json"
        if ms.exists() and ms.stat().st_mtime > after and ms.stat().st_mtime >= bt:
            best, bt = d, ms.stat().st_mtime
    if best is None:
        raise RuntimeError(f"No new train results dir after t={after}")
    return best


def eval_ckpt(ckpt: Path, config: Path, tag: str) -> float:
    ed = CKPTS / tag
    ed.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, ed / "best_model.pt")
    sh([sys.executable, "-m", "gnn_vuln.evaluate",
        "--checkpoint", ed / "best_model.pt", "--config", config])
    return f1_macro(RESULTS / tag)


def upload_ckpt(run_id: str) -> None:
    """Zip the full checkpoints/<run_id>/ dir as <run_id>_checkpoints.zip → Drive checkpoints/
    (standard convention, inner path checkpoints/<run_id>/) for re-eval or resume."""
    z = f"{run_id}_checkpoints.zip"
    sh(["bash", "-c", f'cd "{ROOT}" && rm -f "{z}" && zip -q -r "{z}" checkpoints/{run_id}'])
    subprocess.run(["rclone", "copy", str(ROOT / z), f"{DRIVE_ROOT}/checkpoints/", "--progress"], check=False)
    (ROOT / z).unlink(missing_ok=True)
    print(f"Uploaded {z} -> {DRIVE_ROOT}/checkpoints/")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run the class-incremental (CIL) continual learning experiment")
    ap.add_argument("--setup", action="store_true",
                    help="download + extract + patch prerequisites from Drive before running")
    args = ap.parse_args()
    if args.setup:
        setup()

    if not TASKA_CKPT.exists():
        sys.exit(f"Missing task-A checkpoint: {TASKA_CKPT}")

    # 0. EWC importance on task-A (26-class, compute_only -> exits after saving cache)
    sh([sys.executable, "-m", "gnn_vuln.train", "--config", IMPORTANCE_CFG])

    # Baseline (Sebelum pembaruan): task-A 26-class model on task-A test.
    # task-B is N.A. — the 26-class model has no head for the new classes.
    taskA_before = eval_ckpt(TASKA_CKPT, TASKA_EVAL26, "rlcil_taskA_before")
    a1_acc = accuracy_of(RESULTS / "rlcil_taskA_before")   # A_1: acc after task-A (initial), for A_avg
    rows = [("Sebelum pembaruan", taskA_before, None, None, None, None, None)]

    # Each method: train on task-B (36-class), eval on task-B (new) and task-A (old).
    ckpt_map = []   # (label, run_id) — uploaded ckpts, for re-eval without retraining
    for label, cfg in METHODS:
        t0 = time.time()
        sh([sys.executable, "-m", "gnn_vuln.train", "--config", cfg])
        rdir = newest_train_dir(t0)
        ckpt = next((CKPTS / rdir.name).glob("best_*.pt"))
        taskB = eval_ckpt(ckpt, TASKB_EVAL,    f"rlcil_taskB_{rdir.name}")   # 10 new classes
        taskA = eval_ckpt(ckpt, TASKA_EVAL36,  f"rlcil_taskA_{rdir.name}")   # 26 old classes
        # A_last: combined over all 36 seen classes (paper metric); A_avg = mean(A_1, A_last_acc)
        alast_f1, alast_acc = combined_alast(f"rlcil_taskA_{rdir.name}", f"rlcil_taskB_{rdir.name}")
        a_avg = (a1_acc + alast_acc) / 2.0
        rows.append((label, taskA, taskB, taskA_before - taskA, alast_f1, alast_acc, a_avg))
        upload_ckpt(rdir.name)                                        # upload trained model for re-eval
        ckpt_map.append((label, rdir.name))

    # Write RELEARN_CIL_RESULTS.md
    md = [
        "# Hasil Continual Learning Class-Incremental (CIL, task-B = 10 CWE baru MegaVul)",
        "",
        "Model: N48 (GNN-only, jknet, checkpoint task-A 20260606_163818, MegaVul Macro F1 0.525).",
        "Satu arsitektur N48; head diperluas 26->36 (load expandable) untuk menampung 10 kelas baru.",
        "Jenis: class-incremental (kelas BARU ditambah, bukan domain). Sesuai setting utama paper EWC-DR.",
        "",
        "Task-A = MegaVul 26 kelas lama. Task-B = 10 CWE baru (megavul_cil, id 26..35).",
        "Sebelum pembaruan task-B = N.A. (model 26 kelas belum punya head untuk kelas baru).",
        "Forgetting = Macro F1 task-A sebelum pembaruan dikurangi sesudah (makin kecil makin baik).",
        "A_last = metrik pada SEMUA kelas terlihat (gabungan task-A + task-B, 36 kelas) setelah",
        "task terakhir; A_avg = rata-rata akurasi setelah tiap task (A_1 task-A, A_2 = A_last).",
        "A_last dan A_avg mengikuti protokol CIL paper EWC-DR; akurasi seperti paper, plus macro-F1.",
        "",
        "Urutan dan asal data:",
        "1. Task-A (MegaVul): N48 dilatih pada MegaVul top-25 CWE plus benign (26 kelas).",
        "2. Task-B (megavul_cil): 10 CWE non-top25 paling banyak di MegaVul (CWE-119,190,362,264,399,400,401,189,617,835), label dipetakan ke 26..35.",
        "3. Pelatihan kontinual: mulai dari bobot task-A (head 26->36), lanjut pada task-B.",
        "4. Split test seed 42 (80/10/10); importance EWC-DR dan buffer replay dari train task-A.",
        "",
        "| Metode | F1 task-A | F1 task-B | A_last F1 (36) | A_last Acc (36) | A_avg Acc | Forgetting ↓ |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, ta, tb, fg, al_f1, al_acc, a_avg in rows:
        tb_s   = "—" if tb is None else f"{tb:.4f}"
        fg_s   = "—" if fg is None else f"{fg:+.4f}"
        alf_s  = "—" if al_f1 is None else f"{al_f1:.4f}"
        ala_s  = "—" if al_acc is None else f"{al_acc:.4f}"
        aavg_s = "—" if a_avg is None else f"{a_avg:.4f}"
        md.append(f"| {label} | {ta:.4f} | {tb_s} | {alf_s} | {ala_s} | {aavg_s} | {fg_s} |")
    md += [
        "",
        "Checkpoint terlatih (Drive checkpoints/, untuk re-evaluasi tanpa latih ulang):",
    ]
    for label, run_id in ckpt_map:
        md.append(f"- {label}: `{run_id}_checkpoints.zip` (config: lihat configs/ablation/relearn/cil/)")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + OUT_MD.read_text())

    subprocess.run(["rclone", "copy", str(OUT_MD), DRIVE, "--progress"], check=False)
    print(f"\nUploaded {OUT_MD.name} -> {DRIVE}")


if __name__ == "__main__":
    main()
