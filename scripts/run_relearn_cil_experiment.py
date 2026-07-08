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
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CKPTS = ROOT / "checkpoints"
RELEARN = ROOT / "configs" / "ablation" / "relearn"
CIL = RELEARN / "cil"
SUF = "_nine" if os.environ.get("RELEARN_NINE") else ""  # unixcoder-base-nine backbone for consistency with bab-4
# Architecture selector — see run_relearn_experiment.py. graph = original N48; hybrid ml5120; seq ml1024.
ARCH = os.environ.get("RELEARN_ARCH", "graph").lower()   # graph | hybrid | seq
_ARCH = {
    "graph":  {"prefix": "N48", "taska": "n48_taskA", "run": "lmgat_codebert", "label": "berbasis graph (N48)",
               "ckpt": {42: "20260707_202747", 1: "20260707_204341", 2: "20260707_205826"}},
    "hybrid": {"prefix": "O1",  "taska": "o1_taskA",  "run": "lmgat_codebert", "label": "hibrida graph-LM (O1)",
               "ckpt": {42: "20260707_201128", 1: "20260707_234710", 2: "20260708_015442"}},
    "seq":    {"prefix": "S1",  "taska": "s1_taskA",  "run": "lmgat_seqgnn",   "label": "sekuensial (S1)",
               "ckpt": {42: "20260707_211550", 1: "20260707_214418", 2: "20260707_222608"}},
}[ARCH]
if ARCH != "graph" and not SUF:
    sys.exit("hybrid/seq continual only wired for the nine backbone — set RELEARN_NINE=1")
PREFIX, RUN_ARCH, ARCH_LABEL = _ARCH["prefix"], _ARCH["run"], _ARCH["label"]
ARCH_TAG = "" if ARCH == "graph" else f"_{ARCH}"
_DS_ML = "ml5120" if ARCH == "hybrid" else "ml1024"
_EVAL36 = "cil_taskA_eval" if ARCH == "graph" else f"{PREFIX}_cil_taskA_eval"
IMPORTANCE_CFG = RELEARN / f"{PREFIX}_taskA_importance{SUF}.yaml"   # 26-class task-A importance (reused)
TASKA_EVAL26 = RELEARN / f"{PREFIX}_taskA_importance{SUF}.yaml"     # 26-class megavul eval (baseline before)
TASKA_EVAL36 = CIL / f"{_EVAL36}{SUF}.yaml"             # 36-class megavul eval (after)
TASKB_EVAL = CIL / f"{PREFIX}_cil_naive{SUF}.yaml"      # 36-class megavul_cil eval (task-B)
TASKA_CKPT = CKPTS / _ARCH["taska"] / "best_model.pt"
DRIVE = "gdrive-mesach:tugas-akhir/results/"
OUT_MD = ROOT / (f"RELEARN_CIL_RESULTS{SUF}{ARCH_TAG}.md")
SEED = None   # set from --seed in main; overrides train.seed (split + init) for multi-seed variance


def _seed_args() -> list:
    # nine: fully per-seed. The task-A backbone for seed N is the classification seed-N checkpoint,
    # trained on split N, so eval uses split N too — matched (no leak) and multi-seed varies split+init.
    s = SEED if SEED is not None else 42
    if SUF:
        return ["--split-seed", str(s), "--seed", str(s)]
    # base: one fixed backbone -> lock split at 42, seed varies init only
    a = ["--split-seed", "42"]
    if SEED is not None:
        a += ["--seed", str(SEED)]
    return a

# Method set — env RELEARN_METHODS subsets/reorders (e.g. "replay" to fail-fast on the OOM).
# Default = all four, canonical order.
_METHOD_ALL = [
    ("naive",      "Fine-tuning naif",             f"{PREFIX}_cil_naive{SUF}.yaml"),
    ("ewc",        "EWC-DR",                       f"{PREFIX}_cil_ewc{SUF}.yaml"),
    ("replay",     "Experience replay",            f"{PREFIX}_cil_replay{SUF}.yaml"),
    ("ewc_replay", "EWC-DR dan experience replay", f"{PREFIX}_cil_ewc_replay{SUF}.yaml"),
]
_msel = os.environ.get("RELEARN_METHODS", "").strip()
_mmap = {k: (lbl, CIL / f) for k, lbl, f in _METHOD_ALL}
METHODS = ([_mmap[k.strip()] for k in _msel.split(",")] if _msel
           else [(lbl, CIL / f) for _, lbl, f in _METHOD_ALL])

# Trained method checkpoints already on Drive (run_ids from RELEARN_CIL_RESULTS.md) — used by
# --reeval to re-score the saved models with the current evaluate.py, no retraining.
REEVAL_CKPTS = [
    ("Fine-tuning naif",              "20260619_213909_lmgat_codebert_multiclass"),
    ("EWC-DR",                        "20260619_215528_lmgat_codebert_multiclass"),
    ("Experience replay",             "20260619_220817_lmgat_codebert_multiclass"),
    ("EWC-DR dan experience replay",  "20260619_222906_lmgat_codebert_multiclass"),
]

ENV = {**os.environ, "PYTHONPATH": "src"}
ENV.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # seq two-stage + replay fragments VRAM
ENV.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")       # new name (torch deprecates the old)

# ── Drive setup (used only with --setup) ────────────────────────────────────
DRIVE_ROOT = "gdrive-mesach:tugas-akhir"
MEGAVUL_PT_DIR = "data/processed/megavul"
CIL_PT_DIR = "data/processed/relearn"   # both continual task-B datasets live under relearn/ on Drive
if ARCH == "hybrid":   # ml5120: megavul nine already on Drive; cil built + uploaded via build_hybrid_ml5120_datasets.py
    MEGAVUL_PT_ARCHIVE = "lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml5120_f40f2e964_s1600r42_lazy_20260613_203058.tar.gz"
    CIL_PT_ARCHIVE = "lm_dataset_megavul_cil_multiclass_unixcoder-base-nine_ft_ml5120_lazy.tar.gz"
else:
    MEGAVUL_PT_ARCHIVE = ("lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42_lazy_20260613_195029.tar.gz" if SUF
                          else "lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42_lazy_20260513_153956.tar.gz")
    CIL_PT_ARCHIVE = ("lm_dataset_megavul_cil_multiclass_unixcoder-base-nine_ft_ml1024_lazy.tar.gz" if SUF
                      else "lm_dataset_megavul_cil_multiclass_unixcoder-base_ft_ml1024_lazy.tar.gz")
# Task-A backbone = the classification 26-class model reused directly. For nine we use the SAME
# per-seed classification checkpoints (each trained on its own split) so the CIL split follows the
# seed with no leak. Base keeps one backbone (split then locked at 42).
TASKA_CKPT_ARCHIVE_BASE = "20260606_163818_lmgat_codebert_multiclass_checkpoints.zip"  # graph base only


def taska_archive() -> str:
    # Per-seed nine backbone = the arch's own 26-class classification checkpoint (each on its own split).
    if SUF:
        run_id = _ARCH["ckpt"][SEED if SEED is not None else 42]
        return f"{run_id}_{RUN_ARCH}_multiclass_checkpoints.zip"
    return TASKA_CKPT_ARCHIVE_BASE


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


def _ds_base_name(archive: str) -> str:
    """Strip the _lazy/_inmemory marker (with or without timestamp) + .tar.gz."""
    return re.sub(r'(_(lazy|inmemory)(_\d{8}_\d{6})?)?(_\d{8}(_\d{6})?)?\.tar\.gz$', '', archive)


def _newest_archive(remote_dir: str, ds_name: str) -> Optional[str]:
    """Newest tar on Drive for ds_name (patched, timestamped tar wins by sort), or None."""
    out = subprocess.run(["rclone", "lsf", f"{DRIVE_ROOT}/{remote_dir}"],
                         capture_output=True, text=True).stdout.splitlines()
    pat = re.compile(rf"^{re.escape(ds_name)}(\.tar\.gz|_(lazy|inmemory)(_\d{{8}}_\d{{6}})?\.tar\.gz|_\d{{8}}(_\d{{6}})?\.tar\.gz)$")
    cands = sorted(f for f in out if pat.match(f))
    return cands[-1] if cands else None


def _pull_dataset(remote_dir: str, archive_hint: str, proc: Path) -> None:
    """Download + extract the NEWEST (patched) tar for this dataset, once per pod.
    Clears any stale extract first, then marks done so per-seed --setup skips re-extract."""
    ds = _ds_base_name(archive_hint)
    archive = _newest_archive(remote_dir, ds) or archive_hint
    marker = proc / f".{archive}.extracted"
    if marker.exists():
        print(f"{ds}: newest ({archive}) already extracted, skip")
        return
    for p in list(proc.glob(f"{ds}_graphs")) + list(proc.glob(f"{ds}_meta.pt")) + list(proc.glob(f"{ds}.pt")):
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    proc.mkdir(parents=True, exist_ok=True)
    _rclone(f"{DRIVE_ROOT}/{remote_dir}/{archive}", str(proc))
    _extract(proc / archive, proc)
    (proc / archive).unlink(missing_ok=True)
    marker.touch()
    print(f"{ds}: extracted newest {archive}")


def setup() -> None:
    """Download + extract prerequisites from Drive (idempotent)."""
    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    # 1. megavul task-A .pt (importance + replay buffer) — newest/patched
    _pull_dataset(MEGAVUL_PT_DIR, MEGAVUL_PT_ARCHIVE, proc)
    # 2. cil task-B .pt (newest/patched) + label patch (26..35)
    _pull_dataset(CIL_PT_DIR, CIL_PT_ARCHIVE, proc)
    sh([sys.executable, "scripts/patch_cil_labels.py"])   # idempotent — no-op if already 36-class
    # 2b. cil cwe_vocab.json — the dataset constructor requires it under data/raw/<source>
    # even when loading the prebuilt .pt (existence guard runs before the process-skip).
    cil_raw = ROOT / "data" / "raw" / "megavul_cil"
    cil_raw.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CIL / "megavul_cil_cwe_vocab.json", cil_raw / "cwe_vocab.json")
    print(f"placed cil cwe_vocab.json -> {cil_raw / 'cwe_vocab.json'}")
    # 3. task-A backbone → checkpoints/n48_taskA/best_model.pt. Per-seed for nine, so ALWAYS
    #    (re)point to THIS seed's backbone (it differs across seeds on the same pod). The zip is
    #    cached by run_id dir, only the copy into n48_taskA/ is redone each seed.
    arch = taska_archive()
    run_id = arch.replace("_checkpoints.zip", "")
    if not list((ROOT / "checkpoints" / run_id).glob("best_*.pt")):
        _rclone(f"{DRIVE_ROOT}/checkpoints/{arch}", str(ROOT / "checkpoints"))
        _extract(ROOT / "checkpoints" / arch, ROOT)
    src = next((ROOT / "checkpoints" / run_id).glob("best_*.pt"))
    TASKA_CKPT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, TASKA_CKPT)
    print(f"task-A ckpt {src.name} ({run_id}) -> {TASKA_CKPT}")


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
    # macro over classes PRESENT in the merged y_true (F1 undefined for a class with no test
    # samples); labels= keeps the denominator independent of stray predictions into absent classes.
    return (float(f1_score(yt, yp, average="macro", labels=sorted(set(yt)), zero_division=0)),
            float(accuracy_score(yt, yp)))


def newest_train_dir(after: float) -> Path:
    best, bt = None, after
    for d in RESULTS.glob(f"*_{RUN_ARCH}_multiclass"):
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
        "--checkpoint", ed / "best_model.pt", "--config", config] + _seed_args())
    return f1_macro(RESULTS / tag)


def upload_ckpt(run_id: str) -> None:
    """Zip the full checkpoints/<run_id>/ dir as <run_id>_checkpoints.zip → Drive checkpoints/
    (standard convention, inner path checkpoints/<run_id>/) for re-eval or resume."""
    z = f"{run_id}_checkpoints.zip"
    sh(["bash", "-c", f'cd "{ROOT}" && rm -f "{z}" && zip -q -r "{z}" checkpoints/{run_id}'])
    subprocess.run(["rclone", "copy", str(ROOT / z), f"{DRIVE_ROOT}/checkpoints/", "--progress"], check=False)
    (ROOT / z).unlink(missing_ok=True)
    print(f"Uploaded {z} -> {DRIVE_ROOT}/checkpoints/")


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
    ap = argparse.ArgumentParser(description="Run the class-incremental (CIL) continual learning experiment")
    ap.add_argument("--setup", action="store_true",
                    help="download + extract + patch prerequisites from Drive before running")
    ap.add_argument("--reeval", action="store_true",
                    help="re-evaluate the saved method checkpoints (REEVAL_CKPTS) with the current "
                         "evaluate.py, no retraining — recomputes the table and rewrites the md")
    ap.add_argument("--seed", type=int, default=None,
                    help="override train.seed for a multi-seed variance run (suffixes the output md)")
    args = ap.parse_args()
    global SEED
    SEED = args.seed
    out_md = OUT_MD if SEED is None else OUT_MD.with_name(f"RELEARN_CIL_RESULTS{SUF}{ARCH_TAG}_s{SEED}.md")
    if args.setup or args.reeval:
        setup()

    if not TASKA_CKPT.exists():
        sys.exit(f"Missing task-A checkpoint: {TASKA_CKPT}")

    # Baseline (Sebelum pembaruan): task-A 26-class model on task-A test.
    # task-B is N.A. — the 26-class model has no head for the new classes.
    taskA_before = eval_ckpt(TASKA_CKPT, TASKA_EVAL26, "rlcil_taskA_before")
    a1_acc = accuracy_of(RESULTS / "rlcil_taskA_before")   # A_1: acc after task-A (initial), for A_avg
    rows = [("Sebelum pembaruan", taskA_before, None, None, None, None, None)]
    ckpt_map = []   # (label, run_id) — uploaded ckpts, for re-eval without retraining

    if args.reeval:
        # Re-score the saved method checkpoints with the current evaluate.py (no retraining).
        for label, run_id in REEVAL_CKPTS:
            _fetch_method_ckpt(run_id)
            ckpt = next((CKPTS / run_id).glob("best_*.pt"))
            taskB = eval_ckpt(ckpt, TASKB_EVAL,   f"rlcil_taskB_{run_id}")   # 10 new classes
            taskA = eval_ckpt(ckpt, TASKA_EVAL36, f"rlcil_taskA_{run_id}")   # 26 old classes
            alast_f1, alast_acc = combined_alast(f"rlcil_taskA_{run_id}", f"rlcil_taskB_{run_id}")
            a_avg = (a1_acc + alast_acc) / 2.0
            rows.append((label, taskA, taskB, taskA_before - taskA, alast_f1, alast_acc, a_avg))
            ckpt_map.append((label, run_id))
    else:
        # 0. EWC importance on task-A (26-class, compute_only -> exits after saving cache)
        sh([sys.executable, "-m", "gnn_vuln.train", "--config", IMPORTANCE_CFG] + _seed_args())
        # Each method: train on task-B (36-class), eval on task-B (new) and task-A (old).
        for label, cfg in METHODS:
            t0 = time.time()
            sh([sys.executable, "-m", "gnn_vuln.train", "--config", cfg] + _seed_args())
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
        f"Model: arsitektur {ARCH_LABEL}, backbone task-A = checkpoint klasifikasi 26 kelas {taska_archive().replace('_checkpoints.zip','')}, {'per-seed' if SUF else 'tetap seed 42'}.",
        "Satu arsitektur, head diperluas 26->36 (load expandable) untuk menampung 10 kelas baru.",
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
        f"4. Split test {'mengikuti seed (per-seed)' if SUF else 'seed 42'} (80/10/10); importance EWC-DR dan buffer replay dari train task-A.",
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
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + out_md.read_text())

    subprocess.run(["rclone", "copy", str(out_md), DRIVE, "--progress"], check=False)
    print(f"\nUploaded {out_md.name} -> {DRIVE}")


if __name__ == "__main__":
    main()
