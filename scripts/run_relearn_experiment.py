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
SUF = "_nine" if os.environ.get("RELEARN_NINE") else ""   # unixcoder-base-nine backbone for consistency with bab-4
# Architecture selector — the continual protocol is identical across the three proposed archs;
# only the config prefix + task-A backbone + run-dir arch name change. graph = original N48.
# hybrid uses func_max_length 5120: its .pt is auto-re-tokenized from the ml1024 nine base on the
# pod (same node-LM), so all three archs share the SAME ml1024 dataset download.
ARCH = os.environ.get("RELEARN_ARCH", "graph").lower()    # graph | hybrid | seq
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
_DS_ML = "ml5120" if ARCH == "hybrid" else "ml1024"   # hybrid = live LM 5120-token func windows
MEGAVUL_CFG = CFG / f"{PREFIX}_taskA_importance{SUF}.yaml"  # data.source=megavul (task-A, 26-class)
RELEARN_CFG = CFG / f"{PREFIX}_relearn_naive{SUF}.yaml"     # data.source=relearn (task-B, any relearn cfg)
TASKA_CKPT = CKPTS / _ARCH["taska"] / "best_model.pt"
DRIVE = "gdrive-mesach:tugas-akhir/results/"
OUT_MD = ROOT / (f"RELEARN_RESULTS{SUF}{ARCH_TAG}.md")
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

# Method set — env RELEARN_METHODS subsets/reorders (e.g. "replay" to fail-fast on the OOM,
# or "replay,ewc_replay,naive,ewc"). Default = all four, canonical order.
_METHOD_ALL = [
    ("naive",      "Fine-tuning naif",             f"{PREFIX}_relearn_naive{SUF}.yaml"),
    ("ewc",        "EWC-DR",                       f"{PREFIX}_relearn_ewc{SUF}.yaml"),
    ("replay",     "Experience replay",            f"{PREFIX}_relearn_replay{SUF}.yaml"),
    ("ewc_replay", "EWC-DR dan experience replay", f"{PREFIX}_relearn_ewc_replay{SUF}.yaml"),
]
_msel = os.environ.get("RELEARN_METHODS", "").strip()
_mmap = {k: (lbl, CFG / f) for k, lbl, f in _METHOD_ALL}
METHODS = ([_mmap[k.strip()] for k in _msel.split(",")] if _msel
           else [(lbl, CFG / f) for _, lbl, f in _METHOD_ALL])

# Trained method checkpoints already on Drive (run_ids from RELEARN_RESULTS.md) — used by
# --reeval to re-score the saved models with the current evaluate.py, no retraining.
REEVAL_CKPTS = [
    ("Fine-tuning naif",              "20260619_200234_lmgat_codebert_multiclass"),
    ("EWC-DR",                        "20260619_200708_lmgat_codebert_multiclass"),
    ("Experience replay",             "20260619_201757_lmgat_codebert_multiclass"),
    ("EWC-DR dan experience replay",  "20260619_202816_lmgat_codebert_multiclass"),
]

ENV = {**os.environ, "PYTHONPATH": "src"}
ENV.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # seq two-stage + replay fragments VRAM
ENV.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")       # new name (torch deprecates the old)

# ── Drive setup (used only with --setup) ────────────────────────────────────
DRIVE_ROOT = "gdrive-mesach:tugas-akhir"
RELEARN_BUNDLE = "relearn_bundle.tar.gz"               # at DRIVE_ROOT/ (CPG + parquet)
# Task-A = N48 26-class jknet (ABLATION_GNN_ONLY.md run 20260606_163818).
MEGAVUL_PT_DIR = "data/processed/megavul"              # Drive subdir holding the .pt tar
if ARCH == "hybrid":   # ml5120 nine megavul already on Drive (H10/O1 trained on it)
    MEGAVUL_PT_ARCHIVE = "lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml5120_f40f2e964_s1600r42_lazy_20260613_203058.tar.gz"
else:
    MEGAVUL_PT_ARCHIVE = ("lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42_lazy_20260613_195029.tar.gz" if SUF
                          else "lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42_lazy_20260513_153956.tar.gz")
# Task-A backbone = the classification 26-class model reused directly. For nine we use the SAME
# per-seed classification checkpoints (each trained on its own split) so the split follows the
# seed with no leak. Base keeps one backbone (split then locked at 42).
TASKA_CKPT_ARCHIVE_BASE = "20260606_163818_lmgat_codebert_multiclass_checkpoints.zip"  # graph base only


def taska_archive() -> str:
    # Per-seed nine backbone = the arch's own 26-class classification checkpoint (each trained on
    # its own split → eval follows the seed, no leak). Run ids in _ARCH["ckpt"], newest from ABLATION.
    if SUF:
        run_id = _ARCH["ckpt"][SEED if SEED is not None else 42]
        return f"{run_id}_{RUN_ARCH}_multiclass_checkpoints.zip"
    return TASKA_CKPT_ARCHIVE_BASE
# Prebuilt relearn .pt (vocab-aligned) — fill after the first upload to skip rebuilding on the pod.
RELEARN_PT_DIR = "data/processed/relearn"
if ARCH == "hybrid":   # ml5120 relearn built + uploaded via scripts/build_hybrid_ml5120_datasets.py
    RELEARN_PT_ARCHIVE = "lm_dataset_relearn_multiclass_unixcoder-base-nine_ft_ml5120_f40f2e964_s1600r42.tar.gz"
else:
    RELEARN_PT_ARCHIVE = ("lm_dataset_relearn_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42.tar.gz" if SUF
                          else "lm_dataset_relearn_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42.tar.gz")


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
    if not list(proc.glob(f"lm_dataset_megavul_multiclass*{_DS_ML}*")):
        proc.mkdir(parents=True, exist_ok=True)
        _rclone(f"{DRIVE_ROOT}/{MEGAVUL_PT_DIR}/{MEGAVUL_PT_ARCHIVE}", str(proc))
        _extract(proc / MEGAVUL_PT_ARCHIVE, proc)
    else:
        print("megavul .pt already present, skip download.")
    # 3. task-A backbone -> checkpoints/n48_taskA/best_model.pt. Per-seed for nine, so ALWAYS
    #    (re)point to THIS seed's backbone (it differs across seeds on the same pod).
    arch = taska_archive()
    run_id = arch.replace("_checkpoints.zip", "")
    if not list((ROOT / "checkpoints" / run_id).glob("best_*.pt")):
        _rclone(f"{DRIVE_ROOT}/checkpoints/{arch}", str(ROOT / "checkpoints"))
        _extract(ROOT / "checkpoints" / arch, ROOT)   # inner path checkpoints/<run_id>/best_*.pt
    src = next((ROOT / "checkpoints" / run_id).glob("best_*.pt"))
    TASKA_CKPT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, TASKA_CKPT)
    print(f"task-A ckpt {src.name} ({run_id}) -> {TASKA_CKPT}")
    # 4. prebuilt relearn .pt (optional) — skips the GPU rebuild on the pod
    if RELEARN_PT_ARCHIVE and not list(proc.glob(f"lm_dataset_relearn_multiclass*{_DS_ML}*")):
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
    for d in RESULTS.glob(f"*_{RUN_ARCH}_multiclass"):
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
        "--checkpoint", ed / "best_model.pt", "--config", config] + _seed_args())
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
    ap.add_argument("--seed", type=int, default=None,
                    help="override train.seed for a multi-seed variance run (suffixes the output md)")
    args = ap.parse_args()
    global SEED
    SEED = args.seed
    out_md = OUT_MD if SEED is None else OUT_MD.with_name(f"RELEARN_RESULTS{SUF}{ARCH_TAG}_s{SEED}.md")
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
        sh([sys.executable, "-m", "gnn_vuln.train", "--config", MEGAVUL_CFG] + _seed_args())
        # Each method: train on task-B, then eval the trained model on task-A.
        for label, cfg in METHODS:
            t0 = time.time()
            sh([sys.executable, "-m", "gnn_vuln.train", "--config", cfg] + _seed_args())
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
        f"Model: arsitektur {ARCH_LABEL}, 26-kelas, backbone task-A = {taska_archive().replace('_checkpoints.zip','')}, {'per-seed' if SUF else 'tetap seed 42'}.",
        "Satu arsitektur dipakai di semua baris. Setiap metode melanjutkan pelatihan model",
        "task-A yang sama pada task-B. Jenis: domain-incremental (26 kelas tetap, domain data berganti).",
        "",
        "Task-A = MegaVul 26-kelas. Task-B = relearn 26-kelas. Macro F1 pada test masing-masing.",
        "Sebelum pembaruan = model task-A apa adanya (tanpa retraining) dievaluasi pada kedua test.",
        "Forgetting = Macro F1 task-A sebelum pembaruan dikurangi sesudah (makin kecil makin baik).",
        "",
        "Urutan dan asal data:",
        f"1. Task-A (MegaVul): N48 dilatih lebih dulu pada MegaVul top-25 CWE plus benign (26 kelas, maks 1600 per kelas, {'seed mengikuti run' if SUF else 'seed 42'}).",
        "2. Task-B (relearn): dibangun dari BigVul plus TitanVul, vuln top-25 dideduplikasi terhadap MegaVul dan antar keduanya, ditambah benign. Label dipetakan ke vocab kanonik task-A agar id kelas selaras.",
        "3. Pelatihan kontinual: model mulai dari bobot task-A, lalu dilanjutkan pada task-B. Urutan = MegaVul lebih dulu, baru relearn.",
        f"4. Split test {'mengikuti seed (per-seed)' if SUF else 'seed 42'} (80/10/10); importance EWC-DR dan buffer replay diambil dari split train task-A tanpa kebocoran ke test.",
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
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + out_md.read_text())

    # Upload
    subprocess.run(["rclone", "copy", str(out_md), DRIVE, "--progress"], check=False)
    print(f"\nUploaded {out_md.name} -> {DRIVE}")


if __name__ == "__main__":
    main()
