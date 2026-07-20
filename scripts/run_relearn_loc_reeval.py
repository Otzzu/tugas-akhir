"""
run_relearn_loc_reeval.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Harvest LOCALIZATION metrics for the continual-learning runs (graph N48, nine,
per-seed) by re-evaluating the saved method checkpoints — no retraining.
The two orchestrators only harvested f1_macro; gnn_vuln.evaluate already
computes the localization block at every eval, so this script re-runs the same
evals and collects Top-1/Top-5/IFA from metrics_summary.json into
RELEARN_LOC_RESULTS_nine.md (+ upload to Drive results/).

Self-contained: downloads the three .pt datasets (megavul + relearn + cil,
graph ml1024 nine, newest/patched tar), aligns the vocabs, patches the cil
labels, and fetches method checkpoints from Drive — all idempotent, so a fresh
pod only needs the repo + rclone.conf.

Run (cloud, Linux):
  PYTHONPATH=src python scripts/run_relearn_loc_reeval.py
"""
from __future__ import annotations
import json, os, re, shutil, statistics, subprocess, sys
from pathlib import Path
from typing import Optional

os.environ["RELEARN_NINE"] = "1"   # semua run di sini backbone nine; patch_cil_labels membacanya

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CKPTS = ROOT / "checkpoints"
CFG = ROOT / "configs" / "ablation" / "relearn"
DRIVE_ROOT = "gdrive-mesach:tugas-akhir"
# Architecture selector — graph (N48, 3-seed, includes joint) or seq (S1, 1-seed, no joint here;
# seq joint loc comes from its own RELEARN_METHODS=joint run since the orchestrator now harvests
# loc). Run ids from the per-seed RELEARN md files (graph on ABLATION_RESULTS.md, seq s42 from
# RELEARN_*_nine_seq_s42.md). ARCH set via env RELEARN_ARCH like the orchestrators.
ARCH = os.environ.get("RELEARN_ARCH", "graph").lower()
_ARCH = {
    "graph": {
        "suffix": "_lmgat_codebert_multiclass", "prefix": "N48", "cil_eval": "cil_taskA_eval_nine",
        "seeds": [42, 1, 2],
        "before": {42: "20260707_202747", 1: "20260707_204341", 2: "20260707_205826"},
        "domain": [
            ("Fine-tuning naif",             {42: "20260708_074620", 1: "20260708_083036", 2: "20260708_090639"}),
            ("EWC-DR",                       {42: "20260708_075054", 1: "20260708_083539", 2: "20260708_091300"}),
            ("Experience replay",            {42: "20260708_080018", 1: "20260708_084024", 2: "20260708_092634"}),
            ("EWC-DR dan experience replay", {42: "20260708_081600", 1: "20260708_085327", 2: "20260708_094056"}),
            ("Pelatihan ulang gabungan",     {42: "20260719_120429", 1: "20260719_122249", 2: "20260719_124654"}),
        ],
        "cil": [
            ("Fine-tuning naif",             {42: "20260709_164752", 1: "20260709_165238", 2: "20260709_174921"}),
            ("EWC-DR",                       {42: "20260709_171642", 1: "20260709_170738", 2: "20260709_181114"}),
            ("Experience replay",            {42: "20260709_173058", 1: "20260709_172751", 2: "20260709_184528"}),
            ("EWC-DR dan experience replay", {42: "20260709_175941", 1: "20260709_175123", 2: "20260709_190811"}),
            ("Pelatihan ulang gabungan",     {42: "20260719_130119", 1: "20260719_131921", 2: "20260719_135509"}),
        ],
    },
    "seq": {
        "suffix": "_lmgat_seqgnn_multiclass", "prefix": "S1", "cil_eval": "S1_cil_taskA_eval_nine",
        "seeds": [42],
        "before": {42: "20260707_211550"},
        "domain": [
            ("Fine-tuning naif",             {42: "20260720_005532"}),
            ("EWC-DR",                       {42: "20260720_011157"}),
            ("Experience replay",            {42: "20260720_012921"}),
            ("EWC-DR dan experience replay", {42: "20260720_014553"}),
        ],
        "cil": [
            ("Fine-tuning naif",             {42: "20260720_020650"}),
            ("EWC-DR",                       {42: "20260720_022047"}),
            ("Experience replay",            {42: "20260720_024917"}),
            ("EWC-DR dan experience replay", {42: "20260720_031248"}),
        ],
    },
}[ARCH]

ARCH_TAG = "" if ARCH == "graph" else f"_{ARCH}"
OUT_MD = ROOT / f"RELEARN_LOC_RESULTS_nine{ARCH_TAG}.md"
SEEDS = _ARCH["seeds"]
ARCH_SUFFIX = _ARCH["suffix"]
BEFORE = _ARCH["before"]
DOMAIN_METHODS = _ARCH["domain"]
CIL_METHODS = _ARCH["cil"]
_PFX = _ARCH["prefix"]

# Eval configs. CIL "before" pakai head 26 (config importance); ckpt metode CIL berkepala 36
# sehingga pakai config eval 36. Task-B CIL untuk "before" dilewati — model 26 kelas tidak bisa
# dimuat pada config 36 kelas (bukan load expandable), dan barisnya memang N.A. di tabel utama.
DOMAIN_A_CFG = CFG / f"{_PFX}_taskA_importance_nine.yaml"
DOMAIN_B_CFG = CFG / f"{_PFX}_relearn_naive_nine.yaml"
CIL_A26_CFG = CFG / f"{_PFX}_taskA_importance_nine.yaml"
CIL_A36_CFG = CFG / "cil" / f"{_ARCH['cil_eval']}.yaml"
CIL_B_CFG = CFG / "cil" / f"{_PFX}_cil_naive_nine.yaml"

LOC_KEYS = ("top_1_accuracy", "top_5_accuracy", "ifa_mean")


def sh(args: list) -> None:
    print("\n+ " + " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True, cwd=str(ROOT))


# ── Dataset setup (idempotent, mirrors the orchestrators) ───────────────────
DATASETS = [  # (Drive subdir, archive hint — newest/patched tar wins)
    ("data/processed/megavul", "lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42_lazy_20260613_195029.tar.gz"),
    ("data/processed/relearn", "lm_dataset_relearn_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42.tar.gz"),
    ("data/processed/relearn", "lm_dataset_megavul_cil_multiclass_unixcoder-base-nine_ft_ml1024_lazy.tar.gz"),
]


def _ds_base_name(archive: str) -> str:
    return re.sub(r'(_(lazy|inmemory)(_\d{8}_\d{6})?)?(_\d{8}(_\d{6})?)?\.tar\.gz$', '', archive)


def _newest_archive(remote_dir: str, ds_name: str) -> Optional[str]:
    out = subprocess.run(["rclone", "lsf", f"{DRIVE_ROOT}/{remote_dir}"],
                         capture_output=True, text=True).stdout.splitlines()
    pat = re.compile(rf"^{re.escape(ds_name)}(\.tar\.gz|_(lazy|inmemory)(_\d{{8}}_\d{{6}})?\.tar\.gz|_\d{{8}}(_\d{{6}})?\.tar\.gz)$")
    cands = sorted(f for f in out if pat.match(f))
    return cands[-1] if cands else None


def setup() -> None:
    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    # canonical vocabs BEFORE any dataset load
    (ROOT / "data" / "raw" / "relearn").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CFG / "taskA_cwe_vocab.json", ROOT / "data" / "raw" / "relearn" / "cwe_vocab.json")
    (ROOT / "data" / "raw" / "megavul_cil").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CFG / "cil" / "megavul_cil_cwe_vocab.json",
                 ROOT / "data" / "raw" / "megavul_cil" / "cwe_vocab.json")
    for remote_dir, hint in DATASETS:
        ds = _ds_base_name(hint)
        archive = _newest_archive(remote_dir, ds) or hint
        marker = proc / f".{archive}.extracted"
        if marker.exists():
            print(f"{ds}: newest ({archive}) already extracted, skip")
            continue
        sh(["rclone", "copy", f"{DRIVE_ROOT}/{remote_dir}/{archive}", str(proc), "--progress"])
        sh(["bash", "-c",
            f'tar -I "$(command -v pigz || echo gzip)" -xf "{proc / archive}" -C "{proc}"'])
        (proc / archive).unlink(missing_ok=True)
        marker.touch()
        print(f"{ds}: extracted newest {archive}")
    sh([sys.executable, "scripts/patch_cil_labels.py"])   # idempotent, labels 26..35


def fetch_ckpt(run_id: str) -> Path:
    d = CKPTS / run_id
    if not list(d.glob("best_*.pt")):
        z = f"{run_id}_checkpoints.zip"
        sh(["rclone", "copy", f"{DRIVE_ROOT}/checkpoints/{z}", str(ROOT), "--progress"])
        sh(["unzip", "-o", "-q", str(ROOT / z), "-d", str(ROOT)])
        (ROOT / z).unlink(missing_ok=True)
    return next(d.glob("best_*.pt"))


def eval_loc(run_id: str, cfg: Path, seed: int, tag: str) -> dict:
    """Evaluate run_id's checkpoint on cfg; return the localization block (cached)."""
    ms = RESULTS / tag / "metrics_summary.json"
    if not ms.exists():
        ckpt = fetch_ckpt(run_id)
        ed = CKPTS / tag
        ed.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt, ed / "best_model.pt")
        sh([sys.executable, "-m", "gnn_vuln.evaluate", "--checkpoint", ed / "best_model.pt",
            "--config", cfg, "--split-seed", str(seed), "--seed", str(seed)])
    return json.loads(ms.read_text())["localization"]


def mstd(vals: list[float]) -> str:
    if not vals:
        return "—"
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.3f} ± {s:.3f}"


def collect(scen: str, methods, a_cfg_before: Path, a_cfg_after: Path,
            b_cfg: Path, before_taskB: bool):
    """rows = [(label, {seed: {"A": loc, "B": loc|None}})]"""
    rows = []
    per = {}
    for seed in SEEDS:
        rid = f"{BEFORE[seed]}{ARCH_SUFFIX}"
        per[seed] = {
            "A": eval_loc(rid, a_cfg_before, seed, f"rloc_{scen}_s{seed}_before_A"),
            "B": eval_loc(rid, b_cfg, seed, f"rloc_{scen}_s{seed}_before_B") if before_taskB else None,
        }
    rows.append(("Sebelum pembaruan", per))
    for label, ids in methods:
        per = {}
        for seed in SEEDS:
            rid = f"{ids[seed]}{ARCH_SUFFIX}"
            key = label.lower().replace(" ", "_")[:24]
            per[seed] = {
                "A": eval_loc(rid, a_cfg_after, seed, f"rloc_{scen}_s{seed}_{key}_A"),
                "B": eval_loc(rid, b_cfg, seed, f"rloc_{scen}_s{seed}_{key}_B"),
            }
        rows.append((label, per))
    return rows


def md_tables(title: str, rows) -> list[str]:
    out = [f"## {title}", "",
           "| Model | Top-1 task-A | Top-5 task-A | IFA task-A | Top-1 task-B | Top-5 task-B | IFA task-B |",
           "|---|---|---|---|---|---|---|"]
    for label, per in rows:
        cells = []
        for task in ("A", "B"):
            for k in LOC_KEYS:
                vals = [per[s][task][k] for s in SEEDS
                        if per[s][task] is not None and per[s][task][k] is not None]
                cells.append(mstd(vals))
        out.append("| " + label + " | " + " | ".join(cells) + " |")
    out += ["", "Per-seed:", "",
            "| Model | Seed | Top-1 A | Top-5 A | IFA A | Top-1 B | Top-5 B | IFA B | n loc A | n loc B |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for label, per in rows:
        for s in SEEDS:
            cells = []
            for task in ("A", "B"):
                loc = per[s][task]
                cells += ([f"{loc[k]:.3f}" if loc[k] is not None else "—" for k in LOC_KEYS]
                          if loc is not None else ["—"] * 3)
            na = per[s]["A"]["num_funcs_with_flaw_gt"] if per[s]["A"] else "—"
            nb = per[s]["B"]["num_funcs_with_flaw_gt"] if per[s]["B"] else "—"
            out.append(f"| {label} | {s} | " + " | ".join(cells) + f" | {na} | {nb} |")
    out.append("")
    return out


def main() -> None:
    for p in (DOMAIN_A_CFG, DOMAIN_B_CFG, CIL_A36_CFG, CIL_B_CFG):
        if not p.exists():
            sys.exit(f"Missing config: {p}")
    setup()
    md = [
        "# Lokalisasi pada Continual Learning (graph N48 nine, per-seed, re-eval tanpa retrain)",
        "",
        "Top-1/Top-5/IFA dari blok localization metrics_summary.json, checkpoint metode dari",
        "Drive checkpoints/ (run id sama dengan tabel klasifikasi di ABLATION_RESULTS.md).",
        "Task-B CIL baris Sebelum pembaruan = N.A. (model 26 kelas, config eval 36 kelas).",
        "",
    ]
    md += md_tables("Domain-incremental (task-B = BigVul + TitanVul)",
                    collect("dom", DOMAIN_METHODS, DOMAIN_A_CFG, DOMAIN_A_CFG, DOMAIN_B_CFG, True))
    md += md_tables("Class-incremental (task-B = 10 CWE baru megavul_cil)",
                    collect("cil", CIL_METHODS, CIL_A26_CFG, CIL_A36_CFG, CIL_B_CFG, False))
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print("\n" + OUT_MD.read_text())
    subprocess.run(["rclone", "copy", str(OUT_MD), f"{DRIVE_ROOT}/results/", "--progress"], check=False)
    print(f"\nUploaded {OUT_MD.name} -> {DRIVE_ROOT}/results/")


if __name__ == "__main__":
    main()
