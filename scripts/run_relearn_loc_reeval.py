"""
run_relearn_loc_reeval.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Harvest LOCALIZATION metrics for the continual-learning runs (graph N48, nine,
per-seed) by re-evaluating the saved method checkpoints — no retraining.
The two orchestrators only harvested f1_macro; gnn_vuln.evaluate already
computes the localization block at every eval, so this script re-runs the same
evals and collects Top-1/Top-5/IFA from metrics_summary.json into
RELEARN_LOC_RESULTS_nine.md (+ upload to Drive results/).

Prereqs on the pod (already in place after any prior --setup run of the two
orchestrators on this pod): megavul + relearn + cil .pt datasets extracted,
relearn vocab aligned, cil labels patched. Method checkpoints download
automatically from Drive checkpoints/.

Run (cloud, Linux):
  PYTHONPATH=src python scripts/run_relearn_loc_reeval.py
"""
from __future__ import annotations
import json, statistics, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CKPTS = ROOT / "checkpoints"
CFG = ROOT / "configs" / "ablation" / "relearn"
DRIVE_ROOT = "gdrive-mesach:tugas-akhir"
OUT_MD = ROOT / "RELEARN_LOC_RESULTS_nine.md"
SEEDS = [42, 1, 2]
ARCH_SUFFIX = "_lmgat_codebert_multiclass"

# Task-A backbone (klasifikasi 26 kelas per-seed, patched dataset) = baris "Sebelum pembaruan".
BEFORE = {42: "20260707_202747", 1: "20260707_204341", 2: "20260707_205826"}

# Method run ids per seed — from ABLATION_RESULTS.md per-seed tables (patched runs
# 2026-07-08 domain, 2026-07-09 CIL, 2026-07-19 joint).
DOMAIN_METHODS = [
    ("Fine-tuning naif",             {42: "20260708_074620", 1: "20260708_083036", 2: "20260708_090639"}),
    ("EWC-DR",                       {42: "20260708_075054", 1: "20260708_083539", 2: "20260708_091300"}),
    ("Experience replay",            {42: "20260708_080018", 1: "20260708_084024", 2: "20260708_092634"}),
    ("EWC-DR dan experience replay", {42: "20260708_081600", 1: "20260708_085327", 2: "20260708_094056"}),
    ("Pelatihan ulang gabungan",     {42: "20260719_120429", 1: "20260719_122249", 2: "20260719_124654"}),
]
CIL_METHODS = [
    ("Fine-tuning naif",             {42: "20260709_164752", 1: "20260709_165238", 2: "20260709_174921"}),
    ("EWC-DR",                       {42: "20260709_171642", 1: "20260709_170738", 2: "20260709_181114"}),
    ("Experience replay",            {42: "20260709_173058", 1: "20260709_172751", 2: "20260709_184528"}),
    ("EWC-DR dan experience replay", {42: "20260709_175941", 1: "20260709_175123", 2: "20260709_190811"}),
    ("Pelatihan ulang gabungan",     {42: "20260719_130119", 1: "20260719_131921", 2: "20260719_135509"}),
]

# Eval configs. CIL "before" pakai head 26 (config importance); ckpt metode CIL berkepala 36
# sehingga pakai config eval 36. Task-B CIL untuk "before" dilewati — model 26 kelas tidak bisa
# dimuat pada config 36 kelas (bukan load expandable), dan barisnya memang N.A. di tabel utama.
DOMAIN_A_CFG = CFG / "N48_taskA_importance_nine.yaml"
DOMAIN_B_CFG = CFG / "N48_relearn_naive_nine.yaml"
CIL_A26_CFG = CFG / "N48_taskA_importance_nine.yaml"
CIL_A36_CFG = CFG / "cil" / "cil_taskA_eval_nine.yaml"
CIL_B_CFG = CFG / "cil" / "N48_cil_naive_nine.yaml"

LOC_KEYS = ("top_1_accuracy", "top_5_accuracy", "ifa_mean")


def sh(args: list) -> None:
    print("\n+ " + " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True, cwd=str(ROOT))


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
        import shutil
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
