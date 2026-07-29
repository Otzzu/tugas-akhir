#!/usr/bin/env python
"""collect_baseline_results.py — panen metrik baseline dari Drive jadi tabel siap tempel.

Tiap tar hasil di `results/baselines/` memuat `<model>_recomputed_metrics.json`. Skrip ini
mengambil yang terbaru per pasangan model dan seed pada satu tanggal, lalu mencetak baris per
seed beserta mean dan standard deviation dalam format tabel laporan, koma sebagai desimal.

Dipakai bersama `REVISI_TABEL_BASELINE.md`, yang memuat salinan persis tabel yang harus ditimpa.

    uv run python scripts/collect_baseline_results.py --day 20260729
    uv run python scripts/collect_baseline_results.py --day 20260729 --keep   # simpan unduhan

Hanya membaca Drive, tidak mengubah apa pun di sana.
"""
import argparse
import json
import sys
import re
import shutil
import statistics as st
import subprocess
import tarfile
import tempfile
from pathlib import Path

MODELS = ["losver", "vulexplainer", "livable", "linevul", "linevd"]
NAMA = {"losver": "LOSVER", "vulexplainer": "VulExplainer", "livable": "LIVABLE",
        "linevul": "LineVul", "linevd": "LineVD"}
REMOTE = "gdrive-mesach:tugas-akhir/results/baselines"

CLS = [("accuracy", "Akurasi", 3), ("f1_macro", "Macro F1", 3), ("f1_weighted", "Weighted F1", 3)]
LOC = [("ifa_mean", "IFA", 2), ("top_1_accuracy", "Top-1", 3), ("top_5_accuracy", "Top-5", 3),
       ("recall_at_5pct_loc", "R@5%LOC", 3), ("recall_at_20pct_loc", "R@20%LOC", 3),
       ("effort_at_20pct_recall", "Effort@20%R", 3)]


def num(x, d):
    return f"{x:.{d}f}".replace(".", ",")


def cell(vals, d):
    if len(vals) == 1:
        return num(vals[0], d)
    return f"{num(st.mean(vals), d)} ± {num(st.stdev(vals), d)}"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # tanda ± rusak di konsol Windows
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="20260729", help="cap tanggal pada nama berkas hasil")
    ap.add_argument("--fallback-day", default="20260707",
                    help="dipakai hanya bila --day tidak punya hasil untuk pasangan itu. "
                         "LineVul dan LineVD seed 42 tidak ikut rerun karena bundelnya memang "
                         "sudah benar, jadi angkanya diambil dari gelombang ini")
    ap.add_argument("--seeds", nargs="+", default=["42", "1", "2"])
    ap.add_argument("--cache", default=None, help="direktori unduhan, default sementara")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    listing = subprocess.run(["rclone", "lsf", f"{REMOTE}/"], capture_output=True, text=True).stdout.split()
    cache = Path(a.cache) if a.cache else Path(tempfile.mkdtemp(prefix="baseline_"))
    cache.mkdir(parents=True, exist_ok=True)

    data, eff, hilang, lama = {}, {}, [], []
    for m in MODELS:
        for s in a.seeds:
            hits, dipakai = [], a.day
            # Fallback HANYA untuk LineVul dan LineVD seed 42. Keduanya sengaja tidak ikut rerun
            # karena bundel seed 42-nya memang sudah benar. Model lain tidak boleh jatuh ke
            # gelombang lama, karena itu akan mencampur dua metodologi tanpa terlihat.
            hari_list = (a.day, a.fallback_day) if (m in ("linevul", "linevd") and s == "42") else (a.day,)
            for hari in hari_list:
                pat = re.compile(rf"^{m}_.*_s{s}_{hari}_.*_results\.tar\.gz$")
                hits = sorted(f for f in listing if pat.match(f))
                if hits:
                    dipakai = hari
                    break
            if not hits:
                hilang.append(f"{m}-s{s}")
                continue
            if dipakai != a.day:
                lama.append(f"{m}-s{s}")
            tar = hits[-1]                      # terbaru, nama memuat timestamp
            p = cache / tar
            if not p.exists():
                subprocess.run(["rclone", "copy", f"{REMOTE}/{tar}", str(cache)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with tarfile.open(p) as t:
                for member in t.getnames():
                    base = Path(member).name
                    if base == f"{m}_recomputed_metrics.json":
                        data[(m, s)] = json.load(t.extractfile(member))
                    elif base == "train_efficiency.json":
                        eff[(m, s)] = json.load(t.extractfile(member))

    if hilang:
        print(f"BELUM ADA: {' '.join(hilang)}")
    if lama:
        print(f"DARI GELOMBANG {a.fallback_day}, sengaja tidak di-rerun: {' '.join(lama)}")
    print()

    for tugas, kolom in (("classification", CLS), ("localization", LOC)):
        rows = [m for m in MODELS if any((m, s) in data and tugas in data[(m, s)] for s in a.seeds)]
        if not rows:
            continue
        judul = "Klasifikasi" if tugas == "classification" else "Lokalisasi"
        print(f"\n### {judul}\n")
        print("| Model | " + " | ".join(l for _, l, _ in kolom) + " | Banyak data test |")
        print("| --- |" + " --- |" * (len(kolom) + 1))
        for m in rows:
            got = [(s, data[(m, s)][tugas]) for s in a.seeds if (m, s) in data and tugas in data[(m, s)]]
            sel = []
            for k, _, d in kolom:
                v = [g[k] for _, g in got if k in g]
                sel.append(cell(v, d) if v else "—")
            nk = "n" if "n" in got[0][1] else "num_funcs_with_flaw_gt"
            n = [g[nk] for _, g in got if nk in g]
            sel.append(cell([float(x) for x in n], 0) if n else "—")
            print(f"| {NAMA[m]} | " + " | ".join(sel) + " |")
        print("\nPer seed:\n")
        for m in rows:
            for s in a.seeds:
                if (m, s) not in data or tugas not in data[(m, s)]:
                    continue
                g = data[(m, s)][tugas]
                isi = " ".join(f"{l} {num(g[k], d)}" for k, l, d in kolom if k in g)
                nk = "n" if "n" in g else "num_funcs_with_flaw_gt"
                print(f"  {NAMA[m]:13s} seed {s:2s}  {isi}  n {g.get(nk, '?')}")

    if eff:
        print("\n### Efisiensi, untuk Tabel H.2\n")
        print("| Baseline | Mean waktu total (jam) | Puncak VRAM | GPU |")
        print("| --- | --- | --- | --- |")
        for m in MODELS:
            got = [eff[(m, s)] for s in a.seeds if (m, s) in eff]
            if not got:
                continue
            jam = [g["total_time_s"] / 3600 for g in got]
            vram = max(g["peak_vram_mib"] for g in got) / 1024
            gpu = sorted({g["gpu"].replace("NVIDIA GeForce ", "") for g in got})
            print(f"| {NAMA[m]} | {num(st.mean(jam), 2)} | {num(vram, 1)} GB | {', '.join(gpu)} |")
        print("\nGPU lebih dari satu berarti seed-nya tidak seragam, sebutkan apa adanya.")

    if a.keep or a.cache:
        print(f"\nunduhan di {cache}")
    else:
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
