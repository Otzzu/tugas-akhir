#!/usr/bin/env python
"""export_26cls_perseed.py — bangun bundel baseline 26 kelas untuk seed lain, tanpa dataset .pt.

KENAPA. LineVD dan LineVul dibandingkan pada Tabel IV.12 terhadap model 26 kelas. Ketiga
run-nya memakai bundel `megavul_ml1024_baselines_20260707` yang split-nya seed 42, sehingga
seed 1 dan seed 2 hanya mengubah keacakan pelatihan (P#6). Untuk memperbaikinya perlu bundel
dengan split seed 1 dan seed 2, dan flaw mask yang sudah ditambal.

Bundel `split_s1` dan `split_s2` yang ada di Drive TIDAK bisa dipakai, karena dibuat sebelum
flaw mask ditambal. 85,5 persen himpunan baris penyebabnya dimulai dari baris 1, ciri bug
METHOD-span.

CARA KERJA, dan ini yang membuatnya murah. `splits.json` di dalam bundel seed 42 menyimpan
`parquet_ids` untuk train, val, dan test, urutannya sama persis dengan urutan baris pada
parquet. Urutan itu adalah hasil `get_splits(n, 42)`, yaitu shuffle indeks 0..n-1 lalu potong
80/10/10. Jadi baris ke-j dari gabungan train+val+test berpadanan dengan indeks dataset ke
`idx42[j]`. Dari situ pembagian seed berapa pun bisa disusun ulang cukup dengan menukar urutan
baris, tanpa perlu dataset .pt dan tanpa GPU. Isi tiap barisnya, termasuk `flaw_lines` yang
sudah ditambal, ikut apa adanya.

PAKAI.
    uv run python scripts/export_26cls_perseed.py \
        --src data/baselines/ml1024_0707/megavul_ml1024 \
        --out data/baselines/megavul_26cls_perseed --seeds 42 1 2

Keluarannya `megavul_ml1024_baselines_s{seed}.tar.gz`, nama yang dicari lib_baseline_data.sh.
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def get_splits(n, seed, train_ratio=0.8, val_ratio=0.1):
    """Sama persis dengan dataset_lm.get_splits dan export_baseline_split.py."""
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    return idx[:t], idx[t : t + v], idx[t + v :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="direktori megavul_ml1024 dari bundel seed 42 yang sudah ditambal")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--base-seed", type=int, default=42, help="seed asal bundel sumber")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    meta = json.loads((src / "splits.json").read_text())
    pids = meta["parquet_ids"]

    frames = {}
    for kind in ("linevd", "linevul"):
        parts = []
        for s in ("train", "val", "test"):
            p = src / kind / (f"{s}.parquet" if kind == "linevd" else f"{s}.csv")
            parts.append(pd.read_parquet(p) if kind == "linevd" else pd.read_csv(p))
        frames[kind] = parts

    # sanity: urutan baris harus sama dengan splits.json, kalau tidak pemetaan indeksnya salah
    for i, s in enumerate(("train", "val", "test")):
        if list(frames["linevd"][i]["id"]) != list(pids[s]):
            sys.exit(f"GAGAL. Urutan baris {s}.parquet tidak sama dengan splits.json. "
                     "Bundel sumber bukan hasil export_baseline_split.py yang diharapkan.")

    rows = {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}
    n = len(rows["linevd"])
    if n != sum(len(v) for v in pids.values()):
        sys.exit("GAGAL. Jumlah baris tidak cocok dengan splits.json.")

    # penjaga flaw mask, tolak bundel sumber versi lama
    fl = rows["linevd"]["flaw_lines"].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
    w = fl[fl.apply(len) > 0]
    pct = w.apply(lambda x: min(x) == 1).mean() * 100
    print(f"sumber: {n} baris, {len(w)} berflaw, {pct:.1f}% memuat baris 1")
    if pct > 40:
        sys.exit("GAGAL. Bundel sumber memakai flaw mask versi LAMA. Pakai baselines_20260707.")

    tr, va, te = get_splits(n, a.base_seed)
    if (len(tr), len(va), len(te)) != tuple(len(pids[s]) for s in ("train", "val", "test")):
        sys.exit(f"GAGAL. get_splits({n}, {a.base_seed}) memberi {len(tr)}/{len(va)}/{len(te)}, "
                 f"sedangkan bundel sumber {[len(pids[s]) for s in ('train','val','test')]}. "
                 "Rumus split tidak cocok, jangan lanjut.")
    order = tr + va + te                       # order[j] = indeks dataset untuk baris ke-j
    row_of_index = {di: j for j, di in enumerate(order)}
    if len(row_of_index) != n:
        sys.exit("GAGAL. Pemetaan indeks ke baris tidak bijektif.")

    out.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        d = out / f"s{seed}" / "megavul_ml1024"
        for kind in ("linevd", "linevul"):
            (d / kind).mkdir(parents=True, exist_ok=True)
        parts = get_splits(n, seed)
        new_ids = {}
        for name, part in zip(("train", "val", "test"), parts):
            take = [row_of_index[i] for i in part]
            for kind in ("linevd", "linevul"):
                sub = rows[kind].iloc[take].reset_index(drop=True)
                if kind == "linevd":
                    sub.to_parquet(d / kind / f"{name}.parquet", index=False)
                    new_ids[name] = list(sub["id"])
                else:
                    sub.to_csv(d / kind / f"{name}.csv", index=False)
        (d / "splits.json").write_text(json.dumps(
            {"seed": seed, "ds_name": meta["ds_name"], "class_names": meta["class_names"],
             "parquet_ids": new_ids, "derived_from": f"seed {a.base_seed} bundle, patched flaw mask"}))

        te_df = pd.read_parquet(d / "linevd" / "test.parquet")
        nv = int((te_df["vul"] == 1).sum())
        print(f"  seed {seed}: train {len(parts[0])} val {len(parts[1])} test {len(parts[2])}, rentan di test {nv}")

        tar = out / f"megavul_ml1024_baselines_s{seed}.tar.gz"
        comp = shutil.which("pigz") or "gzip"
        with open(tar, "wb") as f:
            p1 = subprocess.Popen(["tar", "-cf", "-", "-C", str(out / f"s{seed}"), "megavul_ml1024"],
                                  stdout=subprocess.PIPE)
            subprocess.run([comp], stdin=p1.stdout, stdout=f, check=True)
            p1.stdout.close(); p1.wait()
        print(f"    {tar.name}  {tar.stat().st_size / 1e6:.1f} MB")

    print("\nUnggah:")
    print(f"  rclone copy {out} gdrive-mesach:tugas-akhir/data/baselines/ "
          "--include 'megavul_ml1024_baselines_s*.tar.gz' --progress")


if __name__ == "__main__":
    main()
