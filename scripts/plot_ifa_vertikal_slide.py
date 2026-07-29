"""Versi susunan vertikal (3 baris 1 kolom) dari Lokalisasi_IFA_Panjang, khusus slide.

Isi dan datanya sama persis dengan Gambar IV.9, hanya panelnya ditumpuk ke bawah.
Ditulis ke docs/laporan-individu/image/slides/, berkas laporan tidak disentuh.
"""
import os
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Otzzu\Documents\tugas-akhir")
OUTDIR = ROOT / "docs" / "laporan-individu" / "image" / "slides"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_diskusi_diagrams as m

fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.8), sharex=True, sharey=True)
for ax, (key, label) in zip(axes, m.ARCH_LABEL.items()):
    ifa, tok = [], []
    for seed in m.SEEDS:
        pid2len = m._pid_token_len(seed)
        df = pd.read_csv(f"results/docs_figs/seeds/{key}_{seed}/localization_scores.csv")
        for _, g in df.groupby("func_idx"):
            if (g["y_true"] > 0).sum() == 0 or g["is_flaw_line"].sum() == 0:
                continue
            gg = g.sort_values("score", ascending=False).reset_index(drop=True)
            tl = pid2len.get(int(g["parquet_id"].iloc[0]))
            if tl is None:
                continue
            ifa.append(int(gg.index[gg["is_flaw_line"] == 1][0]))
            tok.append(tl)
    ifa, tok = np.array(ifa), np.array(tok)
    s = tok <= 512
    ax.scatter(tok[s], np.clip(ifa[s], 0, 40), s=10, alpha=0.35, color="#3b6fb0",
               edgecolors="none", label="≤512 token")
    ax.scatter(tok[~s], np.clip(ifa[~s], 0, 40), s=10, alpha=0.35, color="#c0392b",
               edgecolors="none", label=">512 token")
    be = np.arange(0, 1025, 128); bc = (be[:-1] + be[1:]) / 2
    med = [np.median(ifa[(tok >= be[i]) & (tok < be[i + 1])])
           if ((tok >= be[i]) & (tok < be[i + 1])).any() else np.nan
           for i in range(len(be) - 1)]
    ax.plot(bc, med, color="black", marker="o", lw=2, label="median IFA per bin")
    ax.axvline(512, color="gray", ls="--", lw=1)
    ax.set_title(label, fontsize=12)
    ax.set_ylabel("IFA")
    ax.set_xlim(0, 1030); ax.set_ylim(-1, 41)

axes[-1].set_xlabel("Panjang fungsi (token UniXcoder)")
axes[0].text(520, 37, " batas 512", fontsize=9)
axes[0].legend(loc="upper left", fontsize=9)
axes[0].text(0.99, 0.94, "titik >40 dipotong", transform=axes[0].transAxes,
             ha="right", va="top", fontsize=8, color="gray")
fig.supylabel("IFA (baris salah sebelum penyebab)", fontsize=11)
for ax in axes:
    ax.set_ylabel("")
fig.tight_layout()
out = OUTDIR / "Lokalisasi_IFA_Panjang_vertikal.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
from PIL import Image
w, h = Image.open(out).size
print(f"ditulis {out.name}  {w}x{h}  ar={w/h:.2f}")
