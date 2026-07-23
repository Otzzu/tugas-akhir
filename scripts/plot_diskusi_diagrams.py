"""Generate bab-4 DISCUSSION diagrams (interpretive figures, not raw results):
  A. Function token-length distribution of vuln test funcs, with the 512-token line
     LineVul cannot exceed  -> docs/Panjang_Fungsi.png   (IV.4.2, easier-subset argument)
  B. Graph vs Hybrid macro F1 flip on 25/26-class (backbone unixcoder-base-nine), mean +/- std
     -> docs/Flip_Benign.png  (IV.4.1, with/without-benign flip)

Run: uv run python scripts/plot_diskusi_diagrams.py
"""
from __future__ import annotations

import os
import glob
import json
import random
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

OUT = Path("docs")
D = "data/processed/lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42_graphs"


def func_lengths() -> np.ndarray:
    """Token length (capped at ml1024) of vuln+flaw test functions, same seed-42 split."""
    n = len(glob.glob(os.path.join(D, "*.pt")))
    idx = list(range(n)); random.seed(42); random.shuffle(idx)
    t, v = int(n * 0.8), int(n * 0.1)
    test_idx = idx[t + v:]
    lens = []
    for k, gi in enumerate(test_idx):
        d = torch.load(os.path.join(D, f"{gi}.pt"), weights_only=False, map_location="cpu")
        if int(d.y) != 0:
            fm = getattr(d, "flaw_line_mask", None)
            if torch.is_tensor(fm) and bool(fm.any()):
                lens.append(int(d.func_attention_mask.sum()))
        del d
        if k % 200 == 0:
            gc.collect()
    return np.array(lens)


def plot_length_hist() -> None:
    a = func_lengths()
    over = (a > 512).mean()
    trunc = (a >= 1024).mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(0, 1075, 50)
    ax.hist(a[a <= 512], bins=bins, color="#4a9b5e", label=f"≤512 token (terbaca baseline LM, {100*(1-over):.0f}%)")
    ax.hist(a[a > 512], bins=bins, color="#c0392b", label=f">512 token (di luar jangkauan, {100*over:.0f}%)")
    ax.axvline(512, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 1075)
    ax.text(512, ax.get_ylim()[1] * 0.92, " batas 512", fontsize=9)
    ax.annotate(f"≥1024 token\n(terpotong, {100*trunc:.0f}%)", xy=(1024, 0), xytext=(880, ax.get_ylim()[1] * 0.55),
                fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color="#c0392b"))
    ax.set_xlabel("Panjang fungsi (token UniXcoder, dibatasi 1024)")
    ax.set_ylabel("Jumlah fungsi")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "Panjang_Fungsi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote Panjang_Fungsi.png  (n={len(a)}, >512={100*over:.0f}%)")


def plot_flip() -> None:
    # macro F1 (mean +/- std over 3 seeds, backbone unixcoder-base-nine): [graph, hybrid]
    groups     = ["25 kelas rentan", "26 kelas"]
    graph      = [0.548, 0.466]
    graph_std  = [0.036, 0.037]
    hybrid     = [0.529, 0.513]
    hybrid_std = [0.043, 0.027]
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - w / 2, graph, w, yerr=graph_std, capsize=5,
           color="#3b6fb0", label="Berbasis graph")
    ax.bar(x + w / 2, hybrid, w, yerr=hybrid_std, capsize=5,
           color="#e08a1e", label="Hibrida graph–LM")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.text(0.0, 0.655, "tanpa benign\ngraph > hibrida", ha="center", va="top", fontsize=9)
    ax.text(1.0, 0.655, "dengan benign\nhibrida > graph", ha="center", va="top", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Macro F1"); ax.set_ylim(0.40, 0.67)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "Flip_Benign.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote Flip_Benign.png")


SEEDS = (42, 1, 2)


def _pid_token_len(seed: int) -> dict:
    """parquet_id -> UniXcoder token length (capped 1024) over THIS seed's test split.
    The split follows train.seed (train.py / evaluate.py), so each seed sees a different
    test set. Cached to results/docs_figs/tokmap_seed{seed}.json."""
    cache = f"results/docs_figs/tokmap_seed{seed}.json"
    if os.path.exists(cache):
        return {int(k): v for k, v in json.load(open(cache)).items()}
    n = len(glob.glob(os.path.join(D, "*.pt")))
    idx = list(range(n)); random.seed(seed); random.shuffle(idx)
    t, v = int(n * 0.8), int(n * 0.1)
    m = {}
    for gi in idx[t + v:]:
        p = os.path.join(D, f"{gi}.pt")
        if not os.path.exists(p):
            continue
        d = torch.load(p, weights_only=False, map_location="cpu")
        if int(d.y) != 0 and hasattr(d, "parquet_id") and hasattr(d, "func_attention_mask"):
            m[int(d.parquet_id)] = int(d.func_attention_mask.sum())
        del d
    json.dump(m, open(cache, "w"))
    return m


ARCH_LABEL = {"graph": "Berbasis graph", "hybrid": "Hibrida", "seq": "Sekuensial"}


def plot_ifa_scatter() -> None:
    """IFA vs function length for the THREE architectures (one panel each), three seeds
    pooled per arch. Each dot is a function scored by the model of the seed whose test
    split contains it. Black line is median IFA per length bin. Shows IFA rising with
    length on all three, so the long functions (>512 token) that baselines drop are where
    localization degrades (IV.4.2, Tabel IV.22)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (key, label) in zip(axes, ARCH_LABEL.items()):
        ifa, tok = [], []
        for seed in SEEDS:
            pid2len = _pid_token_len(seed)
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
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Panjang fungsi (token UniXcoder)")
        ax.set_xlim(0, 1030); ax.set_ylim(-1, 41)
    axes[0].set_ylabel("IFA (baris salah sebelum penyebab)")
    axes[0].text(520, 38, " batas 512", fontsize=8)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[2].text(0.99, 0.02, "titik >40 dipotong", transform=axes[2].transAxes,
                 ha="right", fontsize=7, color="gray")
    fig.tight_layout()
    fig.savefig(OUT / "Lokalisasi_IFA_Panjang.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote Lokalisasi_IFA_Panjang.png (3 arch)")


def plot_cl_tradeoff() -> None:
    """Old-task F1 vs new-task F1 per continual learning method, mean +/- std over 3 seeds
    (Tabel IV.15 / IV.16). Shows the frontier: no method wins on both axes. Dashed line is
    the old-task F1 before the update, so points above it have negative forgetting.

    Joint retraining is plotted as a star: it is the upper-bound reference, not a continual
    method. It makes the domain-incremental result readable at a glance — EWC-DR + replay
    sits above it on BOTH axes there, while on class-incremental it stays out of reach."""
    before_old = 0.472
    # label: (new_f1, new_std, old_f1, old_std)
    domain = {
        "Fine-tuning naif":  (0.410, 0.054, 0.156, 0.006),
        "EWC-DR":            (0.390, 0.050, 0.408, 0.051),
        "Experience replay": (0.460, 0.019, 0.376, 0.024),
        "EWC-DR + replay":   (0.427, 0.019, 0.606, 0.135),
    }
    cil = {
        "Fine-tuning naif":  (0.573, 0.011, 0.000, 0.001),
        "EWC-DR":            (0.518, 0.021, 0.081, 0.028),
        "Experience replay": (0.474, 0.024, 0.370, 0.011),
        "EWC-DR + replay":   (0.331, 0.045, 0.476, 0.020),
    }
    # Joint retraining is the upper-bound reference, NOT a continual method: kept out of the
    # dicts above so it never enters the CIL frontier line, and drawn with its own marker.
    joint = {
        "Domain-incremental": (0.397, 0.052, 0.469, 0.043),
        "Class-incremental":  (0.497, 0.028, 0.461, 0.078),
    }
    colors = {"Fine-tuning naif": "#c0392b", "EWC-DR": "#e08a1e",
              "Experience replay": "#3b6fb0", "EWC-DR + replay": "#4a9b5e"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, (data, title) in zip(axes, [(domain, "Domain-incremental"),
                                        (cil, "Class-incremental")]):
        for lab, (nx, nsd, oy, osd) in data.items():
            ax.errorbar(nx, oy, xerr=nsd, yerr=osd, fmt="o", ms=9, capsize=3,
                        color=colors[lab], label=lab, zorder=3)
        jx, jxsd, jy, jysd = joint[title]
        ax.errorbar(jx, jy, xerr=jxsd, yerr=jysd, fmt="*", ms=16, capsize=3,
                    color="#6c3483", label="Pelatihan ulang gabungan (batas atas)", zorder=4)
        ax.axhline(before_old, color="gray", ls="--", lw=1)
        ax.text(0.98, before_old + 0.015, "F1 task lama sebelum pembaruan", ha="right",
                transform=ax.get_yaxis_transform(), fontsize=7.5, color="gray")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Macro F1 task baru (menyerap yang baru)")
        ax.set_xlim(0.30, 0.62); ax.set_ylim(-0.05, 0.85)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("Macro F1 task lama (menjaga yang lama)")
    # CIL points are monotone; the line only guides the eye along their order
    fr = sorted((v[0], v[2]) for v in cil.values())
    axes[1].plot([p[0] for p in fr], [p[1] for p in fr], ls=":", lw=1.1, color="gray", zorder=1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    fig.savefig(OUT / "Tradeoff_Continual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote Tradeoff_Continual.png")


def main() -> None:
    plot_length_hist()
    plot_flip()
    plot_ifa_scatter()
    plot_cl_tradeoff()


if __name__ == "__main__":
    main()
