"""Generate bab-4 DISCUSSION diagrams (interpretive figures, not raw results):
  A. Function token-length distribution of vuln test funcs, with the 512-token line
     LineVul cannot exceed  -> docs/Panjang_Fungsi.png   (IV.4.2, easier-subset argument)
  B. Graph vs Hybrid macro F1 flip across 25/26-class x base/nine backbone
     -> docs/Flip_Benign.png  (IV.4.1, with/without-benign flip, robust across backbones)

Run: uv run python scripts/plot_diskusi_diagrams.py
"""
from __future__ import annotations

import os
import glob
import random
import gc
from pathlib import Path

import numpy as np
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
    ax.hist(a[a <= 512], bins=bins, color="#4a9b5e", label=f"≤512 token (diproses LineVul, {100*(1-over):.0f}%)")
    ax.hist(a[a > 512], bins=bins, color="#c0392b", label=f">512 token (dibuang LineVul, {100*over:.0f}%)")
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
    # macro F1 (mean over 3 seeds): [graph, hybrid] per (setting, backbone)
    groups = ["25-kelas\nbase", "25-kelas\nnine", "26-kelas\nbase", "26-kelas\nnine"]
    graph = [0.573, 0.568, 0.474, 0.490]
    hybrid = [0.542, 0.546, 0.521, 0.535]
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, graph, w, color="#3b6fb0", label="Berbasis Graph")
    ax.bar(x + w / 2, hybrid, w, color="#e08a1e", label="Hibrida Graph–LM")
    ax.axvline(1.5, color="gray", linestyle=":", linewidth=1)
    ax.text(0.5, 0.60, "tanpa benign\ngraph > hibrida", ha="center", fontsize=9)
    ax.text(2.5, 0.60, "dengan benign\nhibrida > graph", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Macro F1"); ax.set_ylim(0.40, 0.64)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "Flip_Benign.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote Flip_Benign.png")


def main() -> None:
    plot_length_hist()
    plot_flip()


if __name__ == "__main__":
    main()
