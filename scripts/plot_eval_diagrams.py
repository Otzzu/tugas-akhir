"""Generate bab-4 evaluation diagrams from research artifacts (predictions.csv + localization_scores.csv):
  1. Confusion matrix of the best model (graph_based, 26-class)      -> docs/Confusion_Matrix.png
  2. Per-class F1 comparison across the 3 proposed architectures     -> docs/F1_per_Kelas.png
  3. Qualitative localization example (reserved for IV.4 discussion) -> docs/Lokalisasi_Contoh.png

All three architectures are evaluated on the same 26-class test split. Run:
  uv run python scripts/plot_eval_diagrams.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

OUT = Path("docs")
# Canonical seed-42 runs, backbone unixcoder-base-nine, 26-class, ÷present macro fix,
# matching the mean±std tables in bab-4. Bundles on Drive results/ as <run>_results.zip.
# best model used for the confusion matrix + localization example
RUN = Path("results/docs_figs/graph")
# the 3 proposed architectures on the same 26-class test, for the per-class F1 comparison
ARCHS = [
    ("Berbasis Graph", "#3b6fb0", "results/docs_figs/graph"),
    ("Hibrida Graph-LM", "#e08a1e", "results/docs_figs/hybrid"),
    ("Sekuensial", "#4a9b5e", "results/docs_figs/seq"),
]


def class_names(pred_cols: list[str]) -> list[str]:
    return [c[len("prob_"):] for c in pred_cols if c.startswith("prob_")]


def plot_confusion(df: pd.DataFrame, names: list[str], present: list[int],
                   out_name: str = "Confusion_Matrix.png", title: str | None = None) -> None:
    pnames = [names[i] for i in present]
    cm = confusion_matrix(df["y_true"], df["y_pred"], labels=present)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(pnames))); ax.set_yticks(range(len(pnames)))
    ax.set_xticklabels(pnames, rotation=90, fontsize=7)
    ax.set_yticklabels(pnames, fontsize=7)
    ax.set_xlabel("Kelas prediksi"); ax.set_ylabel("Kelas sebenarnya")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proporsi (recall per kelas)")
    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_name}")


def plot_confusion_grid(names: list[str], present: list[int]) -> None:
    """3-panel confusion (Berbasis Graph / Hibrida / Sekuensial) on the same 26-class test."""
    pnames = [names[i] for i in present]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5), constrained_layout=True)
    im = None
    for j, (ax, (label, _c, path)) in enumerate(zip(axes, ARCHS)):
        key = Path(path).name
        cmns = []
        for s in ("42", "1", "2"):
            adf = pd.read_csv(f"results/docs_figs/seeds/{key}_{s}/predictions.csv")
            cm = confusion_matrix(adf["y_true"], adf["y_pred"], labels=present)
            cmns.append(cm / cm.sum(axis=1, keepdims=True).clip(min=1))
        cmn = np.mean(cmns, axis=0)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(pnames))); ax.set_xticklabels(pnames, rotation=90, fontsize=6)
        ax.set_yticks(range(len(pnames)))
        ax.set_xlabel("Kelas prediksi"); ax.set_title(label)
        if j == 0:
            ax.set_yticklabels(pnames, fontsize=6); ax.set_ylabel("Kelas sebenarnya")
        else:
            ax.set_yticklabels([])
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, label="Proporsi (recall per kelas)")
    fig.savefig(OUT / "Confusion_Matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote Confusion_Matrix.png (3-panel)")


def plot_per_class_f1_compare(names: list[str], present: list[int], support: np.ndarray) -> None:
    order = [i for i in np.argsort(-support) if i in present]   # present classes, by support desc
    x = np.arange(len(order))
    w = 0.27
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for j, (label, color, path) in enumerate(ARCHS):
        key = Path(path).name
        f1s = []
        for s in ("42", "1", "2"):
            df = pd.read_csv(f"results/docs_figs/seeds/{key}_{s}/predictions.csv")
            f1s.append(f1_score(df["y_true"], df["y_pred"], labels=range(len(names)), average=None, zero_division=0))
        f1m = np.mean(f1s, axis=0); f1sd = np.std(f1s, axis=0)
        ax.bar(x + (j - 1) * w, [f1m[i] for i in order], width=w, color=color, label=label,
               yerr=[f1sd[i] for i in order], capsize=2, error_kw={"elinewidth": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels([f"{names[i]} (n={support[i]})" for i in order], rotation=90, fontsize=7)
    ax.set_ylabel("F1-Score"); ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "F1_per_Kelas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote F1_per_Kelas.png (3 architectures)")


def plot_confusion_benign_mini() -> None:
    """Where do the truly-benign functions land? One 100%-stacked bar per architecture,
    split into predicted-benign (correct) / leaked to a memory class (CWE-416/476/787) /
    other. Mean over the 3 seeds of the benign confusion row (26-class runs), matching the
    mean±std tables. Both segments are monotone: predicted-benign rises graph->seq->hybrid
    and the memory leak falls, so graph leaks most and hybrid separates benign best.
    (Values recomputed from predictions.csv of the 3-seed 26-class runs.)"""
    labels = ["Berbasis\ngraph", "Sekuensial", "Hibrida\ngraph–LM"]
    ben  = [0.348, 0.492, 0.529]   # predicted benign (correct)
    memo = [0.377, 0.267, 0.218]   # leaked to CWE-416/476/787
    other = [1 - b - m for b, m in zip(ben, memo)]
    x = np.arange(len(labels)); w = 0.5
    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x, ben, w, label="Tetap tidak rentan (benar)", color="#4a9b5e")
    b2 = ax.bar(x, memo, w, bottom=ben, label="Salah jadi kelas memori", color="#c0392b")
    b3 = ax.bar(x, other, w, bottom=[a + b for a, b in zip(ben, memo)],
                label="Salah jadi kelas lain", color="#b8bcc2")
    for xi, bn, mm in zip(x, ben, memo):
        ax.text(xi, bn + mm / 2, f"{mm:.2f}", ha="center", va="center",
                color="white", fontweight="bold", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Proporsi fungsi tidak rentan")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "Confusion_Benign_Mini.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote Confusion_Benign_Mini.png")


def plot_localization_example(names: list[str]) -> None:
    loc = pd.read_csv(RUN / "localization_scores.csv")
    cands = []  # clean single-cause example: 1 flaw line ranked #1, not the signature line
    for fidx, g in loc.groupby("func_idx"):
        if g["y_true"].iloc[0] != g["y_pred"].iloc[0]:
            continue
        n = len(g)
        if not (6 <= n <= 16) or g["is_flaw_line"].sum() != 1:
            continue
        flaw = g[g["is_flaw_line"] == 1].iloc[0]
        if flaw["score"] != g["score"].max() or flaw["line_number"] <= 1:
            continue
        margin = flaw["score"] - g[g["is_flaw_line"] == 0]["score"].max()
        cands.append((margin, fidx, g))
    if not cands:
        print("no clean localization example found"); return
    _, fidx, g = max(cands, key=lambda t: t[0])
    g = g.sort_values("line_number")
    cwe = names[int(g["y_true"].iloc[0])]
    colors = ["#c0392b" if f == 1 else "#9bb7d4" for f in g["is_flaw_line"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(g["line_number"], g["score"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Skor kecurigaan"); ax.set_ylabel("Nomor baris")
    ax.set_yticks(g["line_number"])
    handles = [plt.Rectangle((0, 0), 1, 1, color="#c0392b"),
               plt.Rectangle((0, 0), 1, 1, color="#9bb7d4")]
    ax.legend(handles, ["Baris penyebab (label)", "Baris lain"], loc="lower right")
    ax.set_title(f"Skor kecurigaan per baris pada satu fungsi rentan ({cwe})")
    fig.tight_layout()
    fig.savefig(OUT / "Lokalisasi_Contoh.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote Lokalisasi_Contoh.png (func_idx={fidx}, {cwe})")


def plot_classification_compare() -> None:
    """Macro F1 (25-class vuln-only) bar chart, arch usulan vs baseline, mean±std over seeds 42,1,2.
    Bars are the mean, whiskers the std — the overlapping error bars are the point (top-5 within noise).
    Numbers mirror Tabel IV.10."""
    # (name, mean, std, group)
    data = [
        ("LOSVER", 0.635, 0.030, "baseline"),
        ("VulExplainer", 0.593, 0.030, "baseline"),
        ("Sekuensial", 0.553, 0.006, "usulan"),
        ("Berbasis Graph", 0.548, 0.036, "usulan"),
        ("Hibrida", 0.529, 0.043, "usulan"),
        ("LIVABLE", 0.041, 0.008, "baseline"),
    ]
    names = [d[0] for d in data]
    means = np.array([d[1] for d in data])
    stds = np.array([d[2] for d in data])
    colors = ["#3b6fb0" if d[3] == "usulan" else "#9aa0a6" for d in data]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, means, yerr=stds, color=colors, capsize=5,
           error_kw={"elinewidth": 1.3, "ecolor": "#333"})
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylabel("Macro F1 (25 kelas)")
    ax.set_ylim(0, 0.72)
    ax.grid(axis="y", alpha=0.3)
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.012, f"{m:.3f}", ha="center", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#3b6fb0", label="Arsitektur usulan"),
                       Patch(color="#9aa0a6", label="Baseline")], loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "Perbandingan_Macro_F1.png", dpi=150)
    plt.close(fig)
    print("wrote Perbandingan_Macro_F1.png")


def main() -> None:
    plot_classification_compare()
    df = pd.read_csv(RUN / "predictions.csv")
    names = class_names(list(df.columns))
    support = df["y_true"].value_counts().reindex(range(len(names)), fill_value=0).to_numpy()
    present = [i for i in range(len(names)) if support[i] > 0]
    absent = [names[i] for i in range(len(names)) if support[i] == 0]
    print(f"loaded {len(df)} predictions, {len(names)} classes, {len(present)} present, absent={absent}")
    # confusion matrix per architecture (all on the same 26-class test → same present classes).
    # graph_based is also written as the default Confusion_Matrix.png used by IV.5.
    plot_confusion_grid(names, present)
    plot_per_class_f1_compare(names, present, support)
    plot_localization_example(names)


if __name__ == "__main__":
    main()
