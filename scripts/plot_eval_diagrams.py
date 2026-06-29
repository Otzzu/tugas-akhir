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
# Canonical seed-42 runs under the full ÷present macro fix (multi-seed seed 42), matching
# the mean±std tables in bab-4. Bundles on Drive results/ as <run>_results.zip.
# best model used for the confusion matrix + localization example
RUN = Path("results/ablation/gnn_only/20260625_061202_lmgat_codebert_multiclass")
# the 3 proposed architectures on the same 26-class test, for the per-class F1 comparison
ARCHS = [
    ("Berbasis Graph", "#3b6fb0", "results/ablation/gnn_only/20260625_061202_lmgat_codebert_multiclass"),
    ("Hibrida Graph-LM", "#e08a1e", "results/ablation/phase13/20260625_072202_lmgat_codebert_multiclass"),
    ("Sekuensial", "#4a9b5e", "results/ablation/seqgnn/20260625_063256_lmgat_seqgnn_multiclass"),
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


def plot_per_class_f1_compare(names: list[str], present: list[int], support: np.ndarray) -> None:
    order = [i for i in np.argsort(-support) if i in present]   # present classes, by support desc
    x = np.arange(len(order))
    w = 0.27
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for j, (label, color, path) in enumerate(ARCHS):
        df = pd.read_csv(Path(path) / "predictions.csv")
        f1 = f1_score(df["y_true"], df["y_pred"], labels=range(len(names)), average=None, zero_division=0)
        ax.bar(x + (j - 1) * w, [f1[i] for i in order], width=w, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{names[i]} (n={support[i]})" for i in order], rotation=90, fontsize=7)
    ax.set_ylabel("F1-Score"); ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "F1_per_Kelas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote F1_per_Kelas.png (3 architectures)")


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


def main() -> None:
    df = pd.read_csv(RUN / "predictions.csv")
    names = class_names(list(df.columns))
    support = df["y_true"].value_counts().reindex(range(len(names)), fill_value=0).to_numpy()
    present = [i for i in range(len(names)) if support[i] > 0]
    absent = [names[i] for i in range(len(names)) if support[i] == 0]
    print(f"loaded {len(df)} predictions, {len(names)} classes, {len(present)} present, absent={absent}")
    # confusion matrix per architecture (all on the same 26-class test → same present classes).
    # graph_based is also written as the default Confusion_Matrix.png used by IV.5.
    shorts = {"Berbasis Graph": "graph", "Hibrida Graph-LM": "hybrid", "Sekuensial": "seq"}
    for label, _color, path in ARCHS:
        adf = pd.read_csv(Path(path) / "predictions.csv")
        asup = adf["y_true"].value_counts().reindex(range(len(names)), fill_value=0).to_numpy()
        apres = [i for i in range(len(names)) if asup[i] > 0]
        short = shorts[label]
        out = "Confusion_Matrix.png" if short == "graph" else f"Confusion_Matrix_{short}.png"
        plot_confusion(adf, names, apres, out_name=out, title=f"{label} (26 kelas)")
    plot_per_class_f1_compare(names, present, support)
    plot_localization_example(names)


if __name__ == "__main__":
    main()
