"""Plot class distributions for the thesis datasets with one shared style, so Gambar IV.1
(dataset utama) and the two continual-learning figures look identical.

Counts from Tabel IV.5, IV.6, IV.7 (docs/bab-4/prosedur_dan_data.md).

Outputs (docs/):
  Distribusi_CWE.png          dataset utama          (MegaVul, 26 kelas)
  Distribusi_CWE_relearn.png  domain-incremental     (BigVul + TitanVul, 26-class space)
  Distribusi_CWE_cil.png      class-incremental      (MegaVul, 10 new CWE)

Run:  uv run python scripts/plot_dataset_distribution.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOCS = Path(__file__).resolve().parents[1] / "docs"

# Tabel IV.5 — dataset utama, 26 kelas
UTAMA = [
    ("Tidak rentan", 1600), ("CWE-787", 1553), ("CWE-125", 1330), ("CWE-476", 1265),
    ("CWE-20", 1187), ("CWE-416", 975), ("CWE-200", 603), ("CWE-120", 424),
    ("CWE-79", 336), ("CWE-22", 323), ("CWE-89", 209), ("CWE-770", 138),
    ("CWE-502", 112), ("CWE-122", 111), ("CWE-284", 107), ("CWE-863", 98),
    ("CWE-78", 94), ("CWE-862", 77), ("CWE-94", 56), ("CWE-918", 47),
    ("CWE-434", 44), ("CWE-352", 44), ("CWE-77", 38), ("CWE-121", 22),
    ("CWE-306", 21), ("CWE-639", 5),
]

# Tabel IV.6 — domain-incremental, total 4.546
RELEARN = [
    ("Tidak rentan", 893), ("CWE-125", 893), ("CWE-20", 893), ("CWE-787", 470),
    ("CWE-200", 351), ("CWE-416", 318), ("CWE-476", 212), ("CWE-120", 193),
    ("CWE-284", 126), ("CWE-22", 53), ("CWE-79", 46), ("CWE-122", 23),
    ("CWE-78", 20), ("CWE-94", 13), ("CWE-770", 13), ("CWE-863", 9),
    ("CWE-77", 5), ("CWE-862", 4), ("CWE-502", 3), ("CWE-352", 3),
    ("CWE-918", 1), ("CWE-639", 1), ("CWE-89", 1), ("CWE-121", 1), ("CWE-434", 1),
]

# Tabel IV.7 — class-incremental, total 5.166 (all vulnerable, no benign)
CIL = [
    ("CWE-119", 1618), ("CWE-190", 708), ("CWE-362", 543), ("CWE-399", 470),
    ("CWE-264", 446), ("CWE-400", 339), ("CWE-401", 283), ("CWE-189", 261),
    ("CWE-617", 256), ("CWE-835", 242),
]


def plot(data, title, out, figsize):
    labels = [d[0] for d in data]
    counts = [d[1] for d in data]
    colors = ["#9e9e9e" if l == "Tidak rentan" else "#1f77b4" for l in labels]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(labels)), counts, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("Jumlah fungsi")
    ax.set_xlabel("Kelas")
    ax.set_title(title)
    ax.margins(x=0.01)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + max(counts) * 0.01, str(c),
                ha="center", va="bottom", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.name}  (total {sum(counts)} fungsi, {len(labels)} kelas)")


plot(UTAMA, "Distribusi kelas dataset utama",
     DOCS / "Distribusi_CWE.png", (12, 5))
plot(RELEARN, "Distribusi kelas dataset domain-incremental",
     DOCS / "Distribusi_CWE_relearn.png", (12, 5))
plot(CIL, "Distribusi kelas dataset class-incremental",
     DOCS / "Distribusi_CWE_cil.png", (8, 5))
