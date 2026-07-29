"""Error analysis — kapan model salah, dan di mana bedanya dengan baseline.

Membaca artefak yang sudah ada di results/docs_figs/ (tidak menjalankan model ulang):
  seeds/{graph,hybrid,seq}_{42,1,2}/predictions.csv   26 kelas, per seed
  vulnonly/{graph,hybrid,seq}/predictions.csv         25 kelas rentan, seed 42
  baselines/{vulexp,losver}_cls_preds.csv             25 kelas rentan, seed 42

Keluaran: ERROR_ANALYSIS.md + 3 gambar di root.

FP  = fungsi tidak rentan diprediksi rentan (hanya ada pada ruang 26 kelas).
FN  = fungsi rentan diprediksi tidak rentan.
Pada ruang 25 kelas rentan, FP/FN dihitung one-vs-rest per kelas, karena model
vuln-only tidak pernah melihat fungsi tidak rentan.

Usage:
    uv run python scripts/error_analysis.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd  # sebelum torch (bentrok DLL di Windows)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(".")
FIG = ROOT / "results/docs_figs"
PARQUET = Path("data/datasets/megavul/train.parquet")
GRAPHS = Path("data/processed/lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42_graphs")
ARCHS = {"Berbasis graph": "graph", "Hibrida": "hybrid", "Sekuensial": "seq"}
SEEDS = [42, 1, 2]
BUCKETS = ["≤20", "21–50", "51–100", "101–200", ">200"]
CALL_BUCKETS = ["≤2", "3–5", "6–12", ">12"]

# kata kunci yang menyerupai pemanggilan fungsi tetapi bukan
KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "catch", "do", "else"}
CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
# API yang lazim menjadi sink kerentanan memori dan injeksi
DANGEROUS = re.compile(
    r"\b(memcpy|memmove|strcpy|strncpy|strcat|strncat|sprintf|vsprintf|gets|scanf|sscanf|"
    r"alloca|malloc|calloc|realloc|free|system|popen|exec[lv]p?e?|fork)\s*\(")


def n_calls(code: str) -> int:
    """Jumlah pemanggilan fungsi eksternal yang berbeda (proksi ketergantungan lintas fungsi)."""
    return len({m for m in CALL_RE.findall(code) if m not in KEYWORDS})


def has_dangerous(code: str) -> bool:
    return bool(DANGEROUS.search(code))


# ── metadata fungsi ──────────────────────────────────────────────────────────
def load_meta(ids: set[int]) -> pd.DataFrame:
    df = pd.read_parquet(PARQUET, columns=["func_before", "language", "flaw_lines"])
    df = df.loc[sorted(ids)].copy()
    df["parquet_id"] = df.index
    df["n_lines"] = df.func_before.str.count("\n") + 1
    df["n_flaw"] = df.flaw_lines.apply(lambda x: 0 if x is None else len(x))
    df["n_calls"] = df.func_before.apply(n_calls)
    df["api_bahaya"] = df.func_before.apply(has_dangerous)
    return df.drop(columns=["func_before", "flaw_lines"])


def node_counts(ids: set[int]) -> pd.DataFrame:
    rows = []
    for f in GRAPHS.glob("*.pt"):
        g = torch.load(f, weights_only=False)
        pid = getattr(g, "parquet_id", None)
        if pid is not None and int(pid) in ids:
            rows.append({"parquet_id": int(pid), "n_nodes": int(g.num_nodes)})
    return pd.DataFrame(rows)


def bucket(s: pd.Series, edges: list[int]) -> pd.Series:
    labels = [f"≤{edges[0]}"] + [f"{edges[i]+1}–{edges[i+1]}" for i in range(len(edges) - 1)] + [f">{edges[-1]}"]
    return pd.cut(s, [-1] + edges + [10**9], labels=labels)


def tag(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["is_vuln"] = d.y_true != 0
    d["is_fn"] = d.is_vuln & (d.y_pred == 0)
    d["is_fp"] = (~d.is_vuln) & (d.y_pred != 0)
    d["correct"] = d.y_true == d.y_pred
    d["salah_cwe"] = d.is_vuln & (d.y_pred != 0) & ~d.correct
    return d


def md_table(df: pd.DataFrame) -> list[str]:
    out = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values]
    return out


# ── ruang 26 kelas, tiga seed ────────────────────────────────────────────────
def analyse_26(meta: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    md, ringkas, iris = ["## Ruang 26 kelas (mean ± std atas seed 42, 1, 2)\n"], [], []
    for name, key in ARCHS.items():
        per_seed = []
        for s in SEEDS:
            p = FIG / f"seeds/{key}_{s}/predictions.csv"
            if not p.exists():
                continue
            d = tag(pd.read_csv(p)).merge(meta, on="parquet_id", how="left")
            v, b = d[d.is_vuln], d[~d.is_vuln]
            per_seed.append({
                "akurasi": d.correct.mean(), "FN": v.is_fn.mean(), "FP": b.is_fp.mean(),
                "salah CWE": v.salah_cwe.mean(),
                "conf benar": d[d.correct].confidence.mean(), "conf salah": d[~d.correct].confidence.mean(),
            })
            d["Panjang fungsi"] = bucket(d.n_lines, [20, 50, 100, 200])
            for k, g in d.groupby("Panjang fungsi", observed=True):
                gv, gb = g[g.is_vuln], g[~g.is_vuln]
                iris.append({"arsitektur": name, "seed": s, "Panjang fungsi": k, "n": len(g),
                             "FN": gv.is_fn.mean() if len(gv) else None,
                             "salah CWE": gv.salah_cwe.mean() if len(gv) else None,
                             "FP": gb.is_fp.mean() if len(gb) >= 5 else None})
        t = pd.DataFrame(per_seed)
        ringkas.append({"Arsitektur": name, **{c: f"{t[c].mean():.3f} ± {t[c].std():.3f}" for c in t.columns}})
    md += md_table(pd.DataFrame(ringkas))
    md.append("\nFN dihitung atas fungsi rentan, FP atas fungsi tidak rentan, salah CWE atas fungsi rentan yang terdeteksi.\n")

    iris = pd.DataFrame(iris)
    piv = (iris.groupby(["arsitektur", "Panjang fungsi"], observed=True)[["FN", "salah CWE", "FP"]]
           .mean().round(3).reset_index())
    piv["Panjang fungsi"] = pd.Categorical(piv["Panjang fungsi"], categories=BUCKETS, ordered=True)
    piv = piv.sort_values(["arsitektur", "Panjang fungsi"]).fillna("—")
    md.append("\n### Menurut panjang fungsi\n")
    md += md_table(piv)
    md.append("\nFP kosong pada fungsi panjang karena hampir tidak ada fungsi tidak rentan di sana.\n")
    return md, iris


# ── ruang 25 kelas rentan, melawan baseline ──────────────────────────────────
def per_class(d: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    rows = []
    for c in classes:
        t, p = d.cwe_true == c, d.cwe_pred == c
        if t.sum() == 0:
            continue
        tp, fn, fp = int((t & p).sum()), int((t & ~p).sum()), int((~t & p).sum())
        rows.append({"CWE": c, "n": int(t.sum()), "recall": round(tp / (tp + fn), 3), "FP": fp})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# kelompok kelas menurut letak alur data (analisis memori vs taint)
MEMORI = ["CWE-787", "CWE-125", "CWE-476", "CWE-416", "CWE-120"]
TAINT = ["CWE-20", "CWE-200", "CWE-22", "CWE-89", "CWE-79", "CWE-862"]


def _p25(rel_new: str, rel_legacy: str, seed: int) -> Path:
    """Path artefak 25 kelas satu seed. Seed 42 boleh jatuh ke layout lama."""
    p = FIG / rel_new
    if p.exists() or seed != 42:
        return p
    return FIG / rel_legacy


def load_25_models(seed: int) -> tuple[dict, list[str]]:
    """Prediksi 25 kelas rentan tiap model untuk satu seed."""
    models, classes = {}, []
    for name, key in ARCHS.items():
        p = pd.read_csv(_p25(f"vulnonly/{key}_{seed}/predictions.csv",
                             f"vulnonly/{key}/predictions.csv", seed))
        classes = [c[len("prob_"):] for c in p.columns if c.startswith("prob_")]
        p["cwe_true"] = p.y_true.map(lambda i: classes[int(i)])
        p["cwe_pred"] = p.y_pred.map(lambda i: classes[int(i)])
        models[name] = p

    # VulExplainer: id = indeks pada daftar CWE yang diurutkan alfabetis
    v = pd.read_csv(_p25(f"baselines/vulexp_cls_preds_{seed}.csv",
                         "baselines/vulexp_cls_preds.csv", seed))
    order = sorted(classes)
    v["cwe_true"] = v.y_true.map(lambda i: order[int(i)])
    v["cwe_pred"] = v.y_pred.map(lambda i: order[int(i)])
    models["VulExplainer"] = v

    # LOSVER: nama CWE tiap baris ada di test.jsonl miliknya, urut sama dengan prediksinya
    lo = pd.read_csv(_p25(f"baselines/losver_cls_preds_{seed}.csv",
                          "baselines/losver_cls_preds.csv", seed))
    jsonl = _p25(f"baselines/losver_{seed}/test.jsonl", "baselines/losver/test.jsonl", seed)
    rows = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
    lo["cwe_true"] = [r["CWE ID"] for r in rows]
    m = lo.groupby("y_true").cwe_true.agg(lambda s: s.mode()[0]).to_dict()
    lo["cwe_pred"] = lo.y_pred.map(lambda i: m.get(int(i), "?"))
    models["LOSVER"] = lo
    return models, classes


def _group_recall(d: pd.DataFrame, classes: list[str]) -> float:
    """Recall micro-average (ditimbang jumlah sampel) atas sekelompok kelas."""
    tp = fn = 0
    for c in classes:
        t = d.cwe_true == c
        tp += int((t & (d.cwe_pred == c)).sum())
        fn += int((t & (d.cwe_pred != c)).sum())
    return tp / (tp + fn) if (tp + fn) else float("nan")


def analyse_memori_taint() -> list[str]:
    """Recall kelompok memori vs taint tiap model, mean ± std atas seed yang tersedia."""
    order = ["Berbasis graph", "Hibrida", "Sekuensial", "VulExplainer", "LOSVER"]
    per = {name: {"memori": [], "taint": []} for name in order}
    used = []
    for s in SEEDS:
        vp = _p25(f"vulnonly/graph_{s}/predictions.csv", "vulnonly/graph/predictions.csv", s)
        if not vp.exists():
            continue
        used.append(s)
        models, _ = load_25_models(s)
        for name in order:
            per[name]["memori"].append(_group_recall(models[name], MEMORI))
            per[name]["taint"].append(_group_recall(models[name], TAINT))
    rows = []
    for name in order:
        me, ta = pd.Series(per[name]["memori"]), pd.Series(per[name]["taint"])
        rows.append({"Model": name,
                     "Recall memori": f"{me.mean():.3f} ± {me.std():.3f}",
                     "Recall taint": f"{ta.mean():.3f} ± {ta.std():.3f}",
                     "Selisih": f"{ta.mean() - me.mean():+.3f}"})
    md = [f"\n## Memori versus taint (mean ± std atas seed {', '.join(map(str, used))})\n",
          "Recall ditimbang jumlah sampel tiap kelas.\n"]
    md += md_table(pd.DataFrame(rows))
    return md


def analyse_25() -> tuple[list[str], pd.DataFrame]:
    models, classes = load_25_models(42)

    base = per_class(models["Berbasis graph"], classes)[["CWE", "n"]]
    for name, d in models.items():
        pc = per_class(d, classes).set_index("CWE")
        base[f"recall {name}"] = base.CWE.map(pc.recall)
        base[f"FP {name}"] = base.CWE.map(pc.FP)

    md = ["\n## Ruang 25 kelas rentan, melawan baseline (seed 42)\n",
          "FP di sini one-vs-rest, yaitu fungsi kelas lain yang diprediksi sebagai kelas ini.\n"]
    md += md_table(base.head(12).fillna("—"))
    for name, d in models.items():
        conf = d[d.cwe_true != d.cwe_pred].groupby(["cwe_true", "cwe_pred"]).size().sort_values(ascending=False).head(5)
        md.append(f"\n### {name} — akurasi {(d.cwe_true == d.cwe_pred).mean():.3f} atas {len(d)} fungsi\n")
        md.append("| kelas benar | diprediksi | n |\n|---|---|---|")
        md += [f"| {t} | {p} | {n} |" for (t, p), n in conf.items()]
    return md, base


# ── gambar ───────────────────────────────────────────────────────────────────
def plots(iris: pd.DataFrame, per_cls: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, kind in enumerate(["FN", "FP"]):
        for name in ARCHS:
            d = (iris[iris.arsitektur == name].groupby("Panjang fungsi", observed=True)[kind]
                 .mean().reindex(BUCKETS))
            ax[i].plot(BUCKETS, d.values, marker="o", label=name)
        ax[i].set_title(f"{kind} menurut panjang fungsi")
        ax[i].set_xlabel("baris kode"); ax[i].set_ylabel(kind); ax[i].grid(alpha=.3)
    ax[0].legend()
    fig.tight_layout(); fig.savefig("Error_FP_FN_Panjang.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for name in ARCHS:
        d = (iris[iris.arsitektur == name].groupby("Panjang fungsi", observed=True)["salah CWE"]
             .mean().reindex(BUCKETS))
        ax.plot(BUCKETS, d.values, marker="o", label=name)
    ax.set_title("Kesalahan jenis CWE menurut panjang fungsi"); ax.set_xlabel("baris kode")
    ax.set_ylabel("proporsi fungsi rentan"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig("Error_Salah_CWE.png", dpi=150); plt.close(fig)

    top = per_cls.head(8)
    cols = [c for c in per_cls.columns if c.startswith("recall ")]
    x = range(len(top)); w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(11, 4))
    for j, c in enumerate(cols):
        ax.bar([i + j * w for i in x], top[c].fillna(0), width=w, label=c.replace("recall ", ""))
    ax.set_xticks([i + 0.4 for i in x]); ax.set_xticklabels(top.CWE, rotation=30)
    ax.set_ylabel("recall"); ax.set_title("Recall tiap kelas, arsitektur usulan melawan baseline")
    ax.grid(alpha=.3, axis="y"); ax.legend(ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig("Error_Recall_Per_Kelas.png", dpi=150); plt.close(fig)


# ── uji tambahan ─────────────────────────────────────────────────────────────
def analyse_calls(meta: pd.DataFrame) -> list[str]:
    """Ketergantungan lintas fungsi, diproksikan jumlah pemanggilan fungsi eksternal.

    Dibandingkan dengan LOSVER pada HIMPUNAN FUNGSI YANG SAMA, yaitu 478 fungsi yang
    dinilai LOSVER, agar selisihnya tidak tercampur perbedaan subset.
    """
    md = ["\n## Menurut jumlah pemanggilan fungsi eksternal\n",
          "Proksi ketergantungan lintas fungsi. Makin banyak fungsi yang dipanggil, makin "
          "banyak bagian cerita yang berada di luar fungsi yang dinilai.\n"]

    rows = []
    for name, key in ARCHS.items():
        for s in SEEDS:
            p = FIG / f"seeds/{key}_{s}/predictions.csv"
            if not p.exists():
                continue
            d = tag(pd.read_csv(p)).merge(meta, on="parquet_id", how="left")
            d["b"] = pd.cut(d.n_calls, [-1, 2, 5, 12, 10**6], labels=CALL_BUCKETS)
            for k, g in d.groupby("b", observed=True):
                gv = g[g.is_vuln]
                rows.append({"arsitektur": name, "Panggilan": k, "n": len(g),
                             "akurasi": g.correct.mean(), "salah CWE": gv.salah_cwe.mean()})
    t = (pd.DataFrame(rows).groupby(["arsitektur", "Panggilan"], observed=True)[["akurasi", "salah CWE"]]
         .mean().round(3).reset_index())
    t["Panggilan"] = pd.Categorical(t["Panggilan"], categories=CALL_BUCKETS, ordered=True)
    md += md_table(t.sort_values(["arsitektur", "Panggilan"]))

    # LOSVER vs arsitektur usulan pada fungsi yang sama
    lo = pd.read_csv(FIG / "baselines/losver_cls_preds.csv")
    rows_j = [json.loads(l) for l in open(FIG / "baselines/losver/test.jsonl", encoding="utf-8")]
    lo["code"] = [r["func_before"] for r in rows_j]
    par = pd.read_parquet(PARQUET, columns=["func_before"])
    pid_of = {c: i for i, c in zip(par.index, par.func_before)}
    lo["parquet_id"] = lo.code.map(pid_of)
    lo = lo.dropna(subset=["parquet_id"])
    lo["parquet_id"] = lo.parquet_id.astype(int)
    lo["benar"] = lo.y_true == lo.y_pred
    lo["n_calls"] = lo.code.apply(n_calls)
    lo["b"] = pd.cut(lo.n_calls, [-1, 2, 5, 12, 10**6], labels=CALL_BUCKETS)

    # Fungsi yang sama persis. Prediksi 26 kelas kami dibatasi ke 25 kelas CWE dengan
    # mengambil argmax hanya atas kolom prob CWE, sehingga setara dengan LOSVER yang
    # tidak pernah memprediksi kelas tidak rentan.
    ref = pd.read_csv(FIG / "seeds/graph_42/predictions.csv")
    ids = set(lo.parquet_id) & set(ref.parquet_id)
    lo = lo[lo.parquet_id.isin(ids)]
    comp = lo.groupby("b", observed=True).agg(n=("benar", "size"), LOSVER=("benar", "mean"))
    for name, key in ARCHS.items():
        d = pd.read_csv(FIG / f"seeds/{key}_42/predictions.csv")
        d = d[d.parquet_id.isin(ids)].merge(meta[["parquet_id", "n_calls"]], on="parquet_id")
        cwe_cols = [c for c in d.columns if c.startswith("prob_CWE-")]
        names = [c[len("prob_"):] for c in d.columns if c.startswith("prob_")]
        pred_cwe = d[cwe_cols].idxmax(axis=1).str[len("prob_"):]
        d["benar"] = pred_cwe.values == d.y_true.map(lambda i: names[int(i)]).values
        d["b"] = pd.cut(d.n_calls, [-1, 2, 5, 12, 10**6], labels=CALL_BUCKETS)
        comp[name] = d.groupby("b", observed=True).benar.mean()
    comp = comp.reindex(CALL_BUCKETS).round(3).reset_index().rename(columns={"b": "Panggilan"})
    md.append(f"\n### Akurasi jenis CWE pada {len(ids)} fungsi yang sama, arsitektur usulan melawan LOSVER\n")
    md.append("Prediksi arsitektur usulan dibatasi ke 25 kelas CWE, sehingga tidak dirugikan oleh "
              "kemampuannya menebak kelas tidak rentan yang tidak dimiliki LOSVER.\n")
    md += md_table(comp.fillna("—"))
    return md


def analyse_threshold(meta: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Fungsi hanya ditandai rentan bila confidence melewati ambang."""
    rows = []
    for name, key in ARCHS.items():
        for s in SEEDS:
            p = FIG / f"seeds/{key}_{s}/predictions.csv"
            if not p.exists():
                continue
            d = tag(pd.read_csv(p))
            v, b = d[d.is_vuln], d[~d.is_vuln]
            for t in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                flag_v = (v.y_pred != 0) & (v.confidence >= t)
                flag_b = (b.y_pred != 0) & (b.confidence >= t)
                benar = flag_v & (v.y_true == v.y_pred)
                rows.append({"arsitektur": name, "ambang": t,
                             "FP": flag_b.mean(), "FN": 1 - flag_v.mean(),
                             "CWE benar dari yang ditandai": benar.sum() / max(flag_v.sum(), 1)})
    t = (pd.DataFrame(rows).groupby(["arsitektur", "ambang"])[["FP", "FN", "CWE benar dari yang ditandai"]]
         .mean().round(3).reset_index())
    md = ["\n## Ambang confidence\n",
          "Fungsi hanya ditandai rentan bila kelas prediksinya bukan tidak rentan DAN confidence-nya "
          "melewati ambang. FN naik sebagai gantinya.\n"]
    md += md_table(t)
    return md, t


def analyse_api(meta: pd.DataFrame) -> list[str]:
    rows = []
    for name, key in ARCHS.items():
        for s in SEEDS:
            p = FIG / f"seeds/{key}_{s}/predictions.csv"
            if not p.exists():
                continue
            d = tag(pd.read_csv(p)).merge(meta, on="parquet_id", how="left")
            for api, g in d.groupby("api_bahaya"):
                gv, gb = g[g.is_vuln], g[~g.is_vuln]
                rows.append({"arsitektur": name, "API berbahaya": "ada" if api else "tidak ada",
                             "n": len(g), "akurasi": g.correct.mean(),
                             "FN": gv.is_fn.mean(), "FP": gb.is_fp.mean() if len(gb) else None,
                             "salah CWE": gv.salah_cwe.mean()})
    t = (pd.DataFrame(rows).groupby(["arsitektur", "API berbahaya"])[["n", "akurasi", "FN", "FP", "salah CWE"]]
         .mean().round(3).reset_index())
    t["n"] = t.n.round(0).astype(int)   # mean n antarseed, dibulatkan
    md = ["\n## Kehadiran API berbahaya\n",
          "API yang lazim menjadi sink, misalnya memcpy, strcpy, sprintf, system, dan exec.\n"]
    md += md_table(t)
    return md


def main() -> None:
    # gabungan id test dari SELURUH seed, karena tiap seed memakai split sendiri
    ids: set[int] = set()
    for key in ARCHS.values():
        for s in SEEDS:
            p = FIG / f"seeds/{key}_{s}/predictions.csv"
            if p.exists():
                ids |= set(pd.read_csv(p).parquet_id)
    meta = load_meta(ids).merge(node_counts(ids), on="parquet_id", how="left")

    md = ["# Analisis Error — kapan model salah\n",
          "Dihitung dari artefak prediksi yang sudah ada di `results/docs_figs/`, tanpa menjalankan model ulang.\n"]
    m26, iris = analyse_26(meta)
    m25, per_cls = analyse_25()
    mmt = analyse_memori_taint()
    mcall = analyse_calls(meta)
    mthr, thr = analyse_threshold(meta)
    mapi = analyse_api(meta)
    md += m26 + m25 + mmt + mcall + mthr + mapi
    md.append("\n## Gambar\n\n![FP dan FN](Error_FP_FN_Panjang.png)\n\n![Salah CWE](Error_Salah_CWE.png)\n\n"
              "![Recall per kelas](Error_Recall_Per_Kelas.png)\n\n![Ambang confidence](Error_Ambang.png)\n")

    # bagian "Temuan baru" ditulis tangan, jangan tertimpa saat tabel diregenerasi
    out = Path("ERROR_ANALYSIS.md")
    MARK = "\n---\n\n# Temuan baru"
    tail = ""
    if out.exists() and MARK in out.read_text(encoding="utf-8"):
        tail = MARK + out.read_text(encoding="utf-8").split(MARK, 1)[1]
    out.write_text("\n".join(md) + tail, encoding="utf-8")
    plots(iris, per_cls)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ARCHS:
        d = thr[thr.arsitektur == name]
        ax.plot(d.FN, d.FP, marker="o", label=name)
        for _, r in d.iterrows():
            ax.annotate(f"{r.ambang:g}", (r.FN, r.FP), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("FN (fungsi rentan terlewat)"); ax.set_ylabel("FP (fungsi aman ditandai)")
    ax.set_title("Trade-off ambang confidence"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig("Error_Ambang.png", dpi=150); plt.close(fig)
    print("tulis ERROR_ANALYSIS.md + 3 gambar di root")


if __name__ == "__main__":
    main()
