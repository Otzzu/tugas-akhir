"""RM2 deep-dive per arsitektur (berbasis graph, hibrida, sekuensial) x 3 seed, mean+/-std.

Menghasilkan tiga tabel Bab IV.4 untuk KETIGA arsitektur, bukan hanya graph:
  - IV.22  IFA/Top-1/Top-5 lokalisasi menurut panjang fungsi (bucket token <=512 / >512)
  - IV.23  peringkat baris pada fungsi TANPA anotasi (heuristik variabel baris-ditambah)
  - IV.25  FN/salah-CWE/FP klasifikasi menurut panjang fungsi (bucket baris)

Semua dihitung dari artefak prediksi yang sudah ada, tanpa menjalankan model ulang:
  results/docs_figs/seeds/<arch>_<seed>/predictions.csv        (klasifikasi)
  results/docs_figs/seeds/<arch>_<seed>/localization_scores.csv (skor per baris)
  data/datasets/megavul/train.parquet                          (func_before, func_after)

Panjang token dihitung ulang dari func_before dengan tokenizer UniXcoder (sama dengan varian
-nine), karena artefak prediksi tidak menyimpan panjang token.

Heuristik IV.23 (fungsi rentan tanpa baris anotasi, yaitu perbaikan hanya MENAMBAH baris).
Menyederhanakan depadd LineVD (get_dep_add_lines): LineVD menandai baris lama yang menjadi
dependensi DATA dan kontrol dari baris yang ditambah, diambil dari edge PDG Joern. Versi ini
meniru bagian data-dependency-nya pada tingkat string, tanpa membangun PDG func_after:
  1. diff func_before vs func_after -> baris yang ditambah perbaikan.
  2. ambil identifier pada baris ditambah yang SUDAH ADA di func_before -> variabel yang dijaga.
  3. target = baris lama tempat variabel itu di-ASSIGN (sumber reaching-def), bukan tiap
     kemunculannya. TARGET_MODE="use" mengembalikan perilaku longgar lama (semua kemunculan).
  4. nilai peringkat model atas target, yaitu Top-1/Top-5 dan median peringkat terbaik.

Run: uv run python scripts/rm2_by_arch.py
"""
from __future__ import annotations

import io
import re
import difflib
from functools import lru_cache

import numpy as np
import pandas as pd

ROOT = "."
SEEDS = [42, 1, 2]
ARCHS = {"graph": "Berbasis graph", "hybrid": "Hibrida", "seq": "Sekuensial"}
TARGET_MODE = "use_before"   # "use_before" = pemakaian variabel SEBELUM titik sisip guard; "def" = baris assignment; "use" = semua kemunculan

LOG = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n")
    print(s.encode("ascii", "replace").decode())


PQ = pd.read_parquet(f"{ROOT}/data/datasets/megavul/train.parquet",
                     columns=["func_before", "func_after"])
out("parquet loaded", len(PQ))

from transformers import AutoTokenizer
TOK = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")

KW = {"if", "for", "while", "switch", "return", "sizeof", "catch", "do", "else", "case",
      "break", "continue", "goto", "default", "struct", "const", "static", "void", "int",
      "char", "unsigned", "long", "short", "float", "double", "NULL", "true", "false",
      "typedef", "enum", "union"}
IDENT = re.compile(r"[A-Za-z_]\w*")


def loc_path(key, s):
    return f"{ROOT}/results/docs_figs/seeds/{key}_{s}/localization_scores.csv"


def pred_path(key, s):
    return f"{ROOT}/results/docs_figs/seeds/{key}_{s}/predictions.csv"


@lru_cache(maxsize=None)
def toklen(pid: int) -> int:
    n = len(TOK(PQ.at[pid, "func_before"], truncation=False, add_special_tokens=True)["input_ids"])
    return min(n, 1024)


def nlines(pid: int) -> int:
    return PQ.at[pid, "func_before"].count("\n") + 1


# ── IV.22 IFA menurut panjang token ─────────────────────────────────────────
def iv22_perfunc(df):
    rows = []
    for fi, g in df.groupby("func_idx"):
        if not g.is_flaw_line.any():
            continue
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        best = int(np.where(g.is_flaw_line.values)[0].min())
        rows.append({"ifa": best, "top1": best == 0, "top5": best < 5,
                     "tok": toklen(int(g.parquet_id.iloc[0]))})
    return pd.DataFrame(rows)


def iv22():
    out("\n########## IV.22 IFA/Top-k menurut panjang token (mean+/-std, 3 seed) ##########")
    for key, name in ARCHS.items():
        seed_stats = {"<=512": [], ">512": []}
        for s in SEEDS:
            pf = iv22_perfunc(pd.read_csv(loc_path(key, s)))
            pf["b"] = np.where(pf.tok <= 512, "<=512", ">512")
            for b in seed_stats:
                sub = pf[pf.b == b]
                seed_stats[b].append(dict(n=len(sub), ifa=sub.ifa.mean(),
                                          ifamed=sub.ifa.median(), top1=sub.top1.mean(),
                                          top5=sub.top5.mean()))
        out(f"\n=== {name} ===")
        for b in ["<=512", ">512"]:
            r = seed_stats[b]
            def ms(k):
                v = np.array([x[k] for x in r], float); return v.mean(), v.std()
            n, ifa, imd, t1, t5 = ms("n"), ms("ifa"), ms("ifamed"), ms("top1"), ms("top5")
            out(f"  {b}: n {n[0]:.0f}+/-{n[1]:.0f}  IFA {ifa[0]:.1f}+/-{ifa[1]:.1f}  "
                f"IFAmed {imd[0]:.1f}  Top1 {t1[0]:.3f}+/-{t1[1]:.3f}  Top5 {t5[0]:.3f}+/-{t5[1]:.3f}")


# ── IV.23 fungsi tanpa anotasi ──────────────────────────────────────────────
@lru_cache(maxsize=None)
def added_vars(pid: int) -> frozenset:
    before, after = PQ.at[pid, "func_before"], PQ.at[pid, "func_after"]
    if not isinstance(after, str):
        return frozenset()
    bset = {l.strip() for l in before.splitlines()}
    added = [l for l in after.splitlines() if l.strip() and l.strip() not in bset]
    bvars = set(IDENT.findall(before)) - KW
    av = {t for l in added for t in IDENT.findall(l) if t not in KW and t in bvars}
    return frozenset(av)


@lru_cache(maxsize=None)
def targets_before(pid: int) -> frozenset:
    """String func_before yang MEMAKAI variabel dijaga DAN berada sebelum titik sisip guard."""
    before, after = PQ.at[pid, "func_before"], PQ.at[pid, "func_after"]
    if not isinstance(after, str):
        return frozenset()
    bstrip = [l.strip() for l in before.splitlines()]
    astrip = [l.strip() for l in after.splitlines()]
    bvars = set(IDENT.findall(before)) - KW
    tgt = set()
    for tag, a1, a2, b1, b2 in difflib.SequenceMatcher(None, bstrip, astrip).get_opcodes():
        if tag in ("insert", "replace"):
            vs = {t for l in astrip[b1:b2] for t in IDENT.findall(l) if t not in KW and t in bvars}
            if not vs:
                continue
            pat = re.compile(r"\b(" + "|".join(re.escape(v) for v in vs) + r")\b")
            for j in range(a1):                 # baris sebelum titik sisip
                if bstrip[j] and pat.search(bstrip[j]):
                    tgt.add(bstrip[j])
    return frozenset(tgt)


def iv23():
    out("\n########## IV.23 fungsi tanpa anotasi (mean+/-std, 3 seed) ##########")
    for key, name in ARCHS.items():
        per = []
        for s in SEEDS:
            d = pd.read_csv(loc_path(key, s))
            d = d[d.y_true != 0]
            nf, t1, t5, ranks = 0, 0, 0, []
            for fi, g in d.groupby("func_idx"):
                if g.is_flaw_line.any():
                    continue
                pid = int(g.parquet_id.iloc[0])
                g = g.sort_values("score", ascending=False).reset_index(drop=True)
                if TARGET_MODE == "use_before":
                    tset = targets_before(pid)
                    if not tset:
                        continue
                    tgt = g.index[g.code.fillna("").str.strip().isin(tset)].tolist()
                else:
                    av = added_vars(pid)
                    if not av:
                        continue
                    alt = "|".join(re.escape(v) for v in av)
                    if TARGET_MODE == "def":
                        pat = re.compile(r"(?<![=!<>])\b(" + alt + r")\b\s*(?:\[[^\]]*\])?\s*=(?!=)")
                    else:
                        pat = re.compile(r"\b(" + alt + r")\b")
                    tgt = g.index[g.code.fillna("").str.contains(pat)].tolist()
                if not tgt:
                    continue
                best = min(tgt)
                nf += 1; t1 += best == 0; t5 += best < 5; ranks.append(best + 1)
            per.append(dict(n=nf, top1=t1 / nf, top5=t5 / nf, med=np.median(ranks)))
        def ms(k):
            v = np.array([x[k] for x in per], float); return v.mean(), v.std()
        n, a1, a5, md = ms("n"), ms("top1"), ms("top5"), ms("med")
        out(f"  {name}: n {n[0]:.0f}+/-{n[1]:.0f}  Top1 {a1[0]:.3f}+/-{a1[1]:.3f}  "
            f"Top5 {a5[0]:.3f}+/-{a5[1]:.3f}  median {md[0]:.1f}+/-{md[1]:.1f}")


# ── IV.25 error klasifikasi menurut panjang baris ───────────────────────────
def iv25():
    out("\n########## IV.25 FN/salah-CWE/FP menurut panjang baris (mean+/-std, 3 seed) ##########")
    order = ["<=20", "21-50", "51-100", "101-200", ">200"]
    def bkt(n):
        for e, l in [(20, "<=20"), (50, "21-50"), (100, "51-100"), (200, "101-200")]:
            if n <= e:
                return l
        return ">200"
    rows = []
    for key, name in ARCHS.items():
        for s in SEEDS:
            d = pd.read_csv(pred_path(key, s))
            d["b"] = d.parquet_id.map(nlines).map(bkt)
            d["vuln"] = d.y_true != 0
            for b, g in d.groupby("b"):
                gv, gb = g[g.vuln], g[~g.vuln]
                rows.append({"arch": name, "b": b,
                             "FN": (gv.y_pred == 0).mean() if len(gv) else np.nan,
                             "salahCWE": (gv.vuln & (gv.y_pred != 0) & (gv.y_true != gv.y_pred)).mean() if len(gv) else np.nan,
                             "FP": (gb.y_pred != 0).mean() if len(gb) >= 5 else np.nan})
    r = pd.DataFrame(rows)
    for name in ARCHS.values():
        out(f"\n=== {name} ===")
        for b in order:
            sub = r[(r.arch == name) & (r.b == b)]
            def ms(c):
                v = sub[c].dropna().values
                return f"{v.mean():.3f}+/-{v.std():.3f}" if len(v) else "-"
            out(f"  {b}: FN {ms('FN')}  salahCWE {ms('salahCWE')}  FP {ms('FP')}")


# ── IV.26 trade-off ambang confidence ───────────────────────────────────────
def iv26():
    out("\n########## IV.26 trade-off ambang confidence (mean+/-std, 3 seed) ##########")
    THR = [0.0, 0.4, 0.6, 0.8]
    for key, name in ARCHS.items():
        out(f"\n=== {name} ===")
        for t in THR:
            fp, fn, cw = [], [], []
            for s in SEEDS:
                d = pd.read_csv(pred_path(key, s))
                v, b = d[d.y_true != 0], d[d.y_true == 0]
                fv = (v.y_pred != 0) & (v.confidence >= t)
                fb = (b.y_pred != 0) & (b.confidence >= t)
                fp.append(fb.mean()); fn.append(1 - fv.mean())
                cw.append((fv & (v.y_true == v.y_pred)).sum() / max(fv.sum(), 1))
            f = lambda a: (np.mean(a), np.std(a, ddof=1))
            (a1, a2), (c1, c2), (e1, e2) = f(fp), f(fn), f(cw)
            out(f"  {t}: FP {a1:.3f}+/-{a2:.3f}  FN {c1:.3f}+/-{c2:.3f}  CWE {e1:.3f}+/-{e2:.3f}")


if __name__ == "__main__":
    iv22()
    iv23()
    iv25()
    iv26()
    with open("results/rm2_by_arch.txt", "w", encoding="utf-8") as f:
        f.write(LOG.getvalue())
    out("\nwrote results/rm2_by_arch.txt")
