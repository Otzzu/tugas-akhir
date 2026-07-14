"""Where do the ~16 s/function of CPG generation actually go?

Decomposes the cost with the stock CLI only — no Joern internals, so the numbers survive a
version bump:

    boot     = joern-parse --help                 (JVM start + class load)
    parse    = joern-parse --nooverlays  - boot   (Eclipse CDT frontend)
    overlay  = joern-parse               - parse  (CFG / CDG / REACHING_DEF)
    export   = joern-export                       (CPG -> GraphML)

Also re-runs the full parse with --max-num-def 100 (default 4000): if overlay dominates and
this knob moves it, the dataflow fixpoint is the lever and no runtime rewrite is needed.

Node/edge counts of the --nooverlays graph vs the full one say what the overlays actually add —
the graph our models were trained on is the FULL one.

    uv run python scripts/profile_joern.py --joern-cli C:/joern/joern-cli --reps 3
"""
from __future__ import annotations

import argparse
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

PCTS = [0.10, 0.50, 0.90, 0.99]


def _bin(joern_cli: Path, name: str) -> str:
    exe = joern_cli / (f"{name}.bat" if sys.platform == "win32" else name)
    if not exe.exists():
        sys.exit(f"not found: {exe}")
    return str(exe)


def _timed(cmd: list[str]) -> float:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        print(f"    ! rc={p.returncode} {p.stderr.strip().splitlines()[-1:]}")
    return dt


def _export(export_bin: str, cpg: Path, out: Path) -> float:
    shutil.rmtree(out, ignore_errors=True)   # joern-export refuses an existing --out
    # same argv as joern_runner.export_cpg — cpg positional LAST — or we time a different command
    return _timed([export_bin, "--repr", "all", "--format", "graphml", "--out", str(out), str(cpg)])


def _graphml_size(d: Path) -> tuple[int, int]:
    n = e = 0
    for f in d.rglob("*.xml"):
        t = f.read_text(encoding="utf-8", errors="ignore")
        n += len(re.findall(r"<node ", t))
        e += len(re.findall(r"<edge ", t))
    return n, e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joern-cli", default="C:/joern/joern-cli", type=Path)
    ap.add_argument("--parquet", default="data/datasets/megavul/train.parquet", type=Path)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    parse_bin = _bin(args.joern_cli, "joern-parse")
    export_bin = _bin(args.joern_cli, "joern-export")

    df = pd.read_parquet(args.parquet, columns=["func_before", "language"])
    df = df[df.language.isin(["C", "C++"])]
    lens = df.func_before.str.len()

    work = Path(tempfile.mkdtemp(prefix="joern_prof_"))
    samples = []
    for q in PCTS:
        target = lens.quantile(q)
        row = df.loc[(lens - target).abs().idxmin()]
        f = work / f"p{int(q * 100):02d}.c"
        f.write_text(row.func_before, encoding="utf-8")
        samples.append((f"p{int(q * 100)}", f, len(row.func_before)))

    print(f"joern : {args.joern_cli}\nreps  : {args.reps}\nwork  : {work}\n")

    boot = statistics.median(_timed([parse_bin, "--help"]) for _ in range(args.reps))
    print(f"boot (JVM + class load)  {boot:6.2f}s\n")

    hdr = f"{'func':>5} {'chars':>6} | {'parse':>7} {'overlay':>8} {'export':>7} {'TOTAL':>7} | {'ovl@def100':>10} | nodes/edges"
    print(hdr)
    print("-" * len(hdr))

    tot = {"parse": [], "overlay": [], "export": []}
    for tag, f, nchars in samples:
        out_no, out_full = work / f"{tag}_no.bin", work / f"{tag}_full.bin"

        t_no = statistics.median(
            _timed([parse_bin, str(f), "-o", str(out_no), "--nooverlays"]) for _ in range(args.reps))
        t_full = statistics.median(
            _timed([parse_bin, str(f), "-o", str(out_full)]) for _ in range(args.reps))
        t_def = statistics.median(
            _timed([parse_bin, str(f), "-o", str(work / f"{tag}_def.bin"), "--max-num-def", "100"])
            for _ in range(args.reps))

        xd, xd_no = work / f"{tag}_xml", work / f"{tag}_xml_no"
        t_exp = statistics.median(_export(export_bin, out_full, xd) for _ in range(args.reps))
        _export(export_bin, out_no, xd_no)
        n_full, e_full = _graphml_size(xd)
        n_no, e_no = _graphml_size(xd_no)

        parse = max(t_no - boot, 0.0)
        overlay = max(t_full - t_no, 0.0)
        ovl_def = max(t_def - t_no, 0.0)
        for k, v in (("parse", parse), ("overlay", overlay), ("export", t_exp)):
            tot[k].append(v)

        print(f"{tag:>5} {nchars:>6} | {parse:6.2f}s {overlay:7.2f}s {t_exp:6.2f}s {t_full + t_exp:6.2f}s "
              f"| {ovl_def:9.2f}s | {n_no}/{e_no} -> {n_full}/{e_full}")

    p, o, e = (statistics.median(tot[k]) for k in ("parse", "overlay", "export"))
    work_only = p + o + e
    print(f"\nmedian: boot {boot:.2f}s | parse {p:.2f}s | overlay {o:.2f}s | export {e:.2f}s")
    print(f"boot is {boot / (boot + work_only) * 100:.0f}% of a cold call — a warm process deletes exactly that.")
    print(f"of the {work_only:.2f}s that a warm process still pays: "
          f"parse {p / work_only * 100:.0f}%, overlay {o / work_only * 100:.0f}%, export {e / work_only * 100:.0f}%")
    print(f"\nkeep: {work}")


if __name__ == "__main__":
    main()
