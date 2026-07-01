#!/usr/bin/env python3
"""Collect baseline train_efficiency.json (written by lib_timer.sh) into one table.

Point --root at a dir holding extracted baseline run folders (each with a
train_efficiency.json). Groups by baseline name (RUN_ID prefix before the seed)
and reports mean total time + peak VRAM across seeds, like the arch efficiency table.

  python scripts/collect_baseline_efficiency.py --root baseline_runs
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def fmt_hms(s: float) -> str:
    s = int(round(s))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}"


def base_name(run_id: str) -> str:
    # strip trailing _s<seed>_<timestamp> / _s<seed> / _<timestamp> so seeds group together
    n = re.sub(r"_s\d+(_\d{8}_\d{6})?$", "", run_id)
    n = re.sub(r"_\d{8}_\d{6}$", "", n)
    return n.split("_megavul")[0].split("_bigvul")[0] or n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir with extracted baseline run folders")
    ap.add_argument("--glob", default="**/train_efficiency.json")
    a = ap.parse_args()

    groups: dict[str, list[dict]] = defaultdict(list)
    for p in Path(a.root).glob(a.glob):
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print(f"skip {p}: {e}")
            continue
        run_id = p.parent.name
        d["_run"] = run_id
        groups[base_name(run_id)].append(d)

    if not groups:
        print(f"no train_efficiency.json under {a.root}")
        return

    print("| Baseline | GPU | Waktu latih (h:m:s) | Peak VRAM (GB) | n |")
    print("|---|---|---|---|---|")
    for name in sorted(groups):
        runs = groups[name]
        times = [float(r.get("total_time_s", 0)) for r in runs]
        vram = [float(r.get("peak_vram_mib", 0)) / 1024 for r in runs]
        gpu = runs[0].get("gpu", "unknown")
        t = f"{fmt_hms(mean(times))}"
        if len(times) > 1:
            t += f" (±{fmt_hms(stdev(times))})"
        v = f"{mean(vram):.1f}"
        if len(vram) > 1:
            v += f" ±{stdev(vram):.1f}"
        print(f"| {name} | {gpu} | {t} | {v} | {len(runs)} |")


if __name__ == "__main__":
    main()
