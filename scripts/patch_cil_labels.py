"""
patch_cil_labels.py
~~~~~~~~~~~~~~~~~~~~
Remap the megavul_cil lazy .pt labels for class-incremental (CIL) task-B.

The cil dataset is built standalone with its own contiguous labels: benign=0,
and the 10 new CWEs = 1..10. For CIL we append these as classes 26..35 after
task-A's 26 (0..25), so the unified 36-class model can be evaluated on both
old and new classes. This rewrites every graph's y (j → 25+j) and sets a
36-entry class_names (task-A 26 + the 10 new) so dataset.num_classes == 36.

Idempotent: re-running is a no-op once class_names already has 36 entries.

Run:
    uv run python scripts/patch_cil_labels.py
"""
from __future__ import annotations
import os
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
_BACKBONE = "unixcoder-base-nine" if os.environ.get("RELEARN_NINE") else "unixcoder-base"
DS = f"lm_dataset_megavul_cil_multiclass_{_BACKBONE}_ft_ml1024"

# Task-A 26-class order (= MegaVul dataset class_names, indices 0..25).
TASKA_26 = [
    "benign", "CWE-787", "CWE-125", "CWE-476", "CWE-20", "CWE-416", "CWE-200",
    "CWE-120", "CWE-79", "CWE-22", "CWE-89", "CWE-770", "CWE-502", "CWE-122",
    "CWE-284", "CWE-863", "CWE-78", "CWE-862", "CWE-94", "CWE-918", "CWE-434",
    "CWE-352", "CWE-77", "CWE-121", "CWE-306", "CWE-639",
]
OFFSET = len(TASKA_26) - 1   # benign(0) excluded; cil CWE j(1..10) → 25+j (26..35)


def main() -> None:
    meta_path = PROC / f"{DS}_meta.pt"
    graphs_dir = PROC / f"{DS}_graphs"
    if not meta_path.exists():
        raise FileNotFoundError(f"cil meta not found: {meta_path}")
    meta = torch.load(meta_path, weights_only=False)
    names = meta.get("class_names") or []
    n = meta["n_graphs"]

    if len(names) == len(TASKA_26) + 10:
        print(f"Already patched (class_names has {len(names)} entries). No-op.")
        return
    cil_cwe = names[1:]   # drop benign → the 10 new CWEs in cil order
    assert names[0] == "benign" and len(cil_cwe) == 10, \
        f"unexpected cil class_names: {names}"

    full_names = TASKA_26 + cil_cwe   # 0..25 task-A, 26..35 new
    print(f"cil new classes (-> 26..35): {cil_cwe}")
    print(f"patching {n} graphs: y(j) -> {OFFSET}+j ...")

    bad = 0
    for i in range(n):
        gp = graphs_dir / f"{i}.pt"
        g = torch.load(gp, weights_only=False)
        y = int(g.y)
        if y == 0:        # benign — should not occur (0 samples); skip remap
            bad += 1
            continue
        g.y = g.y + OFFSET           # tensor add preserves shape + dtype
        torch.save(g, gp)

    meta["class_names"] = full_names
    torch.save(meta, meta_path)
    print(f"done. benign/skipped={bad}. class_names now {len(full_names)} entries.")
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
