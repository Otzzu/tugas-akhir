"""Konversi satu dataset lazy (<name>_meta.pt + <name>_graphs/) -> inmemory (<name>.pt).

Format inmemory wajib memuat seluruh graph di RAM, jadi pemakaian memori puncaknya
kira-kira sebesar folder graphs.

Usage:
    uv run python scripts/lazy_to_inmemory.py --meta data/processed/<name>_meta.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm


def convert(meta_path: Path, keep_lazy: bool = False) -> Path:
    name = meta_path.name[: -len("_meta.pt")]
    graphs_dir = meta_path.parent / f"{name}_graphs"
    if not graphs_dir.is_dir():
        raise FileNotFoundError(f"tidak ada {graphs_dir}")

    meta = torch.load(meta_path, weights_only=False)
    n = int(meta["n_graphs"])
    class_names = meta["class_names"]

    graphs = [
        torch.load(graphs_dir / f"{i}.pt", weights_only=False)
        for i in tqdm(range(n), desc=f"  load {name}", unit="graph")
    ]

    out = meta_path.parent / f"{name}.pt"
    torch.save({"n_graphs": n, "class_names": class_names, "graphs": graphs}, out)

    if not keep_lazy:
        meta_path.unlink()
        for f in graphs_dir.iterdir():
            f.unlink()
        graphs_dir.rmdir()

    size = out.stat().st_size / 1e9
    print(f"  {name}: {n} graph, {len(class_names)} kelas -> {out.name} ({size:.1f} GB)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, required=True, help="path ke <name>_meta.pt")
    ap.add_argument("--keep-lazy", action="store_true", help="jangan hapus sumber lazy")
    args = ap.parse_args()
    convert(args.meta, args.keep_lazy)


if __name__ == "__main__":
    main()
