"""
merge.py
~~~~~~~~
Merge several ALREADY-BUILT processed datasets (.pt) into one combined
dataset at the .pt level — concatenate graphs + unify the label space.
No Joern, no re-embedding, no raw CPG: sources must already exist under
`<root>/processed` (built by CodeBERTGraphDataset / build_pt / process_dataset).

CLI
---
    python -m gnn_vuln.data.merge --config <yaml...> \
        --sources <s1> <s2> ... --out-source <name> [--dedup] [--device cpu]

Storage follows cfg.data.storage. inmemory writes one <name>.pt holding every graph.
lazy writes <name>_meta.pt plus a <name>_graphs/{i}.pt file per graph, streamed to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from loguru import logger

from gnn_vuln.config import Config, load_default_config
from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset


def _build_ds(root, source, cfg, device) -> CodeBERTGraphDataset:
    """Construct (load) a CodeBERTGraphDataset for `source` with cfg-derived params."""
    return CodeBERTGraphDataset(
        root=str(root),
        source=source,
        max_nodes=cfg.data.max_nodes,
        embedder_device=device,
        mode=cfg.data.mode,
        pretrained_lm=getattr(cfg.model, "pretrained_lm", "microsoft/codebert-base"),
        func_lm=getattr(cfg.model, "func_lm", ""),
        add_func_tokens=getattr(cfg.model, "add_func_tokens", False),
        func_lm_source=getattr(cfg.model, "func_lm_source", "raw"),
        func_max_length=getattr(cfg.model, "func_max_length", 512),
        storage=getattr(cfg.data, "storage", "inmemory"),
        cwe_list=getattr(cfg.data, "cwe_list", None),
        cwe_groups=getattr(cfg.data, "cwe_groups", None),
        filter_owasp=getattr(cfg.data, "filter_owasp", False),
        filter_top25_dangerous=getattr(cfg.data, "filter_top25_dangerous", False),
        top_cwe=getattr(cfg.data, "top_cwe", 0),
        max_per_class=getattr(cfg.data, "max_per_class", 0),
        resample_seed=getattr(cfg.data, "resample_seed", 42),
        ds_name=source if getattr(cfg.data, "ds_name", "") else "",
    )


def _func_hash(g) -> str | None:
    raw = getattr(g, "raw_func", None)
    if not raw:
        return None
    return hashlib.md5(raw.encode("utf-8", "ignore")).hexdigest()


def merge_processed(cfg, sources, out_source, dedup=False, device="cpu") -> dict:
    storage = getattr(cfg.data, "storage", "inmemory")
    if storage not in ("inmemory", "lazy"):
        raise ValueError(f"unknown storage {storage!r} (expected 'inmemory' or 'lazy')")

    root = Path(cfg.data.processed_dir).parent

    # 1. Load each source.
    datasets, src_class_names = [], []
    for s in sources:
        ds = _build_ds(root, s, cfg, device)
        cn = ds.class_names or ["benign", "vulnerable"]
        datasets.append(ds)
        src_class_names.append(cn)
        logger.info(f"  source {s!r}: {len(ds)} graphs, {len(cn)} classes")

    # 2. Unified vocab: benign at 0, other labels sorted deterministically.
    others = sorted({c for cn in src_class_names for c in cn if c != "benign"})
    unified_names = ["benign"] + others
    unified = {name: i for i, name in enumerate(unified_names)}
    logger.info(f"unified vocab: {len(unified_names)} classes")

    # 5. Save to the exact path CodeBERTGraphDataset(source=out_source) resolves to.
    #    lazy writes each graph straight to disk, so the whole merge never sits in RAM.
    out_path = _out_processed_path(root, out_source, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    graphs_dir = None
    if storage == "lazy":
        graphs_dir = out_path.parent / f"{out_path.name[:-len('_meta.pt')]}_graphs"
        if graphs_dir.exists():
            shutil.rmtree(graphs_dir)
        graphs_dir.mkdir(parents=True)

    # 3/4. Remap each graph's y to the unified index, concatenate, optional dedup.
    merged, seen = [], set()
    n_dupes = n_graphs = 0
    for ds, cn in zip(datasets, src_class_names):
        for i in range(len(ds)):
            g = ds[i]
            if dedup:
                h = _func_hash(g)
                if h is not None:
                    if h in seen:
                        n_dupes += 1
                        continue
                    seen.add(h)
            old_idx = int(g.y)
            new_idx = unified[cn[old_idx]]
            g.y = torch.tensor([new_idx], dtype=g.y.dtype).reshape(g.y.shape)
            if graphs_dir is not None:
                torch.save(g, graphs_dir / f"{n_graphs}.pt")
            else:
                merged.append(g)
            n_graphs += 1

    sample = merged[0] if merged else (g if n_graphs else None)
    fp = datasets[0]._fingerprint(sample)
    fp["source"] = out_source
    meta = {"n_graphs": n_graphs, "class_names": unified_names, "fingerprint": fp}
    if graphs_dir is None:
        meta["graphs"] = merged
    torch.save(meta, out_path)

    # Unified vocab → raw/<out_source>/cwe_vocab.json as {name: idx}.
    vocab_dir = root / "raw" / out_source
    vocab_dir.mkdir(parents=True, exist_ok=True)
    (vocab_dir / "cwe_vocab.json").write_text(
        json.dumps(unified, indent=2), encoding="utf-8"
    )

    logger.info(f"merged → {out_path}")
    return {
        "num_classes": len(unified_names),
        "n_graphs": n_graphs,
        "n_duplicates_removed": n_dupes,
        "out_path": str(out_path),
        "class_names": unified_names,
    }


def _out_processed_path(root, out_source, cfg) -> Path:
    """Resolve the path CodeBERTGraphDataset(source=out_source, <cfg params>) would
    load WITHOUT triggering a build (__init__ eager-loads/builds processed_paths[0]).
    Reuses the real `_ds_name` property by setting only the attrs it reads.
    """
    from gnn_vuln.data.dataset_lm import _filter_suffix

    obj = CodeBERTGraphDataset.__new__(CodeBERTGraphDataset)
    obj._storage = getattr(cfg.data, "storage", "inmemory")
    obj._ds_name_suffix = ""
    obj._ds_name_explicit = getattr(cfg.data, "ds_name", "") or ""   # caller-owned name wins
    obj._add_func_tokens = getattr(cfg.model, "add_func_tokens", False)
    obj._func_max_length = getattr(cfg.model, "func_max_length", 512)
    obj._top_cwe = getattr(cfg.data, "top_cwe", 0)
    obj._max_per_class = getattr(cfg.data, "max_per_class", 0)
    obj._resample_seed = getattr(cfg.data, "resample_seed", 42)
    obj._mode = cfg.data.mode
    obj._source = out_source
    lm = getattr(cfg.model, "pretrained_lm", "microsoft/codebert-base")
    obj._lm_short = lm.split("/")[-1]
    func_lm = getattr(cfg.model, "func_lm", "") or lm
    obj._func_short = func_lm.split("/")[-1]
    # the XML presets are expanded into cwe_list BEFORE hashing (dataset_lm does the same),
    # otherwise merge would write a name the loader never looks for
    from gnn_vuln.data.dataset_lm import _expand_xml_filters
    fowasp = getattr(cfg.data, "filter_owasp", False)
    ftop25 = getattr(cfg.data, "filter_top25_dangerous", False)
    cwe_groups = getattr(cfg.data, "cwe_groups", None)
    cwe_list = _expand_xml_filters(root, getattr(cfg.data, "cwe_list", None), fowasp, ftop25)
    obj._fsuffix = _filter_suffix(cwe_list or None, cwe_groups, fowasp, ftop25)

    name = obj._ds_name
    processed = Path(root) / "processed"
    fname = f"{name}.pt" if obj._storage == "inmemory" else f"{name}_meta.pt"
    return processed / fname


def _load_cfg(cfg_paths):
    cfg_paths = cfg_paths if isinstance(cfg_paths, (list, tuple)) else [cfg_paths]
    return (Config.from_yamls(cfg_paths)
            if all(Path(p).exists() for p in cfg_paths) else load_default_config())


def main() -> None:
    p = argparse.ArgumentParser(description="Merge built .pt datasets into one (label-space unified).")
    p.add_argument("--config", required=True, nargs="+",
                   help="One YAML (monolithic) or several split files merged in order.")
    p.add_argument("--sources", required=True, nargs="+",
                   help="Source names already built under <root>/processed.")
    p.add_argument("--out-source", required=True, help="Name for the merged dataset.")
    p.add_argument("--dedup", action="store_true", help="Best-effort drop duplicate functions (by raw_func hash).")
    p.add_argument("--device", default="cpu", help="Device passed to CodeBERTGraphDataset (cpu/cuda).")
    args = p.parse_args()

    cfg = _load_cfg(args.config)
    res = merge_processed(cfg, args.sources, args.out_source, dedup=args.dedup, device=args.device)

    print(f"num_classes          : {res['num_classes']}")
    print(f"n_graphs             : {res['n_graphs']}")
    print(f"n_duplicates_removed : {res['n_duplicates_removed']}")
    print(f"out_path             : {res['out_path']}")


if __name__ == "__main__":
    main()
