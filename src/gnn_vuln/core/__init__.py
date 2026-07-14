"""Path-first dataset access. Core never derives names, never rebuilds — the caller says
which file, core says whether it fits."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

_META_KEYS = ("n_graphs", "class_names", "fingerprint")


def _split(path: Path) -> tuple[str, str]:
    name = path.name
    if name.endswith("_meta.pt"):
        return name[: -len("_meta.pt")], "lazy"
    if name.endswith(".pt"):
        return name[: -len(".pt")], "inmemory"
    raise ValueError(f"not a dataset file: {path}")


def read_meta(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    return {k: loaded.get(k) for k in _META_KEYS}


def read_fingerprint(path: str | Path) -> dict | None:
    """Stored build identity, or None for datasets built before fingerprints existed."""
    return read_meta(path).get("fingerprint")


def infer_fingerprint(ds) -> dict:
    """Best-effort identity from tensors of a loaded dataset (pre-fingerprint files)."""
    fp: dict[str, Any] = {"schema": 0}
    if len(ds):
        g = ds[0]
        fp["node_feat_dim"] = int(g.x.shape[1])
        ea = getattr(g, "edge_attr", None)
        if ea is not None and ea.dim() == 2:
            fp["edge_dim"] = int(ea.shape[1])
        fii = getattr(g, "func_input_ids", None)
        if fii is not None:
            fp["func_max_length"] = int(fii.shape[-1])
            fp["add_func_tokens"] = True
    if ds.class_names:
        fp["num_classes"] = len(ds.class_names)
    return fp


def open_dataset(path: str | Path, *, device: str = "cpu", target_vocab: dict | None = None):
    """Open a BUILT dataset by its file path. Missing file raises, nothing is rebuilt."""
    from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset

    path = Path(path)
    name, storage = _split(path)
    meta = read_meta(path)
    fp = meta.get("fingerprint") or {}
    if storage == "lazy" and not (path.parent / f"{name}_graphs").is_dir():
        raise FileNotFoundError(f"lazy dataset {name} has no graphs dir next to {path}")

    return CodeBERTGraphDataset(
        root=str(path.parent.parent),
        source=fp.get("source", name),
        ds_name=name,
        storage=storage,
        mode=fp.get("mode", "multiclass"),
        max_nodes=fp.get("max_nodes", 2500),
        pretrained_lm=fp.get("pretrained_lm", "microsoft/codebert-base"),
        func_lm=fp.get("func_lm", ""),
        add_func_tokens=fp.get("add_func_tokens", False),
        func_max_length=fp.get("func_max_length", 512),
        embedder_device=device,
        target_vocab=target_vocab,
    )


def check_compatible(required: dict, path_or_ds) -> list[str]:
    """Compare a model's requirements against a dataset's identity. Returns mismatch
    strings, empty = compatible. Fields absent on either side are skipped."""
    if isinstance(path_or_ds, (str, Path)):
        fp = read_fingerprint(path_or_ds) or {}
    else:
        fp = getattr(path_or_ds, "fingerprint", None) or infer_fingerprint(path_or_ds)

    problems = []
    for key, want in required.items():
        have = fp.get(key)
        if have is not None and have != want:
            problems.append(f"{key}: dataset has {have!r}, model requires {want!r}")
    if not fp:
        problems.append("no fingerprint stored and none inferable")
    return problems
