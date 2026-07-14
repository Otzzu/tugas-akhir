"""Path-first dataset access. Core never derives names, never rebuilds — the caller says
which file, core says whether it fits."""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# pyarrow MUST load before torch: on Windows the reverse order makes the first read_parquet
# segfault (ACCESS_VIOLATION) — the two ship conflicting DLLs. The research pipeline never hit
# this because it reads rows and builds tensors in two separate subprocesses.
import pandas  # noqa: F401
import torch

_META_KEYS = ("n_graphs", "class_names", "fingerprint")


@dataclass
class DatasetInfo:
    path: Path
    n_graphs: int
    class_names: list[str]
    fingerprint: dict
    cwe_vocab: dict[str, int]
    n_cpg_failed: int = 0


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


def build_dataset(
    rows: str | Path,
    out_path: str | Path,
    *,
    featurization: dict,
    joern_cli: str | Path | None = None,
    cwe_vocab: dict[str, int] | None = None,
    max_nodes: int = 2500,
    mode: str = "multiclass",
    storage: str = "inmemory",
    workers: int = 4,
    device: str = "cpu",
    keep_cpg: str | Path | None = None,
) -> DatasetInfo:
    """Rows -> CPG -> .pt, in one call, at the path YOU name.

    The vocab is a value passed from step to step, not a file dropped on disk for the next
    subprocess to find: pass `cwe_vocab` to reuse a parent dataset's class space, or let it be
    derived from the rows. Nothing here derives a name or looks one up.
    """
    from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset
    from gnn_vuln.data.prepare import build_cpgs, load_api

    out_path = Path(out_path)
    name = out_path.name[: -len("_meta.pt")] if out_path.name.endswith("_meta.pt") \
        else out_path.stem
    root = out_path.parent.parent                       # <root>/processed/<name>.pt

    df, vocab = load_api(Path(rows), top_k_cwe=0, binary=(mode == "binary"),
                         cwe_vocab=cwe_vocab)

    cpg_root = Path(keep_cpg) if keep_cpg else Path(tempfile.mkdtemp(prefix="gv_cpg_"))
    cpg_dir = cpg_root / name                           # dataset_lm reads <root>/raw/<source>
    raw_link = root / "raw" / name
    try:
        ok, failed = build_cpgs(df, cpg_dir, joern_cli=Path(joern_cli) if joern_cli else None,
                                workers=workers, is_multi_class=(mode != "binary"))
        if ok == 0:
            raise RuntimeError(f"Joern produced no CPG for any of the {len(df)} rows")

        raw_link.parent.mkdir(parents=True, exist_ok=True)
        if raw_link.exists():
            shutil.rmtree(raw_link)
        shutil.move(str(cpg_dir), str(raw_link))

        ds = CodeBERTGraphDataset(
            root=str(root), source=name, ds_name=name, storage=storage, mode=mode,
            max_nodes=max_nodes, embedder_device=device, cwe_vocab=vocab,
            pretrained_lm=featurization["pretrained_lm"],
            func_lm=featurization.get("func_lm", ""),
            add_func_tokens=featurization.get("add_func_tokens", True),
            func_max_length=featurization.get("func_max_length", 1024),
            func_lm_source=featurization.get("func_lm_source", "raw"),
        )
        built = read_meta(out_path)
        return DatasetInfo(
            path=out_path, n_graphs=int(built["n_graphs"] or 0),
            class_names=list(built["class_names"] or []),
            fingerprint=built.get("fingerprint") or {},
            cwe_vocab=vocab, n_cpg_failed=failed,
        )
    finally:
        if not keep_cpg:
            shutil.rmtree(cpg_root, ignore_errors=True)
            shutil.rmtree(raw_link, ignore_errors=True)


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
