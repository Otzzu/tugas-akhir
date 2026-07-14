"""Serving contract — the three production models, one module each.

Each declares DESIGN (Bab III, locked) and Params (sizing, tunable). The config is the
contract: the ablation surface is unreachable and an unknown key raises, unlike the research
path where Config._from_raw setattr's any key without validating it.

    cfg = build_config("graph_based",
                       data=DataParams(source="megavul_26"),
                       train=TrainParams(epochs=50))

Architecture code is not duplicated — all three build the same model class, so existing
checkpoints still load.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from gnn_vuln.config import Config, EWCConfig, ReplayConfig
from gnn_vuln.serving import graph_based, hybrid_graph_lm, sequential
from gnn_vuln.serving._common import DataParams, TrainParams, to_raw

MODELS = {m.NAME: m for m in (graph_based, hybrid_graph_lm, sequential)}

__all__ = ["MODELS", "DataParams", "TrainParams", "build_config", "design_of", "params_of"]


def _as_raw(section: str, value, expected) -> dict:
    """Accept the section's dataclass, or a plain dict validated against it."""
    if value is None:
        return {}
    if is_dataclass(value):
        if not isinstance(value, expected):
            raise TypeError(f"{section}: expected {expected.__name__}, got {type(value).__name__}")
        return to_raw(value)
    known = {f.name for f in fields(expected)}
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"{section}: unknown key(s) {unknown}. Allowed: {sorted(known)}")
    return dict(value)


def build_config(
    arch: str,
    *,
    data: DataParams | dict,
    params: Any | dict | None = None,
    train: TrainParams | dict | None = None,
    ewc: dict | None = None,
    replay: dict | None = None,
) -> Config:
    """Compose a Config from a model's locked DESIGN plus its tunable Params."""
    if arch not in MODELS:
        raise ValueError(f"unknown model {arch!r}. Known: {sorted(MODELS)}")
    mod = MODELS[arch]

    raw_params = _as_raw("params", params, mod.Params)
    locked = sorted(k for k in raw_params if k in mod.DESIGN)
    if locked:
        raise ValueError(
            f"params: {locked} are design decisions of '{arch}' (Bab III) and cannot be "
            f"overridden. Change {mod.__name__}.DESIGN, not the caller."
        )

    model_section = {**mod.DESIGN, **to_raw(mod.Params()), **raw_params}
    for section, val, cls in (("ewc", ewc, EWCConfig), ("replay", replay, ReplayConfig)):
        if val:
            _as_raw(section, val, cls)

    return Config._from_raw({
        "data": _as_raw("data", data, DataParams),
        "model": model_section,
        "train": _as_raw("train", train, TrainParams),
        "ewc": ewc or {},
        "replay": replay or {},
    })


def design_of(arch: str) -> dict[str, Any]:
    """Locked design values for `arch`."""
    return dict(MODELS[arch].DESIGN)


def params_of(arch: str) -> dict[str, Any]:
    """Default tunables for `arch`."""
    return to_raw(MODELS[arch].Params())
