"""
process_dataset.py
~~~~~~~~~~~~~~~~~~
Generate processed .pt dataset cache without starting training.
Useful for pre-computing embeddings locally before transferring to cloud.

Usage:
    uv run python scripts/process_dataset.py --config configs/lmgat/top1_cwe.yaml
    uv run python scripts/process_dataset.py --config configs/lmgat/top1_cwe.yaml --device cuda
    uv run python scripts/process_dataset.py --config configs/lmgat/top1_cwe.yaml --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gnn_vuln.config import Config, load_default_config
from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute dataset .pt cache from config.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--device", default=None, help="Device for embedding (cpu/cuda). Defaults to config value.")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which source split to build (default: train)",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else load_default_config()

    pretrained_lm  = getattr(cfg.model, "pretrained_lm", "microsoft/codebert-base")
    add_func_tokens = getattr(cfg.model, "add_func_tokens", False)
    device         = args.device or str(cfg.train.device)
    top_cwe        = getattr(cfg.data, "top_cwe", 0)
    cwe_list       = getattr(cfg.data, "cwe_list", None)
    cwe_groups     = getattr(cfg.data, "cwe_groups", None)
    max_per_class  = getattr(cfg.data, "max_per_class", 0)
    resample_seed  = getattr(cfg.data, "resample_seed", 42)

    source_map = {
        "train": getattr(cfg.data, "source",      "bigvul"),
        "val":   getattr(cfg.data, "source_val",  ""),
        "test":  getattr(cfg.data, "source_test", ""),
    }
    source = source_map[args.split]
    if not source:
        print(f"No source configured for split='{args.split}' in {args.config}")
        raise SystemExit(1)

    print(f"Config        : {args.config}")
    print(f"Split         : {args.split}  source={source}")
    print(f"Mode          : {cfg.data.mode}")
    print(f"LM            : {pretrained_lm}")
    print(f"func_tokens   : {add_func_tokens}")
    print(f"Device        : {device}")
    print(f"Max nodes     : {cfg.data.max_nodes}")
    print(f"top_cwe       : {top_cwe if top_cwe > 0 else 'all'}")
    print(f"cwe_list      : {cwe_list}")
    print(f"cwe_groups    : {cwe_groups}")
    print(f"max_per_class : {max_per_class if max_per_class > 0 else 'unlimited'}")
    if max_per_class > 0:
        print(f"resample_seed : {resample_seed}")
    print()

    dataset = CodeBERTGraphDataset(
        root=str(cfg.data.processed_dir.parent),
        source=source,
        max_nodes=cfg.data.max_nodes,
        embedder_device=device,
        mode=cfg.data.mode,
        pretrained_lm=pretrained_lm,
        add_func_tokens=add_func_tokens,
        top_cwe=top_cwe,
        cwe_list=cwe_list,
        cwe_groups=cwe_groups,
        max_per_class=max_per_class,
        resample_seed=resample_seed,
    )

    print(f"\nDone: {len(dataset)} graphs cached.")
    print(f"num_classes   : {dataset.num_classes}")
    if dataset.class_names:
        print(f"class_names   : {dataset.class_names}")
    print(f"File          : {dataset.processed_paths[0]}")


if __name__ == "__main__":
    main()
