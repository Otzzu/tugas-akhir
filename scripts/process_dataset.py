"""
process_dataset.py
~~~~~~~~~~~~~~~~~~
Generate processed .pt dataset cache without starting training.
Useful for pre-computing embeddings locally before transferring to cloud.

Usage:
    uv run python scripts/process_dataset.py --config configs/lmgat/binary_graphcodebert.yaml
    uv run python scripts/process_dataset.py --config configs/lmgat_codebert/multiclass.yaml --device cuda
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
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else load_default_config()

    pretrained_lm = getattr(cfg.model, "pretrained_lm", "microsoft/codebert-base")
    add_func_tokens = getattr(cfg.model, "add_func_tokens", False)
    device = args.device or str(cfg.train.device)

    print(f"Config      : {args.config}")
    print(f"Mode        : {cfg.data.mode}")
    print(f"LM          : {pretrained_lm}")
    print(f"func_tokens : {add_func_tokens}")
    print(f"Device      : {device}")
    print(f"Max nodes   : {cfg.data.max_nodes}")
    print()

    dataset = CodeBERTGraphDataset(
        root=str(cfg.data.processed_dir.parent),
        max_nodes=cfg.data.max_nodes,
        embedder_device=device,
        mode=cfg.data.mode,
        pretrained_lm=pretrained_lm,
        add_func_tokens=add_func_tokens,
    )

    print(f"\nDone: {len(dataset)} graphs processed and cached.")
    print(f"File: data/processed/lm_dataset_{cfg.data.mode}_{pretrained_lm.split('/')[-1]}{'_ft' if add_func_tokens else ''}.pt")


if __name__ == "__main__":
    main()
