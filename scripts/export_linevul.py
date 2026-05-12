"""
export_linevul.py — Export gnn_vuln processed dataset to LineVul CSV format.

LineVul expects three CSV files (train/val/test) with columns:
  processed_func   : raw C/C++ function text
  target           : binary label (0 = benign, 1 = vulnerable)
  flaw_line        : tab-separated actual code lines that are vulnerable (NaN if benign)
  flaw_line_index  : comma-separated 1-indexed line numbers of flaw lines (NaN if benign)

Uses the same 70/15/15 split with the same seed as your model so results
are directly comparable.

Usage:
    uv run python scripts/export_linevul.py --config configs/lmgat/multiclass.yaml
    uv run python scripts/export_linevul.py --config configs/lmgat/multiclass.yaml --out-dir data/baselines/linevul
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_vuln.config import Config
from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset


def build_row(dataset: CodeBERTGraphDataset, idx: int) -> dict:
    """Build one CSV row from a dataset index."""
    data = dataset[idx]

    # Raw function text
    raw_funcs = getattr(dataset, "raw_funcs", None)
    func_text = (raw_funcs[idx] or "") if raw_funcs and idx < len(raw_funcs) else ""

    # Binary label
    label = int(data.y.item())
    target = 1 if label > 0 else 0

    # Flaw line info — only available for vulnerable functions with GT
    flaw_line        = float("nan")
    flaw_line_index  = float("nan")

    if target == 1 and func_text:
        # Recover flaw line numbers from flaw_line_mask + node_line
        node_line      = getattr(data, "node_line", None)
        flaw_line_mask = getattr(data, "flaw_line_mask", None)

        if node_line is not None and flaw_line_mask is not None:
            # Get unique flaw line numbers (1-indexed, ignore -1 unknowns)
            flaw_nodes = (flaw_line_mask == 1) & (node_line >= 0)
            if flaw_nodes.any():
                flaw_line_nums = sorted(
                    node_line[flaw_nodes].unique().tolist()
                )
                flaw_line_index = ",".join(str(ln) for ln in flaw_line_nums)

                # Extract actual code text for those lines
                func_lines = func_text.splitlines()
                flaw_texts = []
                for ln in flaw_line_nums:
                    line_idx = ln - 1  # convert 1-indexed → 0-indexed
                    if 0 <= line_idx < len(func_lines):
                        flaw_texts.append(func_lines[line_idx].strip())
                if flaw_texts:
                    flaw_line = "\t".join(flaw_texts)

    return {
        "processed_func":  func_text,
        "target":          target,
        "flaw_line":       flaw_line,
        "flaw_line_index": flaw_line_index,
    }


def export_split(
    dataset: CodeBERTGraphDataset,
    indices: list[int],
    out_path: Path,
    split_name: str,
) -> None:
    rows = []
    for i, idx in enumerate(indices):
        if i % 500 == 0:
            print(f"  {split_name}: {i}/{len(indices)}", end="\r")
        rows.append(build_row(dataset, idx))

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    n_vul    = int((df["target"] == 1).sum())
    n_benign = int((df["target"] == 0).sum())
    n_flaw   = int(df["flaw_line"].notna().sum())
    print(
        f"  {split_name}: {len(df)} rows  "
        f"(vuln={n_vul}, benign={n_benign}, with_flaw_lines={n_flaw})  "
        f"→ {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export gnn_vuln dataset to LineVul CSV format"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to gnn_vuln YAML config (dataset settings + split seed)"
    )
    parser.add_argument(
        "--out-dir", default="data/baselines/linevul",
        help="Output directory for train.csv / val.csv / test.csv"
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    out_dir = PROJECT_ROOT / args.out_dir

    pretrained_lm = getattr(cfg.model, "pretrained_lm", "microsoft/codebert-base")
    func_lm       = getattr(cfg.model, "func_lm", "") or pretrained_lm

    print("Loading dataset…")
    dataset = CodeBERTGraphDataset(
        root=str(cfg.data.processed_dir.parent),
        max_nodes=cfg.data.max_nodes,
        embedder_device="cpu",
        mode=cfg.data.mode,
        source=getattr(cfg.data, "source", "bigvul"),
        pretrained_lm=pretrained_lm,
        func_lm=func_lm,
        add_func_tokens=False,
        top_cwe=getattr(cfg.data, "top_cwe", 0),
        cwe_list=getattr(cfg.data, "cwe_list", None),
        cwe_groups=getattr(cfg.data, "cwe_groups", None),
        filter_owasp=getattr(cfg.data, "filter_owasp", False),
        filter_top25_dangerous=getattr(cfg.data, "filter_top25_dangerous", False),
        max_per_class=getattr(cfg.data, "max_per_class", 0),
        resample_seed=getattr(cfg.data, "resample_seed", 42),
    )

    train_idx, val_idx, test_idx = dataset.get_splits(seed=cfg.train.seed)
    print(
        f"Dataset: {len(dataset)} graphs | "
        f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}"
    )

    print("\nExporting splits…")
    export_split(dataset, list(train_idx), out_dir / "train.csv", "train")
    export_split(dataset, list(val_idx),   out_dir / "val.csv",   "val")
    export_split(dataset, list(test_idx),  out_dir / "test.csv",  "test")

    print(f"\nDone. Files written to {out_dir}/")
    print("\nTo train LineVul on this data:")
    print(f"  cd src/LineVul/linevul")
    print(f"  python linevul_main.py \\")
    print(f"    --output_dir=./saved_models \\")
    print(f"    --model_type=roberta \\")
    print(f"    --tokenizer_name=microsoft/codebert-base \\")
    print(f"    --model_name_or_path=microsoft/codebert-base \\")
    print(f"    --do_train --do_test \\")
    print(f"    --train_data_file={out_dir}/train.csv \\")
    print(f"    --eval_data_file={out_dir}/val.csv \\")
    print(f"    --test_data_file={out_dir}/test.csv \\")
    print(f"    --epochs 10 \\")
    print(f"    --block_size 512 \\")
    print(f"    --train_batch_size 16 \\")
    print(f"    --eval_batch_size 16 \\")
    print(f"    --learning_rate 2e-5 \\")
    print(f"    --max_grad_norm 1.0 \\")
    print(f"    --evaluate_during_training \\")
    print(f"    --seed {cfg.train.seed}")


if __name__ == "__main__":
    main()
