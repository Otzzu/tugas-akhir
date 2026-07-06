"""tau_norm.py — tau-normalized classifier (Kang et al. 2020, ICLR), decoupling stage 2.

Post-hoc, ZERO-TRAINING long-tail fix. Loads a trained backbone + its jointly-trained
classifier, rescales the classifier weight norms

    w_i_tilde = w_i / ||w_i||^tau        (bias discarded — paper, negligible effect)

sweeps tau on the validation set for best macro-F1, applies the best tau, and evaluates
on test. Decoupling alternative to cRT: no re-training at all, just a weight-norm
rebalance + a one-dim tau search. Paper shows it often matches or beats cRT.

Runs locally in minutes (inference over val + test once on the frozen backbone). Produces
an ablation-compatible run dir (best_*.pt + config.yaml + training_summary.json in
checkpoints/<run_id>/, metrics_summary.json + plots in results/<run_id>/), so
/ablation-metrics works on it directly.

Usage:
  uv run python scripts/tau_norm.py \
    --config configs/ablation/gnn_only/N56_a1_l1_tau_norm.yaml \
    --checkpoint checkpoints/20260606_163818_lmgat_codebert_multiclass/best_lmgat_codebert.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime

import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import f1_score

from gnn_vuln.config import Config
from gnn_vuln.evaluate import Evaluator
from gnn_vuln.models.registry import build_model, _parse_active_heads
from gnn_vuln.train import TrainingSession
from gnn_vuln.utils import get_device, load_checkpoint, save_checkpoint, setup_logging


def _find_classifier_linear(func_head: nn.Module) -> nn.Linear:
    """Return the final nn.Linear inside func_head (the classifier W, b)."""
    last = None
    for m in func_head.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise ValueError("No nn.Linear found in func_head — tau-norm needs a linear classifier")
    return last


@torch.no_grad()
def _extract_features(model, loader, head, device):
    """Run inference, capturing func_head input (the pooled graph feature h_graph)
    via a forward pre-hook, plus labels. Backbone is frozen so these are fixed
    across the whole tau sweep — extract once."""
    feats, labels = [], []
    cap: dict = {}

    def _hook(_mod, inp):
        cap["x"] = inp[0].detach()

    handle = head.register_forward_pre_hook(_hook)
    try:
        for batch in loader:
            batch = batch.to(device)
            node_line = getattr(batch, "node_line", None)
            edge_attr = getattr(batch, "edge_attr", None)
            model(batch.x, batch.edge_index, batch.batch, node_line, edge_attr)
            feats.append(cap["x"].float().cpu())
            labels.append(batch.y.cpu())
    finally:
        handle.remove()
    return torch.cat(feats), torch.cat(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description="tau-normalized classifier (decoupling stage 2)")
    ap.add_argument("--config", required=True, help="Config (same model/data as the backbone)")
    ap.add_argument("--checkpoint", required=True, help="Trained backbone best_*.pt")
    ap.add_argument("--tau-grid", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                    help="Comma-separated tau values to sweep on val")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    setup_logging(cfg.train.log_dir)
    device = get_device(cfg.train.device)

    # Dataset + loaders via TrainingSession for identical splits / collate.
    sess = TrainingSession(cfg)
    sess.device = device
    dataset, loaders, _train_idx = sess._setup_dataset()
    _, val_loader, _test_loader = loaders
    _, _val_idx, test_idx = dataset.get_splits(seed=cfg.train.seed)

    in_ch = dataset[0].x.size(1)
    model = build_model(cfg, in_ch, _parse_active_heads(cfg)).to(device)
    load_checkpoint(model, args.checkpoint, device=str(device))
    model.eval()
    from gnn_vuln.models.heads import StmtHead
    for m in model.modules():
        if isinstance(m, StmtHead):
            m._vectorized = True

    head_linear = _find_classifier_linear(model.func_head)
    W = head_linear.weight.data.clone()          # [C, D]
    has_bias = head_linear.bias is not None
    wnorm = W.norm(dim=1, keepdim=True)          # [C, 1]

    logger.info("Caching val features for tau-sweep (backbone frozen)…")
    feats_val, y_val = _extract_features(model, val_loader, model.func_head, device)
    feats_val = feats_val.to(device)
    yv = y_val.numpy()

    grid = [float(t) for t in args.tau_grid.split(",")]
    best_tau, best_f1 = 0.0, -1.0
    for tau in grid:
        W_t = W / (wnorm.pow(tau) + 1e-12)       # tau-norm (bias discarded)
        logits = feats_val @ W_t.t()             # [N, C]
        pred = logits.argmax(1).cpu().numpy()
        f1 = f1_score(yv, pred, average="macro", zero_division=0)
        logger.info(f"  tau={tau:.2f}  val_f1_macro={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    logger.info(f"Best tau={best_tau:.2f}  val_f1_macro={best_f1:.4f}")

    # Apply best tau to the live classifier; discard bias (paper).
    W_best = W / (wnorm.pow(best_tau) + 1e-12)
    head_linear.weight.data.copy_(W_best)
    if has_bias:
        head_linear.bias.data.zero_()

    # Write an ablation-compatible run dir.
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg.model.architecture}_{cfg.data.mode}"
    ckpt_dir = cfg.train.checkpoint_dir / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"best_{cfg.model.architecture}.pt"
    save_checkpoint(model, best_path, tau=best_tau, val_f1=best_f1)
    shutil.copy(args.config, ckpt_dir / "config.yaml")

    num_params = sum(p.numel() for p in model.parameters())
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    with open(ckpt_dir / "training_summary.json", "w") as f:
        json.dump({
            "run_id": run_id,
            "architecture": cfg.model.architecture,
            "method": "tau_norm",
            "source_checkpoint": str(args.checkpoint),
            "best_tau": best_tau,
            "num_classes": cfg.model.num_classes,
            "num_params": num_params,
            "epochs_trained": 0,
            "best_val_f1": round(best_f1, 6),
            "avg_epoch_time_s": 0,
            "total_time_s": 0,
            "peak_vram_gb": 0.0,
            "gpu": gpu,
        }, f, indent=2)

    res_dir = cfg.train.results_dir / run_id
    res_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(
        model=model, dataset=dataset, test_idx=test_idx, device=device,
        results_dir=res_dir, batch_size=cfg.train.batch_size,
    )
    evaluator.checkpoint_path = str(best_path)
    summary = evaluator.run()
    logger.info(
        f"tau-norm done. run_id={run_id}  best_tau={best_tau}  "
        f"test_f1_macro={summary['function_level']['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()
