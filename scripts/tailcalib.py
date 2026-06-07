"""tailcalib.py — TailCalibX feature synthesis (decoupling stage 2) on a frozen backbone.

Vigneswaran et al. 2021 ("Feature generation for long-tail classification") / Yang et al.
2021 ("Free Lunch for Few-Shot: Distribution Calibration"). Post-backbone, no encoder
backprop — an N48-based long-tail variant parallel to cRT and tau-norm.

Pipeline (all on the FROZEN N48 backbone):
  1. Extract pooled graph features (h_graph) for the train + val splits.
  2. Per-class mean + covariance. "base" classes (>= base_min samples) have usable
     covariance; tail classes do not.
  3. For each class, borrow the covariance from its top-k nearest base classes (by mean
     distance), and sample synthetic features ~ N(real_seed, borrowed_cov) until the
     class reaches the target count (class balance via GENERATION, not resampling).
  4. Train a fresh linear classifier on the balanced real+synthetic features, selecting
     on real val macro-F1.
  5. Load the new head into the model and evaluate on test (full Evaluator), writing an
     ablation-compatible run dir.

Runs locally in minutes. Usage:
  uv run python scripts/tailcalib.py \
    --config configs/ablation/gnn_only/N57_a1_l1_tailcalib.yaml \
    --checkpoint checkpoints/20260606_163818_lmgat_codebert_multiclass/best_lmgat_codebert.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import f1_score

from gnn_vuln.config import Config
from gnn_vuln.evaluate import Evaluator
from gnn_vuln.models.registry import build_model, _parse_active_heads
from gnn_vuln.train import TrainingSession
from gnn_vuln.utils import get_device, load_checkpoint, save_checkpoint, setup_logging


def _find_classifier_linear(func_head: nn.Module) -> nn.Linear:
    last = None
    for m in func_head.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise ValueError("No nn.Linear found in func_head")
    return last


@torch.no_grad()
def _extract_features(model, loader, head, device):
    """Capture func_head input (pooled h_graph) + labels via a pre-hook."""
    feats, labels = [], []
    cap: dict = {}

    def _hook(_m, inp):
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


def _robust_cholesky(cov: torch.Tensor, rel_jitter: float = 0.01) -> torch.Tensor:
    """Cholesky of a (possibly rank-deficient) covariance with growing diagonal jitter.
    Falls back to a diagonal sqrt if it never becomes PD."""
    d = cov.shape[0]
    eye = torch.eye(d, dtype=cov.dtype)
    base = torch.diag(cov).mean().clamp(min=1e-6)
    jitter = rel_jitter * base
    for _ in range(8):
        try:
            return torch.linalg.cholesky(cov + jitter * eye)
        except Exception:
            jitter *= 10.0
    return torch.diag(torch.sqrt(torch.diag(cov).clamp(min=1e-6)))


def _class_stats(feats: torch.Tensor, labels: torch.Tensor, num_classes: int):
    means, covs, counts = {}, {}, {}
    for c in range(num_classes):
        fc = feats[labels == c]
        counts[c] = int(len(fc))
        if len(fc) >= 1:
            means[c] = fc.mean(0)
        if len(fc) >= 2:
            covs[c] = torch.cov(fc.t())          # [D, D]
    return means, covs, counts


def _synthesize(feats, labels, means, covs, counts, num_classes, target,
                topk, base_min, rel_jitter, rng):
    """Generate synthetic features for under-target classes using covariance borrowed
    from the nearest base classes. Returns (syn_feats [M,D], syn_labels [M])."""
    base = [c for c in range(num_classes) if counts.get(c, 0) >= base_min and c in covs]
    if not base:
        raise ValueError(f"No base classes with >= base_min={base_min} samples")
    base_means = torch.stack([means[c] for c in base])      # [B, D]
    logger.info(f"base classes ({len(base)}): {base}")

    syn_f, syn_y = [], []
    for c in range(num_classes):
        n_c = counts.get(c, 0)
        n_gen = target - n_c
        if n_gen <= 0 or n_c == 0:
            continue
        mu = means[c]
        dist = ((base_means - mu) ** 2).sum(1)
        order = torch.argsort(dist).tolist()
        sel = [base[i] for i in order if base[i] != c][:topk]
        cal_cov = torch.stack([covs[b] for b in sel]).mean(0)
        L = _robust_cholesky(cal_cov, rel_jitter)            # [D, D]
        fc = feats[labels == c]
        seed_idx = torch.as_tensor(rng.integers(0, n_c, size=n_gen), dtype=torch.long)
        seeds = fc[seed_idx]                                 # [n_gen, D]
        z = torch.randn(n_gen, mu.shape[0], generator=torch.Generator().manual_seed(int(rng.integers(0, 2**31))))
        syn = seeds + z @ L.t()
        syn_f.append(syn)
        syn_y.append(torch.full((n_gen,), c, dtype=torch.long))
    if not syn_f:
        return torch.empty(0, feats.shape[1]), torch.empty(0, dtype=torch.long)
    return torch.cat(syn_f), torch.cat(syn_y)


def _train_head(X, Y, val_X, val_Y, num_classes, device, epochs=40, bs=256, lr=1e-3, wd=1e-3):
    """Fresh linear classifier on the (balanced) feature set; select on val macro-F1."""
    D = X.shape[1]
    head = nn.Linear(D, num_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    X, Y = X.to(device), Y.to(device)
    val_X = val_X.to(device)
    vy = val_Y.numpy()
    best_f1, best_state = -1.0, None
    n = X.shape[0]
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(head(X[idx]), Y[idx])
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vp = head(val_X).argmax(1).cpu().numpy()
        f1 = f1_score(vy, vp, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, best_f1


def main() -> None:
    ap = argparse.ArgumentParser(description="TailCalibX feature synthesis (decoupling stage 2)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--topk", type=int, default=3, help="nearest base classes to borrow covariance from")
    ap.add_argument("--base-min", type=int, default=100, help="min train samples for a class to be a base (reliable cov)")
    ap.add_argument("--target", type=int, default=0, help="per-class target count (0 = max class count)")
    ap.add_argument("--rel-jitter", type=float, default=0.01, help="relative diagonal jitter for cholesky PD")
    ap.add_argument("--head-epochs", type=int, default=40)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    setup_logging(cfg.train.log_dir)
    device = get_device(cfg.train.device)
    rng = np.random.default_rng(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    sess = TrainingSession(cfg)
    sess.device = device
    dataset, _loaders, _train_idx = sess._setup_dataset()
    train_idx, val_idx, test_idx = dataset.get_splits(seed=cfg.train.seed)

    in_ch = dataset[0].x.size(1)
    model = build_model(cfg, in_ch, _parse_active_heads(cfg)).to(device)
    load_checkpoint(model, args.checkpoint, device=str(device))
    model.eval()
    from gnn_vuln.models.heads import StmtHead
    for m in model.modules():
        if isinstance(m, StmtHead):
            m._vectorized = True

    from torch_geometric.loader import DataLoader
    bs = cfg.train.batch_size
    train_loader = DataLoader(dataset[train_idx], batch_size=bs, shuffle=False)
    val_loader = DataLoader(dataset[val_idx], batch_size=bs, shuffle=False)

    logger.info("Extracting train + val features (frozen backbone)…")
    feats_tr, y_tr = _extract_features(model, train_loader, model.func_head, device)
    feats_val, y_val = _extract_features(model, val_loader, model.func_head, device)

    nc = cfg.model.num_classes
    means, covs, counts = _class_stats(feats_tr, y_tr, nc)
    target = args.target if args.target > 0 else max(counts.values())
    logger.info(f"per-class train counts: {[counts.get(c,0) for c in range(nc)]}")
    logger.info(f"target per class = {target} | topk={args.topk} | base_min={args.base_min}")

    syn_f, syn_y = _synthesize(feats_tr, y_tr, means, covs, counts, nc,
                               target, args.topk, args.base_min, args.rel_jitter, rng)
    logger.info(f"synthesized {syn_f.shape[0]} features | total train = {feats_tr.shape[0] + syn_f.shape[0]}")

    X = torch.cat([feats_tr, syn_f]) if syn_f.numel() else feats_tr
    Y = torch.cat([y_tr, syn_y]) if syn_f.numel() else y_tr
    head, best_val_f1 = _train_head(X, Y, feats_val, y_val, nc, device, epochs=args.head_epochs)
    logger.info(f"head trained — best val_f1_macro={best_val_f1:.4f}")

    # Copy the trained weights into the model's func_head Linear.
    head_linear = _find_classifier_linear(model.func_head)
    head_linear.weight.data.copy_(head.weight.data)
    head_linear.bias.data.copy_(head.bias.data)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg.model.architecture}_{cfg.data.mode}"
    ckpt_dir = cfg.train.checkpoint_dir / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"best_{cfg.model.architecture}.pt"
    save_checkpoint(model, best_path, val_f1=best_val_f1)
    shutil.copy(args.config, ckpt_dir / "config.yaml")

    num_params = sum(p.numel() for p in model.parameters())
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    with open(ckpt_dir / "training_summary.json", "w") as f:
        json.dump({
            "run_id": run_id,
            "architecture": cfg.model.architecture,
            "method": "tailcalib",
            "source_checkpoint": str(args.checkpoint),
            "topk": args.topk, "base_min": args.base_min, "target": target,
            "num_synthetic": int(syn_f.shape[0]),
            "num_classes": nc,
            "num_params": num_params,
            "epochs_trained": 0,
            "best_val_f1": round(best_val_f1, 6),
            "avg_epoch_time_s": 0, "total_time_s": 0,
            "peak_vram_gb": 0.0, "gpu": gpu,
        }, f, indent=2)

    res_dir = cfg.train.results_dir / run_id
    res_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(model=model, dataset=dataset, test_idx=test_idx, device=device,
                          results_dir=res_dir, batch_size=bs)
    evaluator.checkpoint_path = str(best_path)
    summary = evaluator.run()
    logger.info(
        f"tailcalib done. run_id={run_id}  "
        f"test_f1_macro={summary['function_level']['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()
