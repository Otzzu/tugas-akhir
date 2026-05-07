"""
train.py — Training entry point.

Usage (via uv):
    uv run train --config configs/lmgcn/binary.yaml
    uv run train --config configs/lmgat/binary.yaml
    uv run train --config configs/lmgat_codebert/multiclass.yaml
    uv run train --config configs/lmgat_mcs/multiclass.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch_geometric.loader import DataLoader
from loguru import logger

from gnn_vuln.config import Config, load_default_config
from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset
from gnn_vuln.losses import HierarchicalSupConLoss
from gnn_vuln.models.registry import build_model, _parse_active_heads
from gnn_vuln.training.ewc import EWCDR
from gnn_vuln.training.losses import livable_weights
from gnn_vuln.training.optimizer import build_optimizer_and_scheduler
from gnn_vuln.training.trainer import Trainer
from gnn_vuln.utils import (
    set_seed, setup_logging, get_device,
    save_checkpoint, load_resume_checkpoint, save_resume_checkpoint,
)

# Re-export for backward compatibility (evaluate.py, inference.py import from here)
__all__ = ["build_model", "_parse_active_heads"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train GNN vulnerability detector")
    parser.add_argument("--config", type=str, default="configs/lmgcn/binary.yaml")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest last_*.pt checkpoint for this arch/mode.",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else load_default_config()
    set_seed(cfg.train.seed)
    setup_logging(cfg.train.log_dir)
    device = get_device(cfg.train.device)

    # ── Loss / training knobs ─────────────────────────────────────────────────
    active_heads       = _parse_active_heads(cfg)
    mil_k              = getattr(cfg.model, "mil_k", 3)
    mil_weight         = getattr(cfg.model, "mil_weight", 0.5)
    rank_loss_weight   = getattr(cfg.model, "rank_loss_weight", 0.0)
    group_loss_weight  = getattr(cfg.model, "group_loss_weight", 0.0)
    binary_loss_weight = getattr(cfg.model, "binary_loss_weight", 0.0)
    focal_gamma        = getattr(cfg.train, "focal_loss_gamma", 0.0)
    grad_clip          = getattr(cfg.train, "grad_clip", 0.0)
    is_binary          = getattr(cfg.data, "mode", "binary") == "binary"
    use_class_weights  = getattr(cfg.train, "use_class_weights", True)
    use_livable        = getattr(cfg.train, "livable_loss", False) and use_class_weights

    if active_heads:
        if "group"  not in active_heads: group_loss_weight  = 0.0
        if "binary" not in active_heads: binary_loss_weight = 0.0
        logger.info(f"MTL active_heads: {sorted(active_heads)}")

    # ── SupCon loss (optional) ────────────────────────────────────────────────
    use_supcon    = getattr(cfg.model, "use_supcon", False)
    supcon_weight = getattr(cfg.model, "supcon_weight", 0.1) if use_supcon else 0.0
    supcon_fn     = None
    if use_supcon:
        _cwe_vocab_path = (
            Path(getattr(cfg.data, "processed_dir", Path("data/processed"))).parent
            / "raw" / getattr(cfg.data, "source", "megavul") / "cwe_vocab.json"
        )
        _cwe_vocab:       dict[str, int] | None = None
        _dist_matrix_path: Path | None           = None
        if getattr(cfg.model, "supcon_use_distance_matrix", False):
            if _cwe_vocab_path.exists():
                with open(_cwe_vocab_path, encoding="utf-8") as _f:
                    _cwe_vocab = json.load(_f)
            _p = Path(getattr(cfg.model, "cwe_dist_matrix", "data/cwe/cwe_distance_matrix.json"))
            _dist_matrix_path = _p if _p.exists() else None
            if _dist_matrix_path is None:
                logger.warning(f"supcon_use_distance_matrix=true but matrix not found at {_p}; using alpha fallback.")
        supcon_fn = HierarchicalSupConLoss(
            temperature    = getattr(cfg.model, "supcon_temperature",    0.07),
            alpha          = getattr(cfg.model, "supcon_alpha",          0.5),
            dist_matrix_path = _dist_matrix_path,
            cwe_vocab        = _cwe_vocab,
            weight_fn      = getattr(cfg.model, "supcon_weight_fn",      "linear"),
            exp_scale      = getattr(cfg.model, "supcon_exp_scale",      5.0),
            power          = getattr(cfg.model, "supcon_power",          2.0),
            min_weight     = getattr(cfg.model, "supcon_min_weight",     0.0),
            intragroup_only= getattr(cfg.model, "supcon_intragroup_only", True),
        )

    # ── Dataset ───────────────────────────────────────────────────────────────
    pretrained_lm    = getattr(cfg.model, "pretrained_lm", "microsoft/codebert-base")
    func_lm          = getattr(cfg.model, "func_lm", "") or pretrained_lm
    add_func_tokens  = getattr(cfg.model, "add_func_tokens", False)
    func_lm_source   = getattr(cfg.model, "func_lm_source", "raw")
    source_val       = getattr(cfg.data, "source_val",  "")
    source_test      = getattr(cfg.data, "source_test", "")
    use_official     = bool(source_val and source_test)

    logger.info(
        f"Loading dataset… (pretrained_lm={pretrained_lm}, add_func_tokens={add_func_tokens}, "
        f"splits={'official' if use_official else 'internal 70/15/15'})"
    )

    dataset_kwargs = dict(
        root             = str(cfg.data.processed_dir.parent),
        max_nodes        = cfg.data.max_nodes,
        embedder_device  = str(device),
        mode             = cfg.data.mode,
        pretrained_lm    = pretrained_lm,
        func_lm          = func_lm,
        add_func_tokens  = add_func_tokens,
        func_lm_source   = func_lm_source,
        top_cwe          = getattr(cfg.data, "top_cwe", 0),
        cwe_list         = getattr(cfg.data, "cwe_list", None),
        cwe_groups       = getattr(cfg.data, "cwe_groups", None),
        filter_owasp_top10 = getattr(cfg.data, "filter_owasp_top10", False),
        filter_top25     = getattr(cfg.data, "filter_top25", False),
        max_per_class    = getattr(cfg.data, "max_per_class", 0),
        resample_seed    = getattr(cfg.data, "resample_seed", 42),
    )
    dataset = CodeBERTGraphDataset(source=getattr(cfg.data, "source", "bigvul"), **dataset_kwargs)

    if use_official:
        logger.info(f"Official splits: train={cfg.data.source}  val={source_val}  test={source_test}")
        val_dataset  = CodeBERTGraphDataset(source=source_val,  **dataset_kwargs)
        test_dataset = CodeBERTGraphDataset(source=source_test, **dataset_kwargs)
        train_idx    = list(range(len(dataset)))
        train_loader = DataLoader(dataset,      batch_size=cfg.train.batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,  batch_size=cfg.train.batch_size)
        test_loader  = DataLoader(test_dataset, batch_size=cfg.train.batch_size)
    else:
        train_idx, val_idx, test_idx = dataset.get_splits(seed=cfg.train.seed)
        train_loader = DataLoader(dataset[train_idx], batch_size=cfg.train.batch_size, shuffle=True)
        val_loader   = DataLoader(dataset[val_idx],   batch_size=cfg.train.batch_size)
        test_loader  = DataLoader(dataset[test_idx],  batch_size=cfg.train.batch_size)

    in_channels = dataset[0].x.size(1)
    logger.info(f"Dataset: {len(dataset)} graphs | in_channels={in_channels}")

    if dataset.num_classes != cfg.model.num_classes:
        raise ValueError(
            f"Config model.num_classes={cfg.model.num_classes} but dataset has "
            f"{dataset.num_classes} classes. Update config or check cwe_vocab.json."
        )

    # ── Class weights ─────────────────────────────────────────────────────────
    train_counts: torch.Tensor | None = None
    class_weight: torch.Tensor | None = None
    if use_class_weights:
        train_labels = torch.tensor(
            [int(dataset[i].y.item()) for i in train_idx], dtype=torch.long
        )
        train_counts = torch.bincount(train_labels, minlength=cfg.model.num_classes).float()
        if use_livable:
            class_weight = livable_weights(train_counts, 1, cfg.train.epochs, cfg.model.num_classes, device)
            logger.info(f"LIVABLE weights (epoch 1): {[f'{w:.3f}' for w in class_weight.tolist()]}")
        else:
            class_weight = torch.clamp(
                train_counts.sum() / (train_counts * cfg.model.num_classes), max=10.0
            ).to(device)
            logger.info(f"Static class weights: {[f'{w:.3f}' for w in class_weight.tolist()]}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg, in_channels, active_heads).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    loss_mode = "livable" if use_livable else ("focal" if focal_gamma > 0 else "ce")
    logger.info(
        f"Model: {cfg.model.architecture.upper()} | params={n_params:,} | "
        f"mil_weight={mil_weight} | mil_k={mil_k} | rank={rank_loss_weight} | "
        f"class_weights={use_class_weights} | loss={loss_mode}"
    )

    # ── EWC-DR continual learning (optional) ──────────────────────────────────
    ewc = None
    _ewc_cfg = getattr(cfg, "ewc", None)
    if _ewc_cfg is not None and getattr(_ewc_cfg, "enabled", False):
        _ewc_weight  = getattr(_ewc_cfg, "weight",     1000.0)
        _ewc_scope   = getattr(_ewc_cfg, "scope",      "all")
        _ewc_cache   = getattr(_ewc_cfg, "importance_cache", "")
        _ewc_ckpt    = getattr(_ewc_cfg, "source_checkpoint", "")
        _ewc_nbatch  = getattr(_ewc_cfg, "n_batches",  0)

        if _ewc_cache and Path(_ewc_cache).exists():
            logger.info(f"EWC-DR: loading cached importance from {_ewc_cache}")
            ewc = EWCDR.from_file(_ewc_cache, ewc_weight=_ewc_weight)
        else:
            if not _ewc_ckpt or not Path(_ewc_ckpt).exists():
                raise ValueError(
                    f"EWC enabled but source_checkpoint not found: {_ewc_ckpt!r}. "
                    "Provide a valid path to the task-A best checkpoint."
                )
            logger.info(f"EWC-DR: loading task-A model from {_ewc_ckpt}")
            from gnn_vuln.utils import load_checkpoint as _load_ckpt
            _load_ckpt(_ewc_ckpt, model, device=str(device))

            logger.info(
                f"EWC-DR: computing importance (scope={_ewc_scope}, "
                f"n_batches={'all' if _ewc_nbatch == 0 else _ewc_nbatch}) …"
            )
            ewc = EWCDR(
                model=model,
                dataloader=train_loader,
                device=device,
                ewc_weight=_ewc_weight,
                scope=_ewc_scope,
                n_batches=_ewc_nbatch,
            )

            if _ewc_cache:
                Path(_ewc_cache).parent.mkdir(parents=True, exist_ok=True)
                ewc.save(_ewc_cache)

            # Reload task-B model weights (EWC computation overwrites with task-A weights)
            logger.info("EWC-DR: reloading task-B model initialisation …")
            model = build_model(cfg, in_channels, active_heads).to(device)
            ewc._star  = {k: v.cpu() for k, v in ewc._star.items()}  # already CPU

    # ── AMP ───────────────────────────────────────────────────────────────────
    use_amp   = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler    = GradScaler() if use_amp and amp_dtype == torch.float16 else None
    if use_amp:
        logger.info(f"AMP enabled — dtype={amp_dtype}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    total_steps = len(train_loader) * cfg.train.epochs
    optimizer, scheduler, step_per_batch = build_optimizer_and_scheduler(model, cfg, total_steps)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step_per_batch=step_per_batch,
        device=device,
        mil_k=mil_k,
        mil_weight=mil_weight,
        rank_loss_weight=rank_loss_weight,
        focal_gamma=focal_gamma,
        group_loss_weight=group_loss_weight,
        binary_loss_weight=binary_loss_weight,
        supcon_fn=supcon_fn,
        supcon_weight=supcon_weight,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        scaler=scaler,
        ewc=ewc,
    )
    trainer.set_grad_clip(grad_clip)

    # ── Checkpoint setup ──────────────────────────────────────────────────────
    stop_on_f1      = getattr(cfg.train, "early_stop_metric", "f1") == "f1"
    best_val_f1     = -1.0
    best_val_loss   = float("inf")
    patience_counter = 0
    start_epoch     = 1

    run_id  = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg.model.architecture}_{cfg.data.mode}"
    run_dir = cfg.train.checkpoint_dir / run_id

    if args.resume:
        existing = sorted(cfg.train.checkpoint_dir.glob(
            f"*_{cfg.model.architecture}_{cfg.data.mode}/last_{cfg.model.architecture}.pt"
        ))
        if existing:
            last_ckpt_path = existing[-1]
            run_dir  = last_ckpt_path.parent
            run_id   = run_dir.name
            logger.info(f"Resuming run: {run_id}")
        else:
            logger.warning("--resume set but no previous run found — starting fresh.")
            args.resume = False

    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = run_dir / f"best_{cfg.model.architecture}.pt"
    last_ckpt = run_dir / f"last_{cfg.model.architecture}.pt"

    if Path(args.config).exists():
        shutil.copy(args.config, run_dir / "config.yaml")

    logger.info(f"Run ID: {run_id}  |  checkpoints → {run_dir}")

    if args.resume and last_ckpt.exists():
        meta = load_resume_checkpoint(last_ckpt, model, optimizer, scheduler, device=str(device))
        start_epoch      = meta["epoch"] + 1
        best_val_f1      = meta.get("best_val_f1",   -1.0)
        best_val_loss    = meta.get("best_val_loss",  float("inf"))
        patience_counter = meta["patience_counter"]
        logger.info(f"Resuming from epoch {start_epoch} (patience={patience_counter})")
    elif args.resume:
        logger.warning(f"--resume: {last_ckpt} not found — starting from scratch.")

    # ── Training loop ─────────────────────────────────────────────────────────
    train_start    = time.time()
    save_last_every = getattr(cfg.train, "save_last_every", 1)

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        if use_livable and train_counts is not None:
            class_weight = livable_weights(train_counts, epoch, cfg.train.epochs, cfg.model.num_classes, device)

        t0         = time.time()
        train_loss = trainer.train_epoch(train_loader, epoch, cfg.train.epochs, class_weight)
        val_loss, val_acc, val_conf, val_f1, val_f1w = trainer.evaluate(val_loader, is_binary, class_weight)

        if not step_per_batch:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[-1]["lr"]
        improved   = (val_f1 > best_val_f1) if stop_on_f1 else (val_loss < best_val_loss)

        logger.info(
            f"Epoch {epoch:03d}/{cfg.train.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"acc={val_acc:.4f} | f1={val_f1:.4f} | f1w={val_f1w:.4f} | conf={val_conf:.4f} | "
            f"lr={current_lr:.2e} | patience={patience_counter}/{cfg.train.patience} | "
            f"{time.time()-t0:.0f}s" + (" *" if improved else "")
        )

        if improved:
            best_val_f1 = val_f1; best_val_loss = val_loss; patience_counter = 0
            save_checkpoint(model, best_ckpt, epoch=epoch, val_loss=val_loss,
                            val_acc=val_acc, val_conf=val_conf, val_f1=val_f1, val_f1_weighted=val_f1w)
        else:
            patience_counter += 1

        if save_last_every > 0 and epoch % save_last_every == 0:
            save_resume_checkpoint(last_ckpt, model, optimizer, scheduler, epoch=epoch,
                                   best_val_f1=best_val_f1, best_val_loss=best_val_loss,
                                   patience_counter=patience_counter, val_loss=val_loss,
                                   val_acc=val_acc, val_conf=val_conf, val_f1=val_f1, val_f1_weighted=val_f1w)

        if patience_counter >= cfg.train.patience:
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    # ── Test evaluation ───────────────────────────────────────────────────────
    _, test_acc, test_conf, test_f1, test_f1w = trainer.evaluate(test_loader, is_binary, class_weight)
    logger.info(f"Test  acc={test_acc:.4f} | f1={test_f1:.4f} | f1w={test_f1w:.4f} | conf={test_conf:.4f}")

    total_secs = time.time() - train_start
    h, rem = divmod(int(total_secs), 3600); m, s = divmod(rem, 60)
    logger.info(f"Total time: {h:02d}h {m:02d}m {s:02d}s")
    logger.info(f"Best checkpoint → {best_ckpt}")
    logger.info(f"To evaluate:  uv run evaluate --checkpoint {best_ckpt} --config {run_dir/'config.yaml'}")


if __name__ == "__main__":
    main()
