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
from gnn_vuln.training.losses import epoch_adaptive_class_weights, livable_loss as livable_real_loss
from gnn_vuln.training.optimizer import build_optimizer_and_scheduler
from gnn_vuln.training.trainer import Trainer
from gnn_vuln.utils import (
    set_seed, setup_logging, get_device,
    save_checkpoint, load_resume_checkpoint, save_resume_checkpoint,
    CheckpointManager,
)

__all__ = ["build_model", "_parse_active_heads", "TrainingSession"]


# ---------------------------------------------------------------------------
# TrainingSession — encapsulates full training setup + loop
# ---------------------------------------------------------------------------

class TrainingSession:
    """
    Encapsulates a full training run: dataset, model, optimizer, loop, checkpointing.

    Usage
    -----
        session = TrainingSession.from_args(args)
        session.run()
    """

    def __init__(self, cfg: "Config", resume: bool = False) -> None:
        self.cfg    = cfg
        self.resume = resume
        self.device = get_device(cfg.train.device)

        self._active_heads       = _parse_active_heads(cfg)
        self._mil_k              = getattr(cfg.model, "mil_k", 3)
        self._mil_weight         = getattr(cfg.model, "mil_weight", 0.5)
        self._rank_loss_weight   = getattr(cfg.model, "rank_loss_weight", 0.0)
        self._group_loss_weight  = getattr(cfg.model, "group_loss_weight", 0.0)
        self._binary_loss_weight = getattr(cfg.model, "binary_loss_weight", 0.0)
        self._focal_gamma        = getattr(cfg.train, "focal_loss_gamma", 0.0)
        self._grad_clip          = getattr(cfg.train, "grad_clip", 0.0)
        self._is_binary          = getattr(cfg.data, "mode", "binary") == "binary"
        self._use_class_weights  = getattr(cfg.train, "use_class_weights", True)
        # Support old name (livable_loss/livable_adaptive) for backward compat with existing configs
        self._use_epoch_adaptive = (
            getattr(cfg.train, "epoch_adaptive_weights", False) or
            getattr(cfg.train, "livable_adaptive", False) or
            getattr(cfg.train, "livable_loss_old", False)  # not used, just safety
        ) and self._use_class_weights
        self._use_livable_real   = getattr(cfg.train, "livable_loss", False)

        if self._active_heads:
            if "group"  not in self._active_heads: self._group_loss_weight  = 0.0
            if "binary" not in self._active_heads: self._binary_loss_weight = 0.0

        self._use_supcon    = getattr(cfg.model, "use_supcon", False)
        self._supcon_weight = getattr(cfg.model, "supcon_weight", 0.1) if self._use_supcon else 0.0

    @classmethod
    def from_args(cls, args) -> "TrainingSession":
        cfg = Config.from_yaml(args.config) if Path(args.config).exists() else load_default_config()
        set_seed(cfg.train.seed, deterministic=getattr(cfg.train, "deterministic", False))
        setup_logging(cfg.train.log_dir)
        return cls(cfg, resume=getattr(args, "resume", False))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, config_path: str | Path | None = None) -> None:
        cfg    = self.cfg
        device = self.device

        supcon_fn     = self._build_supcon()
        dataset, loaders, train_idx = self._setup_dataset()
        train_loader, val_loader, test_loader = loaders

        in_channels = dataset[0].x.size(1)
        logger.info(f"Dataset: {len(dataset)} graphs | in_channels={in_channels}")
        if dataset.num_classes != cfg.model.num_classes:
            raise ValueError(
                f"Config model.num_classes={cfg.model.num_classes} but dataset has "
                f"{dataset.num_classes} classes."
            )

        class_weight, train_counts = self._setup_class_weights(dataset, train_idx)
        if self._use_livable_real and class_weight is not None:
            logger.info("livable_loss=true → disabling static class_weight")
            class_weight = None

        model = build_model(cfg, in_channels, self._active_heads).to(device)

        # Vectorized StmtHead: scatter-based pooling, no Python inner loop
        if getattr(cfg.train, "stmt_head_vectorized", False):
            from gnn_vuln.models.heads import StmtHead
            for m in model.modules():
                if isinstance(m, StmtHead):
                    m._vectorized = True
            logger.info("StmtHead: vectorized scatter mode enabled")

        ewc   = self._setup_ewc(model, train_loader, in_channels)

        # torch.compile — fuses kernels, ~20-50% speedup (PyTorch 2.0+, CUDA only)
        if getattr(cfg.train, "compile_model", False) and device.type == "cuda":
            try:
                import torch._dynamo
                torch._dynamo.config.capture_scalar_outputs = True
                model = torch.compile(model, mode="default", dynamic=True)
                logger.info("torch.compile enabled (mode=reduce-overhead)")
            except Exception as e:
                logger.warning(f"torch.compile failed, skipping: {e}")

        use_amp, amp_dtype, scaler = self._setup_amp()
        total_steps = len(train_loader) * cfg.train.epochs
        optimizer, scheduler, step_per_batch = build_optimizer_and_scheduler(model, cfg, total_steps)

        grad_accum_steps = getattr(cfg.train, "grad_accum_steps", 1)
        trainer = Trainer(
            model=model, optimizer=optimizer, scheduler=scheduler,
            step_per_batch=step_per_batch, device=device,
            mil_k=self._mil_k, mil_weight=self._mil_weight,
            rank_loss_weight=self._rank_loss_weight, focal_gamma=self._focal_gamma,
            group_loss_weight=self._group_loss_weight, binary_loss_weight=self._binary_loss_weight,
            supcon_fn=supcon_fn, supcon_weight=self._supcon_weight,
            use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler, ewc=ewc,
            grad_accum_steps=grad_accum_steps,
            label_smoothing=getattr(cfg.train, "label_smoothing", 0.0),
            use_livable_real=self._use_livable_real,
            livable_focal_gamma=getattr(cfg.train, "focal_loss_gamma", 2.0),
            livable_label_smoothing=getattr(cfg.train, "label_smoothing", 0.1),
        )
        trainer.set_grad_clip(self._grad_clip)

        run_id, run_dir = self._setup_run_dir()
        if config_path and Path(config_path).exists():
            shutil.copy(config_path, run_dir / "config.yaml")

        cm = CheckpointManager(run_dir, cfg.model.architecture)
        stop_on_f1 = getattr(cfg.train, "early_stop_metric", "f1") == "f1"
        best_val_f1 = -1.0; best_val_loss = float("inf"); patience_counter = 0; start_epoch = 1

        if self.resume and cm.has_resume():
            meta = cm.load_resume(model, optimizer, scheduler, device=str(device))
            start_epoch      = meta["epoch"] + 1
            best_val_f1      = meta.get("best_val_f1",  -1.0)
            best_val_loss    = meta.get("best_val_loss", float("inf"))
            patience_counter = meta["patience_counter"]
            logger.info(f"Resuming from epoch {start_epoch}")
        elif self.resume:
            logger.warning(f"--resume: {cm.last_path} not found — starting fresh.")

        self._training_loop(
            trainer, train_loader, val_loader, test_loader,
            cm, class_weight, train_counts, stop_on_f1,
            best_val_f1, best_val_loss, patience_counter, start_epoch,
            step_per_batch, optimizer, scheduler,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_supcon(self):
        if not self._use_supcon:
            return None
        cfg = self.cfg
        _cwe_vocab_path = (
            Path(getattr(cfg.data, "processed_dir", Path("data/processed"))).parent
            / "raw" / getattr(cfg.data, "source", "megavul") / "cwe_vocab.json"
        )
        _cwe_vocab = None
        _dist_matrix_path = None
        if getattr(cfg.model, "supcon_use_distance_matrix", False):
            if _cwe_vocab_path.exists():
                with open(_cwe_vocab_path, encoding="utf-8") as f:
                    _cwe_vocab = json.load(f)
            _p = Path(getattr(cfg.model, "cwe_dist_matrix", "data/cwe/cwe_distance_matrix.json"))
            _dist_matrix_path = _p if _p.exists() else None
        fn = HierarchicalSupConLoss(
            temperature=getattr(cfg.model, "supcon_temperature", 0.07),
            alpha=getattr(cfg.model, "supcon_alpha", 0.5),
            dist_matrix_path=_dist_matrix_path, cwe_vocab=_cwe_vocab,
            weight_fn=getattr(cfg.model, "supcon_weight_fn", "linear"),
            exp_scale=getattr(cfg.model, "supcon_exp_scale", 5.0),
            power=getattr(cfg.model, "supcon_power", 2.0),
            min_weight=getattr(cfg.model, "supcon_min_weight", 0.0),
            intragroup_only=getattr(cfg.model, "supcon_intragroup_only", True),
        )
        return fn.to(self.device)

    def _setup_dataset(self):
        cfg = self.cfg
        pretrained_lm   = getattr(cfg.model, "pretrained_lm", "microsoft/codebert-base")
        func_lm         = getattr(cfg.model, "func_lm", "") or pretrained_lm
        add_func_tokens = getattr(cfg.model, "add_func_tokens", False)
        func_lm_source  = getattr(cfg.model, "func_lm_source", "raw")
        source_val      = getattr(cfg.data, "source_val",  "")
        source_test     = getattr(cfg.data, "source_test", "")
        use_official    = bool(source_val and source_test)

        kwargs = dict(
            root=str(cfg.data.processed_dir.parent), max_nodes=cfg.data.max_nodes,
            embedder_device=str(self.device), mode=cfg.data.mode,
            pretrained_lm=pretrained_lm, func_lm=func_lm,
            add_func_tokens=add_func_tokens, func_lm_source=func_lm_source,
            top_cwe=getattr(cfg.data, "top_cwe", 0),
            cwe_list=getattr(cfg.data, "cwe_list", None),
            cwe_groups=getattr(cfg.data, "cwe_groups", None),
            filter_owasp=getattr(cfg.data, "filter_owasp", False),
            filter_top25_dangerous=getattr(cfg.data, "filter_top25_dangerous", False),
            max_per_class=getattr(cfg.data, "max_per_class", 0),
            resample_seed=getattr(cfg.data, "resample_seed", 42),
            func_max_length=getattr(cfg.model, "func_max_length", 512),
            storage=getattr(cfg.data, "storage", "inmemory"),
        )
        bs          = cfg.train.batch_size
        num_workers = getattr(cfg.train, "num_workers",    4)
        prefetch    = getattr(cfg.train, "prefetch_factor", 2)
        pin_mem     = self.device.type == "cuda"

        # Strip heavy func-token tensors from collation when there's no live LM.
        # lmgat_codebert with live_lm=none uses no func_input_ids but the _ft
        # dataset still carries them — 64×1024 token stacks per batch for nothing.
        _needs_func_tokens = getattr(cfg.model, "live_lm", "func") != "none"
        _FUNC_TOKEN_KEYS = ("func_input_ids", "func_attention_mask", "func_token_lines")
        def _strip_collate_fn(batch):
            from torch_geometric.data import Batch
            if not _needs_func_tokens:
                for g in batch:
                    for k in _FUNC_TOKEN_KEYS:
                        if hasattr(g, k):
                            delattr(g, k)
            return Batch.from_data_list(batch)

        _seed = cfg.train.seed

        def _worker_init_fn(worker_id):
            # torch already sets initial_seed() = base_seed + worker_id per worker
            worker_seed = torch.initial_seed() % (2 ** 32)
            import random as _random
            import numpy as _np
            _random.seed(worker_seed)
            _np.random.seed(worker_seed)

        _g = torch.Generator().manual_seed(_seed)

        dl_kw = dict(
            num_workers=num_workers,
            pin_memory=pin_mem,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch if num_workers > 0 else None,
            collate_fn=_strip_collate_fn,
            worker_init_fn=_worker_init_fn,
            generator=_g,
        )

        dataset = CodeBERTGraphDataset(source=getattr(cfg.data, "source", "bigvul"), **kwargs)
        _dataset_pt = Path(dataset.processed_paths[0]).name
        self._dataset_pt = _dataset_pt
        if use_official:
            val_ds  = CodeBERTGraphDataset(source=source_val,  **kwargs)
            test_ds = CodeBERTGraphDataset(source=source_test, **kwargs)
            train_idx = list(range(len(dataset)))
            loaders = (
                DataLoader(dataset, batch_size=bs, shuffle=True, **dl_kw),
                DataLoader(val_ds,  batch_size=bs, **dl_kw),
                DataLoader(test_ds, batch_size=bs, **dl_kw),
            )
        else:
            train_idx, val_idx, test_idx = dataset.get_splits(seed=cfg.train.seed)
            loaders = (
                DataLoader(dataset[train_idx], batch_size=bs, shuffle=True, **dl_kw),
                DataLoader(dataset[val_idx],   batch_size=bs, **dl_kw),
                DataLoader(dataset[test_idx],  batch_size=bs, **dl_kw),
            )
        return dataset, loaders, train_idx

    def _setup_class_weights(self, dataset, train_idx):
        if not self._use_class_weights:
            return None, None
        cfg = self.cfg
        all_y = dataset.get_all_labels()
        train_labels = all_y[torch.tensor(train_idx, dtype=torch.long)]
        counts = torch.bincount(train_labels, minlength=cfg.model.num_classes).float()
        if self._use_epoch_adaptive:
            w = epoch_adaptive_class_weights(counts, 1, cfg.train.epochs, cfg.model.num_classes, self.device)
        else:
            w = torch.clamp(counts.sum() / (counts * cfg.model.num_classes), max=10.0).to(self.device)
        return w, counts

    def _setup_ewc(self, model, train_loader, in_channels):
        cfg = self.cfg
        _ewc_cfg = getattr(cfg, "ewc", None)
        if _ewc_cfg is None or not getattr(_ewc_cfg, "enabled", False):
            return None
        _ewc_weight  = getattr(_ewc_cfg, "weight",             1000.0)
        _ewc_scope   = getattr(_ewc_cfg, "scope",              "all")
        _ewc_cache   = getattr(_ewc_cfg, "importance_cache",   "")
        _ewc_ckpt    = getattr(_ewc_cfg, "source_checkpoint",  "")
        _ewc_nbatch  = getattr(_ewc_cfg, "n_batches",          0)
        if _ewc_cache and Path(_ewc_cache).exists():
            return EWCDR.from_file(_ewc_cache, ewc_weight=_ewc_weight)
        if not _ewc_ckpt or not Path(_ewc_ckpt).exists():
            raise ValueError(f"EWC enabled but source_checkpoint not found: {_ewc_ckpt!r}")
        from gnn_vuln.utils import load_checkpoint as _lc
        _lc(_ewc_ckpt, model, device=str(self.device))
        ewc = EWCDR(model=model, dataloader=train_loader, device=self.device,
                    ewc_weight=_ewc_weight, scope=_ewc_scope, n_batches=_ewc_nbatch)
        if _ewc_cache:
            Path(_ewc_cache).parent.mkdir(parents=True, exist_ok=True)
            ewc.save(_ewc_cache)
        model.__class__ = build_model(self.cfg, in_channels, self._active_heads).__class__
        ewc._star = {k: v.cpu() for k, v in ewc._star.items()}
        return ewc

    def _setup_amp(self):
        cfg = self.cfg
        use_amp = self.device.type == "cuda" and getattr(cfg.train, "use_amp", True)
        amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
        scaler = GradScaler() if use_amp and amp_dtype == torch.float16 else None
        return use_amp, amp_dtype, scaler

    def _setup_run_dir(self):
        cfg = self.cfg
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg.model.architecture}_{cfg.data.mode}"
        if self.resume:
            existing = sorted(cfg.train.checkpoint_dir.glob(
                f"*_{cfg.model.architecture}_{cfg.data.mode}/last_{cfg.model.architecture}.pt"
            ))
            if existing:
                run_dir = existing[-1].parent
                run_id  = run_dir.name
                logger.info(f"Resuming run: {run_id}")
                return run_id, run_dir
            logger.warning("--resume: no previous run found — starting fresh.")
        run_dir = cfg.train.checkpoint_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    def _training_loop(
        self, trainer, train_loader, val_loader, test_loader,
        cm: CheckpointManager, class_weight, train_counts,
        stop_on_f1, best_val_f1, best_val_loss, patience_counter, start_epoch,
        step_per_batch, optimizer, scheduler,
    ) -> None:
        cfg = self.cfg
        save_last_every = getattr(cfg.train, "save_last_every", 1)
        train_start = time.time()
        epoch_log: list[dict] = []

        for epoch in range(start_epoch, cfg.train.epochs + 1):
            if self._use_epoch_adaptive and train_counts is not None:
                class_weight = epoch_adaptive_class_weights(
                    train_counts, epoch, cfg.train.epochs, cfg.model.num_classes, self.device
                )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            train_loss = trainer.train_epoch(train_loader, epoch, cfg.train.epochs, class_weight)
            val_m = trainer.evaluate(val_loader, self._is_binary, class_weight)
            val_loss, val_acc, val_conf = val_m["loss"], val_m["acc"], val_m["conf"]
            val_f1, val_f1w = val_m["f1_macro"], val_m["f1_weighted"]
            val_prec, val_rec = val_m["precision_macro"], val_m["recall_macro"]
            if not step_per_batch:
                # CosineAnnealingLR takes no argument; ReduceLROnPlateau takes val_loss
                if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step()
                else:
                    scheduler.step(val_loss)

            epoch_time = time.time() - t0
            improved = (val_f1 > best_val_f1) if stop_on_f1 else (val_loss < best_val_loss)
            lr_now = optimizer.param_groups[-1]['lr']
            logger.info(
                f"Epoch {epoch:03d}/{cfg.train.epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"acc={val_acc:.4f} | f1={val_f1:.4f} | f1w={val_f1w:.4f} | "
                f"prec={val_prec:.4f} | rec={val_rec:.4f} | "
                f"lr={lr_now:.2e} | "
                f"patience={patience_counter}/{cfg.train.patience} | "
                f"{epoch_time:.0f}s" + (" *" if improved else "")
            )
            epoch_peak_vram = 0.0
            if torch.cuda.is_available():
                epoch_peak_vram = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            epoch_log.append({
                "epoch": epoch, "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6), "val_acc": round(val_acc, 6),
                "val_f1": round(val_f1, 6), "val_f1w": round(val_f1w, 6),
                "val_prec": round(val_prec, 6), "val_rec": round(val_rec, 6),
                "lr": lr_now, "epoch_time_s": round(epoch_time, 1),
                "peak_vram_gb": epoch_peak_vram, "best": improved,
            })
            if improved:
                best_val_f1 = val_f1; best_val_loss = val_loss; patience_counter = 0
                cm.save_best(trainer.model, epoch=epoch, val_loss=val_loss,
                             val_acc=val_acc, val_conf=val_conf, val_f1=val_f1, val_f1_weighted=val_f1w)
            else:
                patience_counter += 1

            if save_last_every > 0 and epoch % save_last_every == 0:
                cm.save_last(trainer.model, optimizer, scheduler, epoch=epoch,
                             best_val_f1=best_val_f1, best_val_loss=best_val_loss,
                             patience_counter=patience_counter, val_loss=val_loss,
                             val_acc=val_acc, val_conf=val_conf, val_f1=val_f1, val_f1_weighted=val_f1w)

            if patience_counter >= cfg.train.patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        test_m = trainer.evaluate(test_loader, self._is_binary, class_weight)
        test_acc, test_conf = test_m["acc"], test_m["conf"]
        test_f1, test_f1w = test_m["f1_macro"], test_m["f1_weighted"]
        test_prec, test_rec = test_m["precision_macro"], test_m["recall_macro"]
        logger.info(
            f"Test  acc={test_acc:.4f} | f1={test_f1:.4f} | f1w={test_f1w:.4f} | "
            f"prec={test_prec:.4f} | rec={test_rec:.4f} | conf={test_conf:.4f}"
        )
        total_time = int(time.time() - train_start)
        h, rem = divmod(total_time, 3600)
        m, s = divmod(rem, 60)
        logger.info(f"Total time: {h:02d}h {m:02d}m {s:02d}s")
        logger.info(f"Best checkpoint → {cm.best_path}")
        logger.info(f"To evaluate: uv run evaluate --checkpoint {cm.best_path}")

        # Save training artifacts to results_dir (separate from model weights)
        import csv as _csv, json as _json
        res_dir = cfg.train.results_dir / cm.run_dir.name
        res_dir.mkdir(parents=True, exist_ok=True)

        if epoch_log:
            log_path = res_dir / "training_log.csv"
            with open(log_path, "w", newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=epoch_log[0].keys())
                writer.writeheader(); writer.writerows(epoch_log)
            logger.info(f"training_log.csv → {log_path}")

        epoch_times = [r["epoch_time_s"] for r in epoch_log]
        num_params = sum(p.numel() for p in trainer.model.parameters())
        peak_vram_gb = peak_reserved_gb = 0.0
        gpu_name = "cpu"
        if torch.cuda.is_available():
            peak_vram_gb     = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            peak_reserved_gb = round(torch.cuda.max_memory_reserved()  / 1024**3, 3)
            gpu_name         = torch.cuda.get_device_name(0)
        summary_path = res_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            _json.dump({
                "run_id":              cm.run_dir.name,
                "architecture":        cfg.model.architecture,
                "dataset_pt":          getattr(self, "_dataset_pt", ""),
                "num_classes":         cfg.model.num_classes,
                "num_params":          num_params,
                "epochs_trained":      len(epoch_log),
                "best_val_f1":         round(best_val_f1, 6),
                "best_val_loss":       round(best_val_loss, 6),
                "test_acc":            round(test_acc, 6),
                "test_f1":             round(test_f1, 6),
                "test_f1w":            round(test_f1w, 6),
                "test_prec":           round(test_prec, 6),
                "test_rec":            round(test_rec, 6),
                "test_prec_weighted":  round(test_m["precision_weighted"], 6),
                "test_rec_weighted":   round(test_m["recall_weighted"],    6),
                "test_per_class":      test_m["per_class"],
                "test_conf":           round(test_conf, 6),
                "total_time_s":        total_time,
                "avg_epoch_time_s":    round(sum(epoch_times) / len(epoch_times), 1) if epoch_times else 0,
                "min_epoch_time_s":    round(min(epoch_times), 1) if epoch_times else 0,
                "max_epoch_time_s":    round(max(epoch_times), 1) if epoch_times else 0,
                "gpu":                 gpu_name,
                "peak_vram_gb":        peak_vram_gb,
                "peak_reserved_gb":    peak_reserved_gb,
            }, f, indent=2)
        logger.info(f"training_summary.json → {summary_path}")

        # Training curves plot
        if epoch_log:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                epochs    = [r["epoch"]      for r in epoch_log]
                tr_loss   = [r["train_loss"] for r in epoch_log]
                val_loss_ = [r["val_loss"]   for r in epoch_log]
                val_f1_   = [r["val_f1"]     for r in epoch_log]
                val_f1w_  = [r["val_f1w"]    for r in epoch_log]
                vram_     = [r.get("peak_vram_gb", 0.0) for r in epoch_log]
                best_ep   = next((r["epoch"] for r in epoch_log if r["best"]), None)
                has_vram  = any(v > 0 for v in vram_)

                nrows = 3 if has_vram else 2
                fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows), sharex=True)
                ax1, ax2 = axes[0], axes[1]

                ax1.plot(epochs, tr_loss,   label="train loss",  color="steelblue")
                ax1.plot(epochs, val_loss_, label="val loss",    color="tomato")
                if best_ep:
                    ax1.axvline(best_ep, color="gray", linestyle="--", alpha=0.6, label=f"best (ep {best_ep})")
                ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
                ax1.set_title("Training Curves")

                ax2.plot(epochs, val_f1_,  label="val F1 macro",    color="seagreen")
                ax2.plot(epochs, val_f1w_, label="val F1 weighted",  color="darkorange", linestyle="--")
                if best_ep:
                    ax2.axvline(best_ep, color="gray", linestyle="--", alpha=0.6)
                ax2.set_ylabel("F1"); ax2.legend(); ax2.grid(True, alpha=0.3)
                if not has_vram:
                    ax2.set_xlabel("Epoch")

                if has_vram:
                    ax3 = axes[2]
                    ax3.plot(epochs, vram_, label="peak VRAM (GB)", color="mediumpurple")
                    ax3.set_ylabel("VRAM (GB)"); ax3.set_xlabel("Epoch")
                    ax3.legend(); ax3.grid(True, alpha=0.3)

                plt.tight_layout()
                curves_path = res_dir / "training_curves.png"
                fig.savefig(curves_path, dpi=150)
                plt.close(fig)
                logger.info(f"training_curves.png → {curves_path}")
            except Exception as e:
                logger.warning(f"Could not save training_curves.png: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train GNN vulnerability detector")
    parser.add_argument("--config", type=str, default="configs/lmgcn/binary.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest last_*.pt for this arch/mode.")
    args = parser.parse_args()

    session = TrainingSession.from_args(args)
    session.run(config_path=args.config)



if __name__ == "__main__":
    main()
