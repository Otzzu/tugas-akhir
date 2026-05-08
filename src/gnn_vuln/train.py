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
        self._use_livable        = getattr(cfg.train, "livable_loss", False) and self._use_class_weights

        if self._active_heads:
            if "group"  not in self._active_heads: self._group_loss_weight  = 0.0
            if "binary" not in self._active_heads: self._binary_loss_weight = 0.0

        self._use_supcon    = getattr(cfg.model, "use_supcon", False)
        self._supcon_weight = getattr(cfg.model, "supcon_weight", 0.1) if self._use_supcon else 0.0

    @classmethod
    def from_args(cls, args) -> "TrainingSession":
        cfg = Config.from_yaml(args.config) if Path(args.config).exists() else load_default_config()
        set_seed(cfg.train.seed)
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
        model = build_model(cfg, in_channels, self._active_heads).to(device)
        ewc   = self._setup_ewc(model, train_loader, in_channels)

        use_amp, amp_dtype, scaler = self._setup_amp()
        total_steps = len(train_loader) * cfg.train.epochs
        optimizer, scheduler, step_per_batch = build_optimizer_and_scheduler(model, cfg, total_steps)

        trainer = Trainer(
            model=model, optimizer=optimizer, scheduler=scheduler,
            step_per_batch=step_per_batch, device=device,
            mil_k=self._mil_k, mil_weight=self._mil_weight,
            rank_loss_weight=self._rank_loss_weight, focal_gamma=self._focal_gamma,
            group_loss_weight=self._group_loss_weight, binary_loss_weight=self._binary_loss_weight,
            supcon_fn=supcon_fn, supcon_weight=self._supcon_weight,
            use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler, ewc=ewc,
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
            filter_owasp_top10=getattr(cfg.data, "filter_owasp_top10", False),
            filter_top25=getattr(cfg.data, "filter_top25", False),
            max_per_class=getattr(cfg.data, "max_per_class", 0),
            resample_seed=getattr(cfg.data, "resample_seed", 42),
        )
        bs = cfg.train.batch_size
        dataset = CodeBERTGraphDataset(source=getattr(cfg.data, "source", "bigvul"), **kwargs)
        if use_official:
            val_ds  = CodeBERTGraphDataset(source=source_val,  **kwargs)
            test_ds = CodeBERTGraphDataset(source=source_test, **kwargs)
            train_idx = list(range(len(dataset)))
            loaders = (
                DataLoader(dataset, batch_size=bs, shuffle=True),
                DataLoader(val_ds,  batch_size=bs),
                DataLoader(test_ds, batch_size=bs),
            )
        else:
            train_idx, val_idx, test_idx = dataset.get_splits(seed=cfg.train.seed)
            loaders = (
                DataLoader(dataset[train_idx], batch_size=bs, shuffle=True),
                DataLoader(dataset[val_idx],   batch_size=bs),
                DataLoader(dataset[test_idx],  batch_size=bs),
            )
        return dataset, loaders, train_idx

    def _setup_class_weights(self, dataset, train_idx):
        if not self._use_class_weights:
            return None, None
        cfg = self.cfg
        train_labels = torch.tensor(
            [int(dataset[i].y.item()) for i in train_idx], dtype=torch.long
        )
        counts = torch.bincount(train_labels, minlength=cfg.model.num_classes).float()
        if self._use_livable:
            w = livable_weights(counts, 1, cfg.train.epochs, cfg.model.num_classes, self.device)
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

        for epoch in range(start_epoch, cfg.train.epochs + 1):
            if self._use_livable and train_counts is not None:
                class_weight = livable_weights(
                    train_counts, epoch, cfg.train.epochs, cfg.model.num_classes, self.device
                )
            t0 = time.time()
            train_loss = trainer.train_epoch(train_loader, epoch, cfg.train.epochs, class_weight)
            val_loss, val_acc, val_conf, val_f1, val_f1w = trainer.evaluate(
                val_loader, self._is_binary, class_weight
            )
            if not step_per_batch:
                scheduler.step(val_loss)

            improved = (val_f1 > best_val_f1) if stop_on_f1 else (val_loss < best_val_loss)
            logger.info(
                f"Epoch {epoch:03d}/{cfg.train.epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"acc={val_acc:.4f} | f1={val_f1:.4f} | f1w={val_f1w:.4f} | "
                f"lr={optimizer.param_groups[-1]['lr']:.2e} | "
                f"patience={patience_counter}/{cfg.train.patience} | "
                f"{time.time()-t0:.0f}s" + (" *" if improved else "")
            )
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

        _, test_acc, test_conf, test_f1, test_f1w = trainer.evaluate(
            test_loader, self._is_binary, class_weight
        )
        logger.info(f"Test  acc={test_acc:.4f} | f1={test_f1:.4f} | f1w={test_f1w:.4f} | conf={test_conf:.4f}")
        h, rem = divmod(int(time.time() - train_start), 3600)
        m, s = divmod(rem, 60)
        logger.info(f"Total time: {h:02d}h {m:02d}m {s:02d}s")
        logger.info(f"Best checkpoint → {cm.best_path}")
        logger.info(f"To evaluate: uv run evaluate --checkpoint {cm.best_path}")


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
