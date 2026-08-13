"""Evaluate the SASRec model on MovieLens."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

import torch
from engine.data import DataPipeline, SASRecEvalDataset, create_dataloader
from engine.evaluation import Evaluator
from engine.models import MODEL_REGISTRY
from engine.utils import (
    Config,
    download_release_asset,
    get_logger,
    load_config,
    resolve_device,
    set_global_log_file,
    set_seed,
)

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval SASRec")
    parser.add_argument("--config", type=Path, default=Path("configs/ml-10m/sasrecnext_tied.yaml"))
    parser.add_argument("--max_seq_len", type=int, default=None, help="Override config evaluation max_seq_len")
    parser.add_argument(
        "--eval_set", type=str, choices=["val", "test", "both"], default=None, help="Override config eval_set"
    )
    parser.add_argument(
        "--override",
        nargs="+",
        help="Override config values, e.g. --override training.eval_batch_size=8",
    )

    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)

    # Override config values if passed
    cfg_dict = cfg.model_dump()
    if args.max_seq_len is not None:
        cfg_dict["evaluation"]["max_seq_len"] = args.max_seq_len
        cfg_dict["model"]["max_seq_len"] = args.max_seq_len
    if args.eval_set is not None:
        cfg_dict["evaluation"]["eval_set"] = args.eval_set

    # Re-instantiate to validate overrides
    cfg = Config(**cfg_dict)

    device = resolve_device(cfg.runtime.device)
    set_global_log_file(Path(cfg.data.log_dir), "eval.log")
    set_seed(cfg.runtime.seed)

    logger.info("Device: %s | Seed: %d", device, cfg.runtime.seed)
    logger.info("Evaluating on %s set(s)", cfg.evaluation.eval_set)

    # Data
    pipeline = DataPipeline(cfg.data)
    pipeline.ensure_data()
    user_seqs, val_targets, test_targets, n_items = pipeline.load_artifacts()

    # Model
    model_cls = MODEL_REGISTRY[cfg.model.model_type]
    model = model_cls(n_items=n_items, cfg=cfg.model).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = param_bytes / (1024**2)

    logger.info("Model parameters: %s", f"{n_params:,}")
    logger.info("Model size on disk/RAM: %.2f MB", model_size_mb)

    ckpt_path = Path(cfg.data.checkpoint_dir) / "best_model.pt"
    if not ckpt_path.exists():
        logger.info("Model checkpoint not found at %s. Downloading from release...", ckpt_path)
        asset_name = next((f for f in cfg.assets.files if "best_model.pt" in f), f"{args.config.stem}_best_model.pt")
        download_release_asset(cfg, asset_name, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Handle potentially compiled weights
    uncompiled = {(k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(uncompiled)
    logger.info("Loaded best checkpoint from epoch %d", ckpt.get("epoch", 0))

    eval_model: nn.Module = torch.compile(model, dynamic=True) if cfg.runtime.compile_model else model  # type: ignore

    model.eval()

    eval_set = cfg.evaluation.eval_set

    # Validation Set Evaluation
    if eval_set in ("val", "both"):
        logger.info("Running validation evaluation ...")
        val_ds = SASRecEvalDataset(
            user_seqs,
            val_targets,
            n_items,
            max_seq_len=cfg.evaluation.max_seq_len,
            mode=cfg.evaluation.mode,
        )
        val_loader = create_dataloader(
            val_ds,
            batch_size=cfg.training.eval_batch_size,
            n_workers=cfg.runtime.n_workers,
            seed=cfg.runtime.seed,
        )
        val_evaluator = Evaluator(
            model=eval_model,
            eval_loader=val_loader,
            metrics=cfg.evaluation.metrics,
            top_k=cfg.evaluation.top_k,
            device=device,
            mode=cfg.evaluation.mode,
        )
        val_metrics = val_evaluator.evaluate()
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
        logger.info("Val results — %s", metrics_str)

    # Test Set Evaluation
    if eval_set in ("test", "both"):
        logger.info("Running test evaluation ...")
        test_seqs = pipeline.build_eval_sequences(user_seqs, val_targets)
        test_ds = SASRecEvalDataset(
            test_seqs,
            test_targets,
            n_items,
            max_seq_len=cfg.evaluation.max_seq_len,
            mode=cfg.evaluation.mode,
        )
        test_loader = create_dataloader(
            test_ds,
            batch_size=cfg.training.eval_batch_size,
            n_workers=cfg.runtime.n_workers,
            seed=cfg.runtime.seed,
        )
        test_evaluator = Evaluator(
            model=eval_model,
            eval_loader=test_loader,
            metrics=cfg.evaluation.metrics,
            top_k=cfg.evaluation.top_k,
            device=device,
            mode=cfg.evaluation.mode,
        )
        test_metrics = test_evaluator.evaluate()
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items())
        logger.info("Test results — %s", metrics_str)


if __name__ == "__main__":
    main()
