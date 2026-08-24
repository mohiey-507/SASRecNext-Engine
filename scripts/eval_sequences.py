"""Evaluate the SASRec model on long sequences of exact lengths."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

import torch
from engine.data import DataPipeline, ExactLengthEvalDataset, create_dataloader
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
    parser = argparse.ArgumentParser(description="Evaluate SASRec on long sequences")
    parser.add_argument("--config", type=Path, default=Path("configs/ml-10m/sasrecnext_tied.yaml"))
    parser.add_argument(
        "--min_history", type=int, default=None, help="Minimum history length. Defaults to max(max_seq_lens) + 1."
    )
    parser.add_argument(
        "--max_seq_lens", type=str, default="2,4,8,16,32,48,64,80,96,112,128,144,160,176,192,200,225,250,275,300", help="Comma-separated sequence lengths"
    )
    parser.add_argument(
        "--eval_set", type=str, choices=["val", "test", "both"], default="test", help="Which set to evaluate on"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # We don't override global max_seq_len here yet, as it changes per loop iteration
    # but we will override eval_set
    cfg_dict = cfg.model_dump()
    cfg_dict["evaluation"]["eval_set"] = args.eval_set
    cfg = Config(**cfg_dict)

    device = resolve_device(cfg.runtime.device)

    log_dir = Path(cfg.data.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    set_global_log_file(log_dir, "eval_sequences.log")

    set_seed(cfg.runtime.seed)

    lens = [int(x) for x in args.max_seq_lens.split(",")]
    min_history = args.min_history if args.min_history is not None else max(lens) + 1

    logger.info("Starting long sequence investigation")
    logger.info("Min History: %d | Windows: %s", min_history, lens)
    logger.info("Device: %s | Seed: %d", device, cfg.runtime.seed)
    logger.info("Evaluating on %s set(s)", cfg.evaluation.eval_set)

    # Data
    pipeline = DataPipeline(cfg.data)
    pipeline.ensure_data()
    user_seqs, val_targets, test_targets, n_items = pipeline.load_artifacts()

    # Pre-filter cohort based on min_history so the set of users is identical across all windows
    power_users = {uid for uid, seq in user_seqs.items() if len(seq) >= min_history}
    power_user_seqs = {uid: seq for uid, seq in user_seqs.items() if uid in power_users}
    power_val_targets = {uid: t for uid, t in val_targets.items() if uid in power_users}
    power_test_targets = {uid: t for uid, t in test_targets.items() if uid in power_users}

    logger.info("Found %d power users out of %d total users", len(power_users), len(user_seqs))

    # Base Model Config setup
    model_cls = MODEL_REGISTRY[cfg.model.model_type]
    ckpt_path = Path(cfg.data.checkpoint_dir) / "best_model.pt"

    if not ckpt_path.exists():
        logger.info("Model checkpoint not found at %s. Downloading from release...", ckpt_path)
        download_release_asset(cfg, "best_model.pt", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    uncompiled = {(k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()}

    eval_set = cfg.evaluation.eval_set

    for w in lens:
        logger.info("Evaluating exact window size: %d", w)

        if cfg.model.model_type == "SASRec" and w > cfg.model.max_seq_len:
            logger.info("Skipping window %d for SASRec (max_seq_len=%d)", w, cfg.model.max_seq_len)
            continue

        # Re-instantiate config for the specific window
        w_cfg_dict = cfg.model_dump()
        w_cfg_dict["evaluation"]["max_seq_len"] = w
        if w > cfg.model.max_seq_len:
            w_cfg_dict["model"]["max_seq_len"] = w
        w_cfg = Config(**w_cfg_dict)

        model = model_cls(n_items=n_items, cfg=w_cfg.model).to(device)

        model.load_state_dict(uncompiled)

        eval_model: nn.Module = torch.compile(model, dynamic=True) if w_cfg.runtime.compile_model else model  # type: ignore
        eval_model.eval()

        if eval_set in ("val", "both"):
            val_ds = ExactLengthEvalDataset(
                user_sequences=power_user_seqs,
                targets=power_val_targets,
                n_items=n_items,
                window_size=w,
            )
            val_loader = create_dataloader(
                val_ds,
                batch_size=w_cfg.training.eval_batch_size,
                n_workers=w_cfg.runtime.n_workers,
                seed=w_cfg.runtime.seed,
            )
            val_evaluator = Evaluator(
                model=eval_model,
                eval_loader=val_loader,
                metrics=w_cfg.evaluation.metrics,
                top_k=w_cfg.evaluation.top_k,
                device=device,
                mode=w_cfg.evaluation.mode,
            )
            val_metrics = val_evaluator.evaluate()
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
            logger.info("Val %d (N=%d) — %s", w, len(val_ds), metrics_str)

        if eval_set in ("test", "both"):
            # For test evaluation, the history includes the val_targets.
            # We build full sequences first for power users.
            test_seqs = pipeline.build_eval_sequences(power_user_seqs, power_val_targets)
            test_ds = ExactLengthEvalDataset(
                user_sequences=test_seqs,
                targets=power_test_targets,
                n_items=n_items,
                window_size=w,
            )
            test_loader = create_dataloader(
                test_ds,
                batch_size=w_cfg.training.eval_batch_size,
                n_workers=w_cfg.runtime.n_workers,
                seed=cfg.runtime.seed,
            )
            test_evaluator = Evaluator(
                model=eval_model,
                eval_loader=test_loader,
                metrics=w_cfg.evaluation.metrics,
                top_k=w_cfg.evaluation.top_k,
                device=device,
                mode=w_cfg.evaluation.mode,
            )
            test_metrics = test_evaluator.evaluate()
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items())
            logger.info("Test %d (N=%d) — %s", w, len(test_ds), metrics_str)

    logger.info("Completed long sequence investigation.")


if __name__ == "__main__":
    main()
