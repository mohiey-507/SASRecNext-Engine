"""Train the SASRec model on MovieLens."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from engine.data import DataPipeline, SASRecEvalDataset, SASRecTrainDataset, create_dataloader
from engine.evaluation import Evaluator
from engine.models import SASRecNext
from engine.training import Trainer
from engine.utils import get_logger, load_config, resolve_device, set_global_log_file, set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SASRec")
    parser.add_argument("--config", type=Path, default=Path("configs/ml-10m.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.runtime.device)

    set_global_log_file(Path(cfg.data.log_dir), "train.log")

    set_seed(cfg.runtime.seed)
    logger.info("Device: %s | Seed: %d", device, cfg.runtime.seed)

    # Data
    pipeline = DataPipeline(cfg.data)
    pipeline.ensure_data()
    user_seqs, val_targets, test_targets, n_items = pipeline.load_artifacts()

    train_ds = SASRecTrainDataset(
        user_seqs,
        max_seq_len=cfg.model.max_seq_len,
        stride=cfg.training.stride,
        train_seq_mode=cfg.data.train_seq_mode,
    )
    val_ds = SASRecEvalDataset(
        user_seqs,
        val_targets,
        n_items,
        max_seq_len=cfg.model.max_seq_len,
        mode=cfg.evaluation.mode,
    )

    train_loader = create_dataloader(
        train_ds,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        n_workers=cfg.runtime.n_workers,
        seed=cfg.runtime.seed,
    )
    val_loader = create_dataloader(
        val_ds,
        batch_size=cfg.training.eval_batch_size,
        n_workers=cfg.runtime.n_workers,
        seed=cfg.runtime.seed,
    )

    # Model
    model = SASRecNext(n_items=n_items, cfg=cfg.model).to(device)
    train_model: nn.Module = torch.compile(model, dynamic=True) if cfg.runtime.compile_model else model  # type: ignore
    n_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = param_bytes / (1024**2)

    logger.info("Model parameters: %s", f"{n_params:,}")
    logger.info("Model size on disk/RAM: %.2f MB", model_size_mb)

    # Training components
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: cfg.training.lr_decay_factor**epoch
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    val_evaluator = Evaluator(
        model=train_model,
        eval_loader=val_loader,
        metrics=cfg.evaluation.metrics,
        top_k=cfg.evaluation.top_k,
        device=device,
        mode=cfg.evaluation.mode,
    )

    trainer = Trainer(
        model=train_model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        evaluator=val_evaluator,
        config=cfg,
        device=device,
    )
    trainer.fit()

    # Test evaluation
    logger.info("Running test evaluation ...")

    trainer.load_best_checkpoint()

    test_seqs = pipeline.build_eval_sequences(user_seqs, val_targets)
    test_ds = SASRecEvalDataset(
        test_seqs,
        test_targets,
        n_items,
        max_seq_len=cfg.model.max_seq_len,
        mode=cfg.evaluation.mode,
    )
    test_loader = create_dataloader(
        test_ds,
        batch_size=cfg.training.eval_batch_size,
        n_workers=cfg.runtime.n_workers,
        seed=cfg.runtime.seed,
    )
    test_evaluator = Evaluator(
        model=train_model,
        eval_loader=test_loader,
        metrics=cfg.evaluation.metrics,
        top_k=cfg.evaluation.top_k,
        device=device,
        mode=cfg.evaluation.mode
    )
    test_metrics = test_evaluator.evaluate()
    metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items())
    logger.info("Test results — %s", metrics_str)


if __name__ == "__main__":
    main()
