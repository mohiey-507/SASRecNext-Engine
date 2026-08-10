from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from engine.utils import get_logger

if TYPE_CHECKING:
    import torch.nn as nn
    from torch import Tensor
    from torch.utils.data import DataLoader

    from engine.evaluation import Evaluator
    from engine.utils import Config

logger = get_logger(__name__)


class Trainer:
    """Orchestrates the training loop with AMP, EMA loss, early stopping, and checkpointing."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        loss_fn: nn.Module,
        train_loader: DataLoader[dict[str, Tensor]],
        evaluator: Evaluator,
        config: Config,
        device: torch.device,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._loss_fn = loss_fn
        self._train_loader = train_loader
        self._evaluator = evaluator
        self._cfg = config
        self._device = device

        # GradScaler only active on CUDA or MPS with AMP
        amp_enabled = config.runtime.enable_amp and device.type in ("cuda", "mps")
        self._scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
        self._amp_enabled = amp_enabled

        self._best_metric = 0.0
        self._best_metrics_dict: dict[str, float] = {}
        self._patience_counter = 0

    # Public API
    def fit(self, epoch_callback: Callable[[int, dict[str, float]], None] | None = None) -> dict[str, float]:
        """Run the full training loop with validation and early stopping."""
        logger.info("Starting training for %d epochs", self._cfg.training.epochs)

        for epoch in range(1, self._cfg.training.epochs + 1):
            train_loss = self._run_train_epoch(epoch)
            val_metrics = self._run_validation(epoch)
            self._log_epoch_summary(epoch, train_loss, val_metrics)

            improved = self._update_best_metric(val_metrics)
            if improved:
                self._save_checkpoint(epoch, val_metrics)

            if epoch_callback is not None:
                epoch_callback(epoch, val_metrics)

            if self._is_patience_exhausted():
                logger.info(
                    "Early stopping triggered after %d epochs without improvement", self._cfg.training.early_stopping
                )
                break

            if self._scheduler is not None:
                self._scheduler.step()
        return self._best_metrics_dict

    # Training
    def _run_train_epoch(self, epoch: int) -> float:
        """Train for one epoch, returning the final EMA loss."""
        self._model.train()
        ema_loss = 0.0
        ema_initialized = False

        pbar = tqdm(
            self._train_loader,
            desc=f"Epoch {epoch:>3d}",
            disable=not self._cfg.runtime.show_progress,
        )

        for batch in pbar:
            loss = self._train_step(batch)
            ema_loss, ema_initialized = _update_ema(loss, ema_loss, ema_initialized)
            pbar.set_postfix({"loss_ema": f"{ema_loss:.4f}"})

        return ema_loss

    def _train_step(self, batch: dict[str, Tensor]) -> float:
        """Single forward + backward + optimizer step with AMP."""
        input_ids = batch["input_ids"].to(self._device)
        target_ids = batch["target_ids"].to(self._device)

        self._optimizer.zero_grad()

        with torch.amp.autocast(device_type=self._device.type, enabled=self._amp_enabled):
            logits = self._model(input_ids)
            loss = self._loss_fn(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
            )

        self._scaler.scale(loss).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()

        return float(loss.item())

    # Validation
    def _run_validation(self, epoch: int) -> dict[str, float]:
        """Evaluate on the validation set and log timing."""
        start = time.perf_counter()
        metrics = self._evaluator.evaluate()
        elapsed = time.perf_counter() - start
        logger.info("Epoch %d — validation took %.1fs", epoch, elapsed)
        return metrics

    # Early stopping & checkpointing
    def _update_best_metric(self, metrics: dict[str, float]) -> bool:
        """Check if the primary metric improved; reset or increment patience."""
        primary_key = self._cfg.evaluation.valid_metric
        primary = metrics[primary_key]
        if primary > self._best_metric:
            self._best_metric = primary
            self._best_metrics_dict = metrics
            self._patience_counter = 0
            return True
        self._patience_counter += 1
        return False

    def _is_patience_exhausted(self) -> bool:
        return self._patience_counter >= self._cfg.training.early_stopping

    def _save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
        ckpt_dir = Path(self._cfg.data.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        logger.info("Saved best checkpoint → %s (epoch %d)", path, epoch)

    def load_best_checkpoint(self) -> None:
        """Reload the best checkpoint into the model before final evaluation."""
        ckpt_path = Path(self._cfg.data.checkpoint_dir) / "best_model.pt"
        if ckpt_path.exists():
            best_ckpt = torch.load(ckpt_path, weights_only=False)
            self._model.load_state_dict(best_ckpt["model_state_dict"])
            logger.info("Reloaded best checkpoint from epoch %d for evaluation", best_ckpt["epoch"])
        else:
            logger.warning("No best checkpoint found at %s. Evaluating with current weights.", ckpt_path)

    # Logging
    def _log_epoch_summary(self, epoch: int, train_loss: float, metrics: dict[str, float]) -> None:
        lr = self._optimizer.param_groups[0]["lr"]
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(
            "Epoch %d — train_loss: %.4f | lr: %.6f | %s",
            epoch,
            train_loss,
            lr,
            metrics_str,
        )


def _update_ema(
    loss: float,
    ema: float,
    initialized: bool,
    alpha: float = 0.05,
) -> tuple[float, bool]:
    """Compute exponential moving average of batch loss."""
    if not initialized:
        return loss, True
    return alpha * loss + (1 - alpha) * ema, True
