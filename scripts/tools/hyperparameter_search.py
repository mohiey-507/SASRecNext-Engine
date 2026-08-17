"""Hyperparameter search for SASRec using Optuna."""

from __future__ import annotations

import argparse
import contextlib
import gc
from pathlib import Path
from typing import Any

import optuna
import torch
import torch.nn as nn
from engine.data import DataPipeline, SASRecEvalDataset, SASRecTrainDataset, create_dataloader
from engine.evaluation import Evaluator
from engine.models import MODEL_REGISTRY
from engine.training import Trainer
from engine.utils import Config, get_logger, load_config, resolve_device, set_seed

logger = get_logger(__name__)

SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "ml-1m": {
        "learning_rate": (1e-4, 1e-2),
        "weight_decay": (1e-6, 1e-2),
        "train_batch_size": [64, 128],
        "lr_decay_factor": (0.6, 0.95),
        "n_layers": [2, 3],
        "n_heads": [2, 4],
        "d_model": [64, 128],
        "ffn_dim": [128, 256, 384],
        "dropout": (0.1, 0.3),
        "epochs": 30,
    },
    "ml-10m": {
        "learning_rate": (2e-4, 5e-3),
        "weight_decay": (1e-6, 5e-4),
        "train_batch_size": [128],
        "lr_decay_factor": (0.85, 0.99),
        "n_layers": [3, 4],
        "n_heads": [4, 8],
        "d_model": [256],
        "ffn_dim": [384, 512],
        "dropout": (0.0, 0.15),
        "epochs": 25,
    },
}


def objective(
    trial: optuna.Trial,
    base_cfg: Config,
    user_seqs: dict[int, list[int]],
    val_targets: dict[int, int],
    n_items: int,
) -> float:
    """Optuna objective function for a single trial."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device(f"cuda:{trial.number % 2}")
    else:
        device = resolve_device(base_cfg.runtime.device)

    dataset_name = base_cfg.data.dataset
    space = SEARCH_SPACES.get(dataset_name, SEARCH_SPACES["ml-1m"])

    # Ensure reproducibility for this specific trial
    set_seed(base_cfg.runtime.seed)

    # 1. Sample hyperparameters
    lr = trial.suggest_float("learning_rate", space["learning_rate"][0], space["learning_rate"][1], log=True)
    wd = trial.suggest_float("weight_decay", space["weight_decay"][0], space["weight_decay"][1], log=True)
    train_bs = int(trial.suggest_categorical("train_batch_size", space["train_batch_size"]))
    lr_decay_factor = trial.suggest_float("lr_decay_factor", space["lr_decay_factor"][0], space["lr_decay_factor"][1])

    n_layers = int(trial.suggest_categorical("n_layers", space["n_layers"]))

    d_model = int(trial.suggest_categorical("d_model", space["d_model"]))

    # Ensure d_model is divisible by n_heads
    valid_n_heads = [h for h in space["n_heads"] if d_model % h == 0]
    if not valid_n_heads:
        raise optuna.TrialPruned("No valid n_heads for chosen d_model.")
    n_heads = int(trial.suggest_categorical("n_heads", valid_n_heads))

    ffn_dim = int(trial.suggest_categorical("ffn_dim", space["ffn_dim"]))
    dropout = trial.suggest_float("dropout", space["dropout"][0], space["dropout"][1])

    # 2. Build new config for this trial
    new_training = base_cfg.training.model_copy(
        update={
            "learning_rate": lr,
            "weight_decay": wd,
            "train_batch_size": train_bs,
            "lr_decay_factor": lr_decay_factor,
            "epochs": space.get("epochs", 20),
        }
    )
    new_model = base_cfg.model.model_copy(
        update={
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "ffn_dim": ffn_dim,
            "attn_dropout": 0.0,
            "ffn_dropout": dropout,
            "embedding_dropout": 0.05,
        }
    )

    trial_checkpoint_dir = Path(base_cfg.data.checkpoint_dir) / f"optuna/trial_{trial.number}"
    trial_log_dir = Path(base_cfg.data.log_dir) / f"optuna/trial_{trial.number}"

    new_data = base_cfg.data.model_copy(
        update={
            "checkpoint_dir": str(trial_checkpoint_dir),
            "log_dir": str(trial_log_dir),
        }
    )
    new_runtime = base_cfg.runtime.model_copy(
        update={
            "show_progress": False
        }
    )
    cfg = base_cfg.model_copy(
        update={
            "training": new_training,
            "model": new_model,
            "data": new_data,
            "runtime": new_runtime,
        }
    )

    # Ensure trial directories exist
    trial_log_dir.mkdir(parents=True, exist_ok=True)
    trial_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 3. Setup Data
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

    try:
        # 4. Setup Model
        model_cls = MODEL_REGISTRY[cfg.model.model_type]
        model = model_cls(n_items=n_items, cfg=cfg.model).to(device)

        # 5. Setup Training Components
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            foreach=False,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: cfg.training.lr_decay_factor**epoch
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        val_evaluator = Evaluator(
            model=model,
            eval_loader=val_loader,
            metrics=cfg.evaluation.metrics,
            top_k=cfg.evaluation.top_k,
            device=device,
            mode=cfg.evaluation.mode,
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            train_loader=train_loader,
            evaluator=val_evaluator,
            config=cfg,
            device=device,
        )

        # 6. Train and Return best metric
        def pruning_callback(epoch: int, metrics: dict[str, float]) -> None:
            trial.report(metrics[cfg.evaluation.valid_metric], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        best_metrics_dict = trainer.fit(epoch_callback=pruning_callback)

        # Print the best results for this trial across all modes
        print("\n" + "-" * 40)
        print(f"[Device: {device}] Trial {trial.number} Finished. Best Results:")
        for metric, val in best_metrics_dict.items():
            print(f"  {metric}: {val:.4f}")
        print("-" * 40 + "\n")

        return best_metrics_dict[cfg.evaluation.valid_metric]

    except torch.cuda.OutOfMemoryError:
        print(f"\n[!] Device: {device} | Trial {trial.number} OOM. Too large for GPU. Pruning.\n")
        raise optuna.TrialPruned() from None

    finally:
        with contextlib.suppress(UnboundLocalError):
            del model, optimizer, scheduler, loss_fn, val_evaluator, trainer

        del train_loader
        del val_loader
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter Search for SASRec")
    parser.add_argument("--config", type=Path, default=Path("configs/ml-10m/sasrecnext_tied.yaml"))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--override",
        nargs="+",
        help="Override config values, e.g. --override model.tied_weights=true data.dataset=ml-10m",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config, overrides=args.override)
    set_seed(base_cfg.runtime.seed)

    # Load Data Once to avoid I/O bottlenecks across trials
    pipeline = DataPipeline(base_cfg.data)
    pipeline.ensure_data()
    user_seqs, val_targets, _, n_items = pipeline.load_artifacts()

    logger.info("Starting Optuna Hyperparameter Search (%d trials)", args.n_trials)

    # Create Optuna Study maximizing the target metric with Hyperband Pruning
    epochs_total = SEARCH_SPACES.get(base_cfg.data.dataset, {}).get("epochs", base_cfg.training.epochs)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=5,
        max_resource=epochs_total,
        reduction_factor=3,
    )
    sampler = optuna.samplers.TPESampler(seed=base_cfg.runtime.seed)
    study = optuna.create_study(
        direction="maximize",
        study_name="sasrec_hpo",
        pruner=pruner,
        sampler=sampler,
    )

    # Run Optimization
    study.optimize(
        lambda trial: objective(trial, base_cfg, user_seqs, val_targets, n_items),
        n_trials=args.n_trials,
        n_jobs=2,
    )

    print("\n" + "=" * 50)
    print("Hyperparameter Search Completed!")
    print(f"Best {base_cfg.evaluation.valid_metric}: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
