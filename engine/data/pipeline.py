from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from engine.preprocessing import DataProcessor, download_movielens
from engine.utils import get_logger

if TYPE_CHECKING:
    from engine.utils import DataConfig

logger = get_logger(__name__)


class DataPipeline:
    """Manages the lifecycle of data artifacts: download, preprocess, and load."""

    def __init__(self, cfg_data: DataConfig) -> None:
        self.cfg = cfg_data
        self.raw_dir = Path(cfg_data.raw_data_dir)
        self.processed_dir = Path(cfg_data.processed_data_dir)
        self.dataset = cfg_data.dataset

    def ensure_data(self) -> None:
        """Download and preprocess data if artifacts are missing."""
        download_movielens(self.raw_dir, self.dataset)
        if not (self.processed_dir / "id_mapping.json").exists():
            logger.info("Processed artifacts not found — running preprocessing")
            DataProcessor(raw_dir=self.raw_dir, output_dir=self.processed_dir).process()

    def load_artifacts(self) -> tuple[dict[int, list[int]], dict[int, int], dict[int, int], int]:
        """Load preprocessed sequences and targets from disk."""
        user_seqs: dict[int, list[int]] = torch.load(
            self.processed_dir / "user_sequences.pt",
            weights_only=False,
        )
        val_targets: dict[int, int] = torch.load(
            self.processed_dir / "val_targets.pt",
            weights_only=False,
        )
        test_targets: dict[int, int] = torch.load(
            self.processed_dir / "test_targets.pt",
            weights_only=False,
        )
        stats = json.loads((self.processed_dir / "stats.json").read_text())
        n_items: int = stats["n_items"]

        logger.info(
            "Loaded artifacts — %d users | %d items | %d total interactions",
            stats["n_users"],
            n_items,
            stats["n_interactions"],
        )
        return user_seqs, val_targets, test_targets, n_items

    @staticmethod
    def build_eval_sequences(
        user_seqs: dict[int, list[int]],
        extra_items: dict[int, int] | None = None,
    ) -> dict[int, list[int]]:
        """Optionally append extra items (e.g., val targets) for test evaluation."""
        if extra_items is None:
            return user_seqs
        return {uid: seq + [extra_items[uid]] for uid, seq in user_seqs.items()}
