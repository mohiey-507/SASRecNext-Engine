from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import torch

from engine.utils import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


class DataProcessor:
    """Processes raw MovieLens data into training/evaluation artifacts."""

    def __init__(self, raw_dir: Path, output_dir: Path) -> None:
        self._raw_dir = raw_dir
        self._output_dir = output_dir

    def process(self) -> None:
        """Run the full preprocessing pipeline."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        ratings = self._load_ratings()
        movies = self._load_movies()

        id_map, ratings = self._remap_item_ids(ratings)
        user_seqs, val_targets, test_targets = self._split_chronologically(ratings)

        self._save_sequences(user_seqs, val_targets, test_targets)
        self._save_id_mapping(id_map)
        self._save_movie_metadata(movies, id_map)
        self._save_stats(user_seqs, val_targets, id_map)

        logger.info("Preprocessing complete — artifacts saved to %s", self._output_dir)

    # Loading
    def _load_ratings(self) -> pd.DataFrame:
        path = self._raw_dir / "ratings.dat"
        logger.info("Loading ratings from %s", path)
        df = pd.read_csv(
            path,
            sep="::",
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python",
        )
        logger.info("Loaded %d ratings from %d users", len(df), df["user_id"].nunique())
        return df

    def _load_movies(self) -> pd.DataFrame:
        path = self._raw_dir / "movies.dat"
        logger.info("Loading movies from %s", path)
        return pd.read_csv(
            path,
            sep="::",
            names=["movie_id", "title", "genres"],
            engine="python",
            encoding="latin-1",
        )

    # ID remapping
    def _remap_item_ids(self, ratings: pd.DataFrame) -> tuple[dict[int, int], pd.DataFrame]:
        """Remap sparse MovieIDs to contiguous 1-indexed IDs (0 = padding)."""
        unique_items = sorted(ratings["movie_id"].unique())
        original_to_internal: dict[int, int] = {int(orig): idx + 1 for idx, orig in enumerate(unique_items)}

        ratings = ratings.copy()
        ratings["movie_id"] = ratings["movie_id"].map(original_to_internal)

        logger.info(
            "Remapped %d unique items to contiguous IDs [1, %d]",
            len(unique_items),
            len(unique_items),
        )
        return original_to_internal, ratings

    # Splitting
    def _split_chronologically(
        self, ratings: pd.DataFrame
    ) -> tuple[dict[int, list[int]], dict[int, int], dict[int, int]]:
        """Leave-one-out split: train = all[:-2], val = [-2], test = [-1]."""
        user_seqs: dict[int, list[int]] = {}
        val_targets: dict[int, int] = {}
        test_targets: dict[int, int] = {}

        grouped = ratings.sort_values(["timestamp", "movie_id"]).groupby("user_id")["movie_id"]
        for user_id, items in grouped:
            item_list: list[int] = items.tolist()
            if len(item_list) < 3:
                logger.warning("User %s has < 3 interactions, skipping", user_id)
                continue
            uid = int(str(user_id))
            user_seqs[uid] = item_list[:-2]
            val_targets[uid] = int(str(item_list[-2]))
            test_targets[uid] = int(str(item_list[-1]))

        avg_len = sum(len(s) for s in user_seqs.values()) / max(len(user_seqs), 1)
        logger.info(
            "Split complete — %d users | avg train seq len: %.1f | min: %d | max: %d",
            len(user_seqs),
            avg_len,
            min(len(s) for s in user_seqs.values()),
            max(len(s) for s in user_seqs.values()),
        )
        return user_seqs, val_targets, test_targets

    # Saving
    def _save_sequences(
        self,
        user_seqs: dict[int, list[int]],
        val_targets: dict[int, int],
        test_targets: dict[int, int],
    ) -> None:
        torch.save(user_seqs, self._output_dir / "user_sequences.pt")
        torch.save(val_targets, self._output_dir / "val_targets.pt")
        torch.save(test_targets, self._output_dir / "test_targets.pt")
        logger.info("Saved sequence artifacts (train / val / test)")

    def _save_id_mapping(self, original_to_internal: dict[int, int]) -> None:
        internal_to_original = {v: k for k, v in original_to_internal.items()}
        mapping = {
            "original_to_internal": {str(k): v for k, v in original_to_internal.items()},
            "internal_to_original": {str(k): v for k, v in internal_to_original.items()},
        }
        path = self._output_dir / "id_mapping.json"
        path.write_text(json.dumps(mapping, indent=2, sort_keys=True))
        logger.info("Saved ID mapping (%d items) to %s", len(original_to_internal), path)

    def _save_movie_metadata(self, movies: pd.DataFrame, original_to_internal: dict[int, int]) -> None:
        metadata: dict[str, dict[str, object]] = {}
        for _, row in movies.iterrows():
            original_id = int(row["movie_id"])
            internal_id = original_to_internal.get(original_id)
            if internal_id is None:
                continue
            metadata[str(internal_id)] = {
                "original_id": original_id,
                "title": str(row["title"]),
                "genres": str(row["genres"]),
            }
        path = self._output_dir / "movie_metadata.json"
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        logger.info("Saved metadata for %d movies", len(metadata))

    def _save_stats(
        self,
        user_seqs: dict[int, list[int]],
        val_targets: dict[int, int],
        original_to_internal: dict[int, int],
    ) -> None:
        seq_lens = [len(s) for s in user_seqs.values()]
        stats = {
            "n_users": len(user_seqs),
            "n_items": len(original_to_internal),
            "n_interactions": sum(seq_lens) + len(val_targets) * 2,
            "avg_train_seq_len": round(sum(seq_lens) / len(seq_lens), 2),
            "min_train_seq_len": min(seq_lens),
            "max_train_seq_len": max(seq_lens),
        }
        path = self._output_dir / "stats.json"
        path.write_text(json.dumps(stats, indent=2, sort_keys=True))
        logger.info("Dataset stats: %s", json.dumps(stats))
