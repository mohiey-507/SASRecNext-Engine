"""Dataset for exact sequence length evaluation."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ExactLengthEvalDataset(Dataset[dict[str, Tensor]]):
    """Evaluation dataset for exact sequence lengths.

    Filters users by minimum sequence length, calculates a dynamic stride to
    generate approximately `target_samples` chunks, and properly slices the
    history mask for each historical chunk to avoid future leakage.
    """
    def __init__(
        self,
        user_sequences: dict[int, list[int]],
        targets: dict[int, int],
        n_items: int,
        window_size: int,
        target_samples: int | None = None,
    ) -> None:
        self._n_items = n_items
        self._window_size = window_size
        self._samples: list[dict[str, Tensor]] = []

        min_required_len = window_size + 1
        eligible_uids = [
            uid
            for uid in sorted(user_sequences.keys())
            if len(user_sequences[uid]) + (1 if uid in targets else 0) >= min_required_len
        ]

        if not eligible_uids:
            return

        full_sequences = {
            uid: user_sequences[uid] + ([targets[uid]] if uid in targets else [])
            for uid in eligible_uids
        }

        stride = 1
        if target_samples is not None and target_samples > 0:
            best_stride = 1
            best_diff = float("inf")
            max_len = max(len(seq) for seq in full_sequences.values())
            for s in range(1, max_len + 1):
                count = sum((len(seq) - min_required_len) // s + 1 for seq in full_sequences.values())
                diff = abs(count - target_samples)
                if diff < best_diff:
                    best_diff = diff
                    best_stride = s

                if count < target_samples:
                    break
            stride = best_stride

        for uid in eligible_uids:
            seq = full_sequences[uid]
            seq_len = len(seq)

            for end_idx in range(seq_len, min_required_len - 1, -stride):
                start_idx = end_idx - min_required_len
                chunk = seq[start_idx:end_idx]

                input_ids = chunk[:-1]
                target_id = chunk[-1]
                history = seq[: end_idx - 1]

                self._add_chunk(input_ids, target_id, history)

    def _add_chunk(self, input_ids: list[int], target_id: int, history: list[int]) -> None:
        history_mask = torch.zeros(self._n_items + 1, dtype=torch.bool)
        history_mask[history] = True

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_id": torch.tensor(target_id, dtype=torch.long),
            "history_mask": history_mask,
        }

        self._samples.append(batch)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self._samples[index]
