"""Dataset for exact sequence length evaluation."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ExactLengthEvalDataset(Dataset[dict[str, Tensor]]):
    """Evaluation dataset for exact sequence lengths."""

    def __init__(
        self,
        user_sequences: dict[int, list[int]],
        targets: dict[int, int],
        n_items: int,
        window_size: int,
    ) -> None:
        self._n_items = n_items
        self._window_size = window_size
        self._samples: list[dict[str, Tensor]] = []

        for uid in sorted(user_sequences.keys()):
            if uid not in targets:
                continue

            sequence = user_sequences[uid]

            if len(sequence) < window_size:
                continue

            input_ids = sequence[-window_size:]
            target_id = targets[uid]

            history_mask = torch.zeros(n_items + 1, dtype=torch.bool)
            history_mask[sequence] = True

            self._samples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "target_id": torch.tensor(target_id, dtype=torch.long),
                    "history_mask": history_mask,
                }
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self._samples[index]
