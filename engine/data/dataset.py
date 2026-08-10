from __future__ import annotations

import random

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SASRecTrainDataset(Dataset[dict[str, Tensor]]):
    """Training dataset: returns input/target sequence pairs per user.

    Uses an overlapping sliding window to dynamically chunk long user sequences
    so that 100% of historical transitions are utilized without context truncation.
    """

    def __init__(
        self,
        user_sequences: dict[int, list[int]],
        max_seq_len: int,
        stride: int,
        train_seq_mode: str = "sliding_window",
    ) -> None:
        self._max_seq_len = max_seq_len
        self._samples: list[dict[str, Tensor]] = []

        for _, sequence in user_sequences.items():
            if len(sequence) < 2:
                continue

            if train_seq_mode == "recent_only":
                self._extract_recent_only(sequence)
            else:
                self._extract_sliding_window(sequence, stride)

    def _add_chunk(self, chunk: list[int]) -> None:
        input_ids = chunk[:-1]
        target_ids = chunk[1:]

        # Left-pad with 0 (padding token)
        pad_len = self._max_seq_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_ids = [0] * pad_len + target_ids

        self._samples.append(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "target_ids": torch.tensor(target_ids, dtype=torch.long),
            }
        )

    def _extract_recent_only(self, sequence: list[int]) -> None:
        start = max(0, len(sequence) - self._max_seq_len - 1)
        chunk = sequence[start:]
        self._add_chunk(chunk)

    def _extract_sliding_window(self, sequence: list[int], stride: int) -> None:
        added_ends = set()

        # Overlapping sliding window anchored at the target
        for end in range(self._max_seq_len, len(sequence), stride):
            # window ends at `end` (exclusive), i.e. positions [end-max_seq_len-1, end)
            start = max(0, end - self._max_seq_len - 1)
            chunk = sequence[start:end]
            self._add_chunk(chunk)
            added_ends.add(end)

        # ALWAYS include one final window ending exactly at len(sequence),
        # so the tail of every user's history is covered.
        end = len(sequence)
        if end not in added_ends:
            start = max(0, end - self._max_seq_len - 1)
            chunk = sequence[start:end]
            self._add_chunk(chunk)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self._samples[index]


class SASRecEvalDataset(Dataset[dict[str, Tensor]]):
    """Evaluation dataset: returns input sequence, target, and history mask.

    Used for both validation and test evaluation. The caller controls what
    sequences and targets to pass in (e.g., appending val targets to train
    sequences for test evaluation).
    """

    def __init__(
        self,
        user_sequences: dict[int, list[int]],
        targets: dict[int, int],
        n_items: int,
        max_seq_len: int,
        mode: str = "full",
    ) -> None:
        self._max_seq_len = max_seq_len
        self._n_items = n_items
        self._mode = mode
        self._user_ids = sorted(targets)
        self._sequences = [user_sequences[uid] for uid in self._user_ids]
        self._targets = [targets[uid] for uid in self._user_ids]

        self._negatives = []
        if self._mode in ("uni100", "both"):
            for sequence, target in zip(self._sequences, self._targets, strict=True):
                history_mask = torch.zeros(self._n_items + 1, dtype=torch.bool)
                history_mask[sequence] = True

                neg: list[int] = []
                while len(neg) < 99:
                    item = random.randint(1, self._n_items)
                    if not history_mask[item] and item != target:
                        neg.append(item)
                self._negatives.append(torch.tensor(neg, dtype=torch.long))

    def __len__(self) -> int:
        return len(self._user_ids)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sequence = self._sequences[index]
        target = self._targets[index]

        # Truncate and left-pad
        input_ids = sequence[-self._max_seq_len :]
        pad_len = self._max_seq_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids

        # History mask: True for all previously interacted items
        history_mask = torch.zeros(self._n_items + 1, dtype=torch.bool)
        history_mask[sequence] = True

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_id": torch.tensor(target, dtype=torch.long),
            "history_mask": history_mask,
        }

        if self._mode in ("uni100", "both"):
            batch["negatives"] = self._negatives[index]

        return batch
