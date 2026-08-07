from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from torch.utils.data import Dataset


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002
    random.seed(worker_seed)


def create_dataloader(
    dataset: Dataset[Any],
    batch_size: int,
    shuffle: bool = False,
    n_workers: int = 0,
    seed: int | None = None,
    **kwargs: Any,
) -> DataLoader[Any]:
    """Factory for DataLoader with sensible defaults and forwarded kwargs.

    Exposes **kwargs so callers can pass pin_memory, persistent_workers,
    prefetch_factor, etc. without modifying this function's signature.
    """
    if seed is not None:
        kwargs.setdefault("worker_init_fn", _seed_worker)
        if "generator" not in kwargs:
            g = torch.Generator()
            g.manual_seed(seed)
            kwargs["generator"] = g

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_workers,
        **kwargs,
    )
