from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor

    from engine.utils.config import ModelConfig


class BaseRecommender(nn.Module, ABC):
    """Abstract Base Class for all recommendation models in the engine.

    Any custom model must inherit from this class and implement the
    required methods to ensure compatibility with the training loop.
    """

    def __init__(self, n_items: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_items = n_items

    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        return_last_only: bool = False,
    ) -> Tensor:
        """Produce logits over the full item catalog.

        Args:
            input_ids: (B, L) item ID sequences where 0 = padding.
            return_last_only: If True, only computes logits for the final step.

        Returns:
            (B, L, n_items + 1) if return_last_only=False
            (B, n_items + 1) if return_last_only=True
        """
        pass

    def _init_weights(self) -> None:
        """Initialize the model's weights. (Optional)"""
        pass
