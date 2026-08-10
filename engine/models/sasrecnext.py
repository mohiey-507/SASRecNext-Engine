from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .blocks import RMSNorm, TransformerLayer

if TYPE_CHECKING:
    from torch import Tensor

    from engine.utils import ModelConfig


class SASRecNext(nn.Module):
    """Modern SASRec with RoPE, RMSNorm, SwiGLU, and weight-tied output.

    Uses full-softmax cross-entropy over the entire item catalog.
    No explicit negative sampling — all non-target items are implicit negatives.
    """

    causal_mask: torch.Tensor
    output_layer: nn.Linear | None

    def __init__(self, n_items: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_items = n_items
        self.max_seq_len = cfg.max_seq_len

        self.item_embedding = nn.Embedding(n_items + 1, cfg.d_model, padding_idx=0)
        self.embedding_dropout = nn.Dropout(cfg.embedding_dropout)

        self.tied_weights = cfg.tied_weights
        self.output_layer = nn.Linear(cfg.d_model, n_items + 1, bias=False) if not self.tied_weights else None

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    max_seq_len=cfg.max_seq_len,
                    attn_dropout=cfg.attn_dropout,
                    ffn_dim=cfg.ffn_dim,
                    ffn_dropout=cfg.ffn_dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )

        self.final_norm = RMSNorm(cfg.d_model)

        # Pre-compute causal mask as a persistent-free buffer
        causal_mask = torch.triu(
            torch.full((cfg.max_seq_len, cfg.max_seq_len), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for embeddings (skip padding) and all linear layers."""
        nn.init.xavier_uniform_(self.item_embedding.weight.data[1:])
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

        if not self.tied_weights and self.output_layer is not None:
            nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(
        self,
        input_ids: Tensor,
        return_last_only: bool = False,
    ) -> Tensor:
        """Produce logits over the full item catalog.

        Args:
            input_ids: (B, L) item ID sequences where 0 = padding.
            return_last_only: If True, only computes logits for the final step to save memory.

        Returns:
            (B, L, n_items + 1) if return_last_only=False
            (B, n_items + 1) if return_last_only=True
        """
        padding_mask = input_ids == 0
        attn_mask = self._build_attention_mask(padding_mask)

        x = self.embedding_dropout(self.item_embedding(input_ids))

        for layer in self.layers:
            x = layer(x, attn_mask)

        x = self.final_norm(x)

        if return_last_only:
            x_last = x[:, -1, :]
            logits = x_last @ self.item_embedding.weight.T if self.output_layer is None else self.output_layer(x_last)
        else:
            logits = x @ self.item_embedding.weight.T if self.output_layer is None else self.output_layer(x)

        return logits  # type: ignore[no-any-return]

    def _build_attention_mask(self, padding_mask: Tensor) -> Tensor:
        """Combine causal mask with key-padding mask for SDPA.

        Returns:
            (B, 1, L, L) float mask with -inf for blocked positions.
        """
        seq_len = padding_mask.shape[1]

        causal = self.causal_mask[:seq_len, :seq_len]  # (L, L)

        # Key padding bias: (B, 1, 1, L) — -inf where key is padding, 0 elsewhere
        pad_bias = torch.zeros(
            padding_mask.shape[0],
            1,
            1,
            seq_len,
            dtype=causal.dtype,
            device=causal.device,
        )
        pad_bias.masked_fill_(padding_mask[:, None, None, :], float("-inf"))

        # Broadcast: (1, 1, L, L) + (B, 1, 1, L) → (B, 1, L, L)
        return causal[None, None, :, :] + pad_bias
