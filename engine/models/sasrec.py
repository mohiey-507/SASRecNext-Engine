from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from engine.models.blocks import SASRecTransformerLayer

if TYPE_CHECKING:
    from torch import Tensor

    from engine.utils.config import ModelConfig


class SASRec(nn.Module):
    """
    Standard SASRec (2018) architecture.
    Uses Absolute Positional Embeddings, LayerNorm, and standard FFN.
    """

    def __init__(self, n_items: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_items = n_items
        self.max_seq_len = cfg.max_seq_len
        self.d_model = cfg.d_model
        self.tied_weights = cfg.tied_weights

        self.item_emb = nn.Embedding(n_items + 1, self.d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.d_model)
        self.emb_dropout = nn.Dropout(cfg.embedding_dropout)

        self.layers = nn.ModuleList(
            [
                SASRecTransformerLayer(
                    d_model=self.d_model,
                    n_heads=cfg.n_heads,
                    attn_dropout=cfg.attn_dropout,
                    ffn_dim=cfg.ffn_dim,
                    ffn_dropout=cfg.ffn_dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(self.d_model, eps=1e-8)

        self.output_layer = nn.Linear(self.d_model, n_items + 1, bias=False) if not self.tied_weights else None

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for embeddings (skip padding) and all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight.data[1:])
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)

    def forward(self, input_ids: Tensor, return_last_only: bool = False) -> Tensor:
        """
        Forward pass.
        Args:
            input_ids: (B, L)
            return_last_only: if True, returns only the prediction for the last token (B, n_items + 1)
                            if False, returns for all tokens (B, L, n_items + 1)
        """
        batch_size, seq_len = input_ids.shape

        # Create causality mask (B, 1, L, L)
        device = input_ids.device
        mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).tril()
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

        seqs = self.item_emb(input_ids) * (self.d_model**0.5)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Add absolute positional embeddings
        seqs = seqs + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        # Pass through Transformer layers
        for layer in self.layers:
            seqs = layer(seqs, attention_mask=mask)

        seqs = self.final_layer_norm(seqs)

        # Output projection
        if return_last_only:
            seqs = seqs[:, -1, :]

        if self.tied_weights:
            logits = torch.matmul(seqs, self.item_emb.weight.transpose(0, 1))
        else:
            assert self.output_layer is not None
            logits = self.output_layer(seqs)

        return logits
