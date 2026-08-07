from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def recall_at_k(top_k_indices: Tensor, targets: Tensor) -> float:
    """Recall@K: fraction of users whose target appears in their top-K."""
    hits = (top_k_indices == targets.unsqueeze(-1)).any(dim=-1)
    return hits.float().mean().item()


def ndcg_at_k(top_k_indices: Tensor, targets: Tensor) -> float:
    """NDCG@K for single-target ranking (IDCG = 1.0)."""
    hits = top_k_indices == targets.unsqueeze(-1)  # (B, K)
    has_hit = hits.any(dim=-1)  # (B,)
    # Position of the hit (0-indexed); argmax returns 0 for all-False rows
    positions = hits.float().argmax(dim=-1)  # (B,)
    # DCG = 1 / log2(rank + 1), where rank is 1-indexed → log2(position + 2)
    dcg = has_hit.float() / torch.log2(positions.float() + 2)
    return dcg.mean().item()


def mrr_at_k(top_k_indices: Tensor, targets: Tensor) -> float:
    """Mean Reciprocal Rank @K."""
    hits = top_k_indices == targets.unsqueeze(-1)  # (B, K)
    has_hit = hits.any(dim=-1)  # (B,)
    positions = hits.float().argmax(dim=-1)  # (B,)
    # RR = 1 / (position + 1), position is 0-indexed
    rr = has_hit.float() / (positions.float() + 1)
    return rr.mean().item()
