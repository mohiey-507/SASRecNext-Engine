from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from engine.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from engine.utils import get_logger

if TYPE_CHECKING:
    import torch.nn as nn
    from torch import Tensor
    from torch.utils.data import DataLoader

logger = get_logger(__name__)

_METRIC_FNS = {
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "mrr": mrr_at_k,
}


class Evaluator:
    """Full-ranking evaluator: scores every item, masks history, computes metrics."""

    def __init__(
        self,
        model: nn.Module,
        eval_loader: DataLoader[dict[str, Tensor]],
        metrics: tuple[str, ...],
        top_k: tuple[int, ...],
        device: torch.device,
        mode: str,
    ) -> None:
        self._model = model
        self._eval_loader = eval_loader
        self._metrics = metrics
        self._top_k = top_k
        self._device = device
        self._mode = mode

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        """Run full-ranking evaluation over the entire eval set."""
        self._model.eval()

        all_topk_indices: dict[str, list[Tensor]] = {}
        all_targets: list[Tensor] = []
        max_k = max(self._top_k)

        for batch in self._eval_loader:
            scores_dict, targets = self._score_batch(batch)
            for mode, scores in scores_dict.items():
                if mode not in all_topk_indices:
                    all_topk_indices[mode] = []
                _, top_k = scores.topk(max_k, dim=-1)
                all_topk_indices[mode].append(top_k.cpu())
            all_targets.append(targets.cpu())

        final_targets = torch.cat(all_targets)
        results: dict[str, float] = {}

        for mode, topk_list in all_topk_indices.items():
            mode_topk = torch.cat(topk_list)
            mode_results = self._aggregate_metrics(mode_topk, final_targets)

            prefix = f"{mode}_" if self._mode == "both" else ""
            for k, v in mode_results.items():
                results[f"{prefix}{k}"] = v

        return results

    def _score_batch(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], Tensor]:
        """Compute masked scores for a single batch."""
        input_ids = batch["input_ids"].to(self._device)
        target_ids = batch["target_id"].to(self._device)
        history_mask = batch["history_mask"].to(self._device)

        base_scores = self._model(input_ids, return_last_only=True)  # (B, n_items + 1)

        scores_dict: dict[str, Tensor] = {}

        if self._mode in ("both", "full"):
            scores_full = base_scores.clone()
            scores_full[:, 0] = float("-inf")
            scores_full.masked_fill_(history_mask, float("-inf"))
            scores_dict["full"] = scores_full

        if self._mode in ("uni100", "both"):
            scores_uni100 = base_scores.clone()
            negatives = batch["negatives"].to(self._device)
            uni100_mask = torch.full_like(scores_uni100, float("-inf"))

            batch_idx = torch.arange(base_scores.shape[0], device=self._device)
            uni100_mask[batch_idx, target_ids] = 0.0
            uni100_mask.scatter_(1, negatives, 0.0)

            scores_dict["uni100"] = scores_uni100 + uni100_mask

        return scores_dict, target_ids

    def _aggregate_metrics(self, top_k_indices: Tensor, targets: Tensor) -> dict[str, float]:
        """Compute all configured metrics at all configured K values."""
        results: dict[str, float] = {}
        for metric_name in self._metrics:
            fn = _METRIC_FNS[metric_name]
            for k in self._top_k:
                results[f"{metric_name}@{k}"] = fn(top_k_indices[:, :k], targets)

        return results
