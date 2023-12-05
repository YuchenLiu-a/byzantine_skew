from __future__ import annotations

import torch

from typing import Tuple
from ..utils import flatten_updates, unflatten_update
from .aggregator import Aggregator

class Rfa(Aggregator):
    def __call__(self, client_messages: dict, knowledge: dict) -> Tuple[dict[str, torch.Tensor], dict]:
        updates = Aggregator.get_updates(client_messages)
        flat_updates, structure = flatten_updates(updates)

        # initialize weight and aggregated update
        n_updates = len(updates)
        weights = torch.ones(n_updates) / n_updates
        flat_agg_update = weights.matmul(flat_updates)
        # update weight and aggregated update
        for _ in range(self.args.rfa_budget):
            distances: torch.Tensor = (flat_updates - flat_agg_update).norm(dim=-1)
            weights = 1 / distances.clamp_min(self.args.rfa_eps)
            weights = weights / weights.sum()
            flat_agg_update = weights.matmul(flat_updates)

        agg_update = unflatten_update(flat_agg_update, structure)

        client_weights = {idx: weight.item() for idx, weight in zip(structure['client_idxs'], weights)}
        verbose_log = {'client_weights': client_weights}

        return agg_update, verbose_log
