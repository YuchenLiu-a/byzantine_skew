from __future__ import annotations

import torch

from typing import Tuple
from ..utils import flatten_updates, unflatten_update
from .aggregator import Aggregator

class Krum(Aggregator):
    def __call__(self, client_messages: dict, knowledge: dict) -> Tuple[dict[str, torch.Tensor], dict]:
        updates = Aggregator.get_updates(client_messages)
        flat_updates, structure = flatten_updates(updates)

        squared_dists = torch.cdist(flat_updates, flat_updates).square()
        n_updates = len(updates)
        n_byz_updates = min(knowledge['n_byz_updates'], (n_updates - 3) // 2)
        topk_dists, _ = squared_dists.topk(k=n_updates - n_byz_updates -1, dim=-1, largest=False, sorted=False)
        scores = topk_dists.sum(dim=-1)

        cand_idxs_tensor: torch.Tensor = scores.topk(k=n_updates - n_byz_updates, dim=-1, largest=False)[1]

        cand_idxs_arr = cand_idxs_tensor.cpu().numpy()
        flat_agg_update = flat_updates[cand_idxs_tensor].mean(dim=0)
        agg_update = unflatten_update(flat_agg_update, structure)

        cand_idxs_lt = cand_idxs_arr.tolist()
        cand_info = Aggregator.get_cand_info(structure, cand_idxs_lt)
        verbose_log = {'cand_info': cand_info}

        return agg_update, verbose_log