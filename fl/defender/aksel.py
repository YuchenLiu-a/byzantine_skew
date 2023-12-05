from __future__ import annotations
import torch

from typing import Tuple
from .aggregator import Aggregator
from ..utils import flatten_updates, unflatten_update

class Aksel(Aggregator):
    @torch.no_grad()
    def __call__(self, client_messages: dict, knowledge: dict) -> Tuple[dict[str, torch.Tensor], dict]:
        updates = Aggregator.get_updates(client_messages)
        flat_updates, structure = flatten_updates(updates)
        
        flat_median, _ = flat_updates.median(dim=0)
        s = (flat_updates - flat_median).square().sum(dim=-1)
        r = s.median()
        bool_idxs = (s <= r)
        flat_agg_update = flat_updates[bool_idxs].mean(dim=0)
        agg_update = unflatten_update(flat_agg_update, structure)

        candidate_idxs_lt = bool_idxs.nonzero().flatten().tolist()
        cand_info = Aggregator.get_cand_info(structure, candidate_idxs_lt)
        verbose_log = {'cand_info': cand_info}

        return agg_update, verbose_log