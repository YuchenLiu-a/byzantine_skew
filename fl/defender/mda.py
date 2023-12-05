from __future__ import annotations

import numpy as np
import torch

from itertools import combinations
from typing import Tuple
from ..utils import flatten_updates, unflatten_update
from .aggregator import Aggregator

class Mda(Aggregator):
    def __call__(self, client_messages: dict, knowledge: dict) -> Tuple[dict[str, torch.Tensor], dict]:
        updates = Aggregator.get_updates(client_messages)
        flat_updates, structure = flatten_updates(updates)

        dists = torch.cdist(flat_updates, flat_updates)
        n_updates = len(updates)
        n_byz_updates = min(knowledge['n_byz_updates'], n_updates // 2)

        min_d = dists.max().item()
        cand_idxs_lt = [idx for idx in range(n_updates)]
        for idxs in combinations(range(n_updates), n_updates - n_byz_updates):
            idxs_np = np.array(idxs)
            d = dists[idxs_np][:, idxs_np].max().item()
            if min_d > d:
                min_d = d
                cand_idxs_lt = idxs

        cand_idxs_arr = np.array(cand_idxs_lt)
        flat_agg_update = flat_updates[cand_idxs_arr].mean(dim=0)
        agg_update = unflatten_update(flat_agg_update, structure)

        cand_info = Aggregator.get_cand_info(structure, cand_idxs_lt)
        verbose_log = {'cand_info': cand_info}

        return agg_update, verbose_log