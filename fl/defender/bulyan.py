from __future__ import annotations

import numpy as np
import torch

from typing import Tuple
from ..utils import flatten_updates, unflatten_update
from .aggregator import Aggregator

class Bulyan(Aggregator):
    def __call__(self, client_messages: dict, knowledge: dict) -> Tuple[dict[str, torch.Tensor], dict]:
        updates = Aggregator.get_updates(client_messages)
        flat_updates, structure = flatten_updates(updates)

        squared_dists = torch.cdist(flat_updates, flat_updates)
        n_updates = len(updates)
        n_byz_updates = min(knowledge['n_byz_updates'], (n_updates - 3) // 4)
        topk_dists, _ = squared_dists.topk(k=n_updates - n_byz_updates -1, dim=-1, largest=False, sorted=False)
        scores = topk_dists.sum(dim=-1)

        _, krum_cand_idxs = scores.topk(k=n_updates - 2 * n_byz_updates, dim=0, largest=False)
        krum_cand_idxs_arr = krum_cand_idxs.cpu().numpy()

        flat_krum_updates = flat_updates[krum_cand_idxs_arr]
        med, _ = flat_krum_updates.median(dim=0)
        dist = (flat_krum_updates - med).abs()
        tr_updates, _ = dist.topk(k=n_updates - 4 * n_byz_updates, dim=0, largest=False)
        flat_agg_update = tr_updates.mean(dim=0)
        agg_update = unflatten_update(flat_agg_update, structure)

        krum_cand_idxs_lt = krum_cand_idxs_arr.tolist()
        krum_cand_info = Aggregator.get_cand_info(structure, krum_cand_idxs_lt)
        verbose_log = {'krum_cand_info': krum_cand_info}

        return agg_update, verbose_log