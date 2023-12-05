from __future__ import annotations

import numpy as np
import numpy.random as random
import torch

from math import ceil
from typing import Tuple
from ..utils import flatten_updates, unflatten_update
from .aggregator import Aggregator

class Dnc(Aggregator):
    def __call__(self, client_messages: dict, knowledge: dict) -> Tuple[dict[str, torch.Tensor], dict]:
        updates = Aggregator.get_updates(client_messages)
        m = knowledge['n_byz_updates']
        flat_updates, structure = flatten_updates(updates)
        
        n_updates = len(client_messages)
        cand_idxs_set = {i for i in range(n_updates)}
        n_filtered_updates = ceil(self.args.dnc_c * m)
        for _ in range(self.args.dnc_n_iters):
            d = flat_updates.size(-1)
            r = random.choice(d, self.args.dnc_b)
            sub_updates = flat_updates[:, r]
            sub_avg = sub_updates.mean(dim=0)
            sub_updates_c = sub_updates - sub_avg
            _, _, V = torch.svd(sub_updates_c)
            v: torch.Tensor = V[:, 0]
            score = sub_updates_c.matmul(v).square()
            _, filtered_idxs = score.topk(k=n_filtered_updates)
            filtered_idxs = set(filtered_idxs.tolist())
            cand_idxs_set = cand_idxs_set.difference(filtered_idxs)
        cand_idx_arr = np.array(list(cand_idxs_set))
        flat_agg_update = flat_updates[cand_idx_arr].mean(dim=0)
        agg_update = unflatten_update(flat_agg_update, structure)

        cand_info = Aggregator.get_cand_info(structure, cand_idxs_set)
        verbose_log = {'cand_info': cand_info}

        return agg_update, verbose_log