from __future__ import annotations

import torch

from typing import Tuple
from ..utils import flatten_updates, unflatten_update
from .attacker import Attacker

class MinSum(Attacker):
    @torch.no_grad()
    def attack(self, sampled_byz_client_idxs: set[str], server_message: dict, knowledge: dict) -> Tuple[dict[str, dict], dict]:
        ref_updates = self.get_ref_updates(sampled_byz_client_idxs, server_message, knowledge)
        flat_ref_updates, structure = flatten_updates(ref_updates)
        deviation = MinSum.get_flat_deviation(self.args.minsum_deviation, flat_ref_updates)

        pairwise_squared_dist = torch.cdist(flat_ref_updates, flat_ref_updates).square()
        dist_sum = pairwise_squared_dist.sum(dim=1)
        max_dist_sum = dist_sum.max()

        lamda = 10.0
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0

        flat_avg = flat_ref_updates.mean(dim=0)
        
        while abs(lamda_succ - lamda) > threshold_diff:
            flat_byz_update = flat_avg - lamda * deviation
            squared_dists: torch.Tensor = torch.norm(flat_byz_update - flat_ref_updates, dim=1).square()
            score = squared_dists.sum()
            
            if score <= max_dist_sum:
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2
        flat_byz_update = flat_avg - lamda * deviation
        byz_update = unflatten_update(flat_byz_update, structure)
        sampled_byz_client_msgs = self.pack_byz_client_msgs(sampled_byz_client_idxs, byz_update)

        verbose_log = {'minsum_lamda': lamda}

        return sampled_byz_client_msgs, verbose_log
