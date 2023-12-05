from __future__ import annotations

import torch
from .. import utils

from typing import Tuple
from .attacker import Attacker

class Skew(Attacker):
    def attack(self, sampled_byz_client_idxs: set[str], server_message: dict, knowledge: dict) -> Tuple[dict[str, dict], dict]:
        verbose_log = {}

        if len(knowledge['known_benign_client_idxs']) == 0:
            # use gradients of byz clients as reference
            ref_updates = self.get_ref_updates(sampled_byz_client_idxs, server_message, knowledge)
        else:
            # use gradients of honest clients as reference
            ref_updates = {idx: knowledge['benign_client_messages'][idx]['update'] for idx in knowledge['known_benign_client_idxs']}
        flat_ref_updates, structure = utils.flatten_updates(ref_updates)

        flat_skew_dir = self.init_dev(flat_ref_updates)
        n_ben = len(knowledge['benign_client_messages'])
        n_byz = len(sampled_byz_client_idxs)
        if n_ben <= n_byz:
            flat_avg = flat_ref_updates.mean(dim=0)
            if n_ben == 1:
                flat_byz_update = -10 * flat_avg
            else:
                flat_byz_update = flat_avg + flat_skew_dir * 10
        else:
            n_skew = int((n_ben - n_byz) / n_ben * len(flat_ref_updates))
            flat_avg = flat_ref_updates.mean(dim=0)
            inner_product = flat_ref_updates @ flat_skew_dir
            _, skew_idxs = inner_product.topk(k=n_skew, sorted=False)
            flat_skew_updates = flat_ref_updates[skew_idxs]
            verbose_log['skew_idxs'] = skew_idxs.tolist()
            
            flat_skew_avg = flat_skew_updates.mean(dim=0)
            flat_dev = (flat_skew_avg - flat_avg).sign() * flat_skew_updates.std(dim=0, unbiased=False)
            skew_diameter = torch.cdist(flat_skew_updates, flat_skew_updates).max().item()
            def f(s: float):
                flat_byz_update = flat_skew_avg + s * flat_dev
                dists: torch.Tensor = (flat_byz_update - flat_skew_updates).norm(dim=-1)
                max_dist = dists.max().item()
                return max_dist - skew_diameter
            max_s = 10.0
            s = utils.bisection(0.0, max_s, 1e-5, f)
            strength = self.args.skew_lambda * s
            flat_byz_update = flat_skew_avg + strength * flat_dev
            verbose_log['strength'] = strength

        byz_update = utils.unflatten_update(flat_byz_update, structure)
        sampled_byz_client_msgs = self.pack_byz_client_msgs(sampled_byz_client_idxs, byz_update)

        return sampled_byz_client_msgs, verbose_log

    def init_dev(self, flat_ref_updates: torch.Tensor):
        flat_avg = flat_ref_updates.mean(dim=0)
        flat_med, _ = flat_ref_updates.median(dim=0)
        flat_dir = flat_med - flat_avg
        return flat_dir