from __future__ import annotations

import torch

from typing import Tuple
from ..utils import flatten_updates, unflatten_update
from .attacker import Attacker

class Ipm(Attacker):
    @torch.no_grad()
    def attack(self, sampled_byz_client_idxs: set[str], server_message: dict, knowledge: dict) -> Tuple[dict[str, dict], dict]:
        ref_updates = self.get_ref_updates(sampled_byz_client_idxs, server_message, knowledge)
        flat_ref_updates, structure = flatten_updates(ref_updates)

        flat_avg_update = flat_ref_updates.mean(dim=0)

        flat_byz_update = - self.args.ipm_epsilon * flat_avg_update
        byz_update = unflatten_update(flat_byz_update, structure)
        sampled_byz_client_msgs = self.pack_byz_client_msgs(sampled_byz_client_idxs, byz_update)

        verbose_log = {}

        return sampled_byz_client_msgs, verbose_log