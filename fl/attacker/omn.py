from __future__ import annotations

import torch

from typing import Tuple
from ..defender import Defender
from ..utils import flatten_update, flatten_updates, line_maximize, unflatten_update
from .attacker import Attacker

class Omn(Attacker):
    @torch.no_grad()
    def attack(self, sampled_byz_client_idxs: set[str], server_message: dict, knowledge: dict) -> Tuple[dict[str, dict], dict]:
        ref_updates = self.get_ref_updates(sampled_byz_client_idxs, server_message, knowledge)
        flat_ref_updates, structure = flatten_updates(ref_updates)

        flat_avg_update = flat_ref_updates.mean(dim=0)
        flat_dev = Omn.get_flat_deviation(self.args.omn_deviation, flat_ref_updates)

        sampled_benign_client_messages: dict[str, dict] = knowledge['benign_client_messages']
        defender: Defender = knowledge['defender']
        defender_knowledge = knowledge['defender_knowledge']
        def eval_epsilon(epsilon: float):
            flat_byz_update = flat_avg_update - epsilon * flat_dev
            byz_update = unflatten_update(flat_byz_update, structure)
            sampled_byz_client_messages = self.pack_byz_client_msgs(sampled_byz_client_idxs, byz_update)
            sampled_client_messages = {**sampled_benign_client_messages, **sampled_byz_client_messages}

            agg_update, _ = defender.defend(sampled_client_messages, defender_knowledge)
            flat_agg_update, _ = flatten_update(agg_update)

            dev_norm: float = (flat_agg_update - flat_avg_update).norm().item()
            return dev_norm
        
        epsilon = line_maximize(scape=eval_epsilon, evals=self.args.omn_evals)

        flat_byz_update = flat_avg_update - epsilon * flat_dev
        byz_update = unflatten_update(flat_byz_update, structure)
        sampled_byz_client_msgs = self.pack_byz_client_msgs(sampled_byz_client_idxs, byz_update)

        verbose_log = {'omn_epsilon': epsilon}

        return sampled_byz_client_msgs, verbose_log