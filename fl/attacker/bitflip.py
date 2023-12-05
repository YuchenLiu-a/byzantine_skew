from __future__ import annotations

import torch
from typing import Tuple
from .attacker import Attacker

class BitFlip(Attacker):
    def attack(self, sampled_byz_client_idxs, server_message: dict, knowledge: dict) -> Tuple[dict[str, dict], dict]:
        client_msgs = {}
        for client_idx in sampled_byz_client_idxs:
            client_msg = self.byz_clients[client_idx].local_update(server_message)
            
            update: dict[str, torch.Tensor] = client_msg['update']
            byz_update = {}
            for name in update:
                byz_update[name] = -update[name]
            client_msg['update'] = byz_update
            client_msgs[client_idx] = client_msg

        verbose_log = {}

        return client_msgs, verbose_log
