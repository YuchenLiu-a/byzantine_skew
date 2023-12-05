from __future__ import annotations

import torch

from argparse import Namespace
from typing import Tuple
from ..client import Client

class Attacker:
    def __init__(self, byz_clients: dict[str, Client], args: Namespace) -> None:
        self.args = args
        self.byz_clients = byz_clients
    
    def attack(self, sampled_byz_client_idxs: set[str], server_message: dict, knowledge: dict) -> Tuple[dict[str, dict], dict]:
        raise Exception('instantiate attack')

    @torch.no_grad()
    def get_ref_updates(self, sampled_byz_client_idxs: set[str], server_message: dict, knowledge: dict) -> dict[str, dict[str, torch.Tensor]]:
        sampled_benign_client_messages: dict[str, dict] = knowledge['benign_client_messages']
        if len(sampled_benign_client_messages) > 0:
            ref_updates = {idx: sampled_benign_client_messages[idx]['update'] for idx in sampled_benign_client_messages}
        else:
            sampled_byz_client_ori_messages = {idx: self.byz_clients[idx].local_update(server_message) for idx in sampled_byz_client_idxs}
            ref_updates = {idx: sampled_byz_client_ori_messages[idx]['update'] for idx in sampled_byz_client_ori_messages}
        return ref_updates
    
    @torch.no_grad()
    def pack_byz_client_msgs(self, sampled_byz_client_idxs: set[str], byz_update: dict[str, torch.Tensor]) -> dict[str, dict]:
        byz_client_messages = {}
        for idx in sampled_byz_client_idxs:
            byz_client_message = self.byz_clients[idx].pack_other_message()
            byz_client_message['update'] = byz_update
            byz_client_messages[idx] = byz_client_message
        return byz_client_messages
    
    @staticmethod
    @torch.no_grad()
    def get_flat_deviation(dev_type: str, flat_ref_updates: torch.Tensor) -> torch.Tensor:
        flat_avg = flat_ref_updates.mean(dim=0)
        flat_median: torch.Tensor = flat_ref_updates.median(dim=0)[0]
        if dev_type == 'unit_vec':
            deviation: torch.Tensor = flat_avg / flat_avg.norm()
        elif dev_type == 'sign':
            deviation = flat_avg.sign()
        elif dev_type == 'skew_mode':
            deviation = 3 * (flat_avg - flat_median)
        elif dev_type == 'skew_std':
            sign = (flat_avg - flat_median).sign()
            deviation = sign * flat_ref_updates.std(dim=0, unbiased=False)
        elif dev_type == 'neg_skew_mode':
            deviation = - 3 * (flat_avg - flat_median)
        elif dev_type == 'neg_skew_std':
            sign = (flat_avg - flat_median).sign()
            deviation = - sign * flat_ref_updates.std(dim=0, unbiased=False)
        elif dev_type == 'neg_std':
            deviation = - flat_ref_updates.std(dim=0, unbiased=False)
        elif dev_type == 'ran_std':
            dev_scale = flat_ref_updates.std(dim=0, unbiased=False)
            dev_direction = 2 * torch.randint_like(dev_scale, high=2) - 1
            deviation = dev_scale * dev_direction
        elif dev_type == 'std':
            deviation = flat_ref_updates.std(dim=0, unbiased=False)
        else:
            raise Exception('invalid deviation type.')

        return deviation
