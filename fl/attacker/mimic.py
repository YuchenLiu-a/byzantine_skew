from __future__ import annotations

import torch

from argparse import Namespace
from typing import Tuple
from ..client import Client
from ..utils import flatten_updates, unflatten_update
from .attacker import Attacker

class Mimic(Attacker):
    def __init__(self, byz_clients: dict[str, Client], args: Namespace) -> None:
        super().__init__(byz_clients, args)
        self.t = 0
    
    @torch.no_grad()
    def attack(self, sampled_byz_client_idxs: set[str], server_message: dict, knowledge: dict) -> Tuple[dict[str, dict], dict]:
        ref_updates = self.get_ref_updates(sampled_byz_client_idxs, server_message, knowledge)
        flat_ref_updates, structure = flatten_updates(ref_updates)
        # compute z
        if self.t == 0:
            self.init_z(flat_ref_updates)
        else:
            self.update_z(flat_ref_updates)
        # find target
        scores = flat_ref_updates.matmul(self.z).abs()
        target_idx = scores.argmax()
        flat_byz_update = flat_ref_updates[target_idx]
        byz_update = unflatten_update(flat_byz_update, structure)
        
        sampled_byz_client_msgs = self.pack_byz_client_msgs(sampled_byz_client_idxs, byz_update)
        # update t
        self.t = self.t + 1

        target_client_idx = structure['client_idxs'][target_idx]

        verbose_log = {'mimic_target_client_idx': target_client_idx}

        return sampled_byz_client_msgs, verbose_log

    def init_z(self, flat_ref_updates: torch.Tensor):
        flat_avg = flat_ref_updates.mean(dim=0)

        z = torch.randn_like(flat_avg)
        weights = (flat_ref_updates - flat_avg).matmul(z)
        z = weights.matmul(flat_ref_updates)

        # initialize z and mu
        self.mu = flat_avg
        self.z: torch.Tensor = z / z.norm()
    
    def update_z(self, flat_ref_updates: torch.Tensor):
        # udpate mu
        flat_avg = flat_ref_updates.mean(dim=0)
        self.mu = self.t / (self.t + 1) * self.mu + 1 / (self.t + 1) * flat_avg
        # udpate z
        weights = (flat_ref_updates - flat_avg).matmul(self.z)
        z_update = weights.matmul(flat_ref_updates)
        z_update: torch.Tensor = z_update / z_update.norm()

        z_new = self.t / (self.t + 1) * self.z + 1 / (self.t + 1) * z_update
        self.z: torch.Tensor = z_new / z_new.norm()