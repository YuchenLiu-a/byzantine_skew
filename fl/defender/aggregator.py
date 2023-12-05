from __future__ import annotations

import itertools
import torch

from argparse import Namespace
from typing import Iterable, Tuple

class Aggregator:
    byz_client_idxs = set()

    def __init__(self, args: Namespace, byz_client_idxs: set[str]) -> None:
        self.args = args
        Aggregator.byz_client_idxs = byz_client_idxs

    @torch.no_grad()
    def __call__(self, client_messages: dict, knowledge: dict) -> Tuple[dict[str, torch.Tensor], dict]:
        raise Exception('specification of aggregator is needed')

    @staticmethod
    def get_updates(client_messages: dict) -> dict[str, dict[str, torch.Tensor]]:
        updates: dict[str, dict[str, torch.Tensor]] = {client_idx: client_messages[client_idx]['update'] for client_idx in client_messages}
        return updates
    
    @staticmethod
    def get_cand_info(structure: dict, candidate_idxs: Iterable[int]) -> dict:
        sampled_client_idxs = structure['client_idxs']
        cand_client_idxs_lt = list(itertools.chain.from_iterable(sampled_client_idxs[idx].split('_') for idx in candidate_idxs))
        n_cand_byz_clients = len(Aggregator.byz_client_idxs.intersection(cand_client_idxs_lt))
        n_cand_ben_clients = len(cand_client_idxs_lt) - n_cand_byz_clients
        cand_info = {
            'cand_client_idxs_lt': cand_client_idxs_lt,
            'n_cand_byz_clients': n_cand_byz_clients,
            'n_cand_ben_clients': n_cand_ben_clients,
        }
        return cand_info