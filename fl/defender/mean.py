from __future__ import annotations
import torch

from .aggregator import Aggregator
from argparse import Namespace
from typing import Tuple
from ..utils import flatten_updates, unflatten_update

class Mean(Aggregator):
    @torch.no_grad()
    def __call__(self, client_messages: dict, knowledge: dict) -> Tuple[dict[str, torch.Tensor], dict]:
        n_datas: list[int] = [client_messages[client_idx]['n_data'] for client_idx in client_messages]
        updates = Aggregator.get_updates(client_messages)

        weight = torch.Tensor(n_datas)
        weight /= weight.sum()

        flat_updates, structure = flatten_updates(updates)
        flat_agg_update = weight.matmul(flat_updates)
        agg_update = unflatten_update(flat_agg_update, structure)

        verbose_log = {}
        
        return agg_update, verbose_log