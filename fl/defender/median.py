from __future__ import annotations
import torch

from typing import Tuple
from .aggregator import Aggregator
from ..utils import flatten_updates, unflatten_update

class Median(Aggregator):
    @torch.no_grad()
    def __call__(self, client_messages: dict, knowledge: dict) -> Tuple[dict[str, torch.Tensor], dict]:
        updates = Aggregator.get_updates(client_messages)
        flat_updates, structure = flatten_updates(updates)
        
        flat_agg_update, _ = flat_updates.median(dim=0)
        
        agg_update = unflatten_update(flat_agg_update, structure)

        verbose_log = {}

        return agg_update, verbose_log