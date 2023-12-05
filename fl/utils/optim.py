from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import SGD
from typing import Iterator

def get_optimizer(optimizer: str, parameters: Iterator[Parameter], lr: float, momentum: float, weight_decay: float):
    if optimizer == 'sgd':
        return SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise Exception('invalid optimizer')

@torch.no_grad()
def step(update: dict[str, torch.Tensor], global_model: nn.Module) -> None:
    model_state: dict[str, torch.Tensor] = global_model.state_dict()
    for name in model_state:
        state_type = model_state[name].dtype
        model_state[name] += update[name].to(dtype=state_type)
    global_model.load_state_dict(model_state)
