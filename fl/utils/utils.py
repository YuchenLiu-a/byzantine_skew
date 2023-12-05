from __future__ import annotations

import numpy.random
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.cuda as cuda
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from typing import Tuple

def setup_seed(seed):
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)
    cuda.manual_seed(seed)
    numpy.random.seed(seed)
    cudnn.deterministic = True

def print_dict(d: dict, title: str):
    if len(d) == 0:
        return
    else:
        print(f'==================== {title} ====================')
        for key, val in d.items():
            print(f'{key}: {val}')

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(best_te_acc=0)
@torch.no_grad()
def eval_perf(model: nn.Module, te_dataloader: DataLoader, loss_fun):
    total_loss = 0.0
    total_correct = 0

    model.cuda()
    model.eval()
    for inputs, targets in te_dataloader:
        inputs: torch.Tensor = inputs.cuda()
        targets: torch.Tensor = targets.cuda()
        outputs: torch.Tensor = model(inputs)
        total_loss += loss_fun(outputs, targets).item()
        predictions = torch.argmax(outputs, dim=1)
        total_correct += (predictions == targets).long().sum().item()
    model.cpu()
    
    n_data = len(te_dataloader.dataset)  # type: ignore
    te_loss = total_loss / n_data
    te_acc = total_correct / n_data * 100.0

    eval_perf.best_te_acc = max(eval_perf.best_te_acc, te_acc)
    perf_stats = {
        'te_loss': te_loss,
        'te_acc': te_acc,
        'best_te_acc': eval_perf.best_te_acc,
    }

    return perf_stats

def get_stats() -> list[Tuple[int, int]]:
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    # total memory, used memory
    stats = [tuple(int(mem) for mem in device_info.split(',')) for device_info in devices_info]
    return stats

def block(devices: str):
    stats = get_stats()
    visible_devices = [int(device) for device in devices.split(',')]
    n_devices = len(visible_devices)
    for visible_id, device in zip(range(n_devices), visible_devices):
        total, used = stats[device]
        block_pct = random.uniform(0.9, 0.95)
        block = int(total * block_pct) - used
        if block > 0:
            x = torch.FloatTensor(256, 1024, block).to(visible_id)  # type: ignore

def bisection(min_val: float, max_val: float, tol: float, f):
    assert min_val < max_val
    # assert f(min_val) * f(max_val) <= 0
    while max_val - min_val > tol:
        mid_val = (min_val + max_val) / 2
        if f(mid_val) * f(min_val) > 0:
            min_val = mid_val
        else:
            max_val = mid_val
    return min_val

def line_maximize(scape, evals=16, start=0., delta=1., ratio=0.8, tol=1e-5):
    """ Best-effort arg-maximize a scape: ℝ⁺⟶ ℝ, by mere exploration.
    Args:
        scape Function to best-effort arg-maximize
        evals Maximum number of evaluations, must be a positive integer
        start Initial x evaluated, must be a non-negative float
        delta Initial step delta, must be a positive float
        ratio Contraction ratio, must be between 0.5 and 1. (both excluded)
    Returns:
        Best-effort maximizer x under the evaluation budget
    """
    # Variable setup
    best_x = start
    best_y = scape(best_x)
    evals -= 1
    # Expansion phase
    while evals > 0:
        prop_x = best_x + delta
        prop_y = scape(prop_x)
        evals -= 1
        # Check if best
        if prop_y > best_y + tol:
            best_y = prop_y
            best_x = prop_x
            delta *= 2
        else:
            delta *= ratio
            break
    # Contraction phase
    while evals > 0:
        if prop_x < best_x:    #type: ignore
            prop_x += delta    #type: ignore
        else:
            x = prop_x - delta    #type: ignore
            while x < 0:
                x = (x + prop_x) / 2    #type: ignore
            prop_x = x
        prop_y = scape(prop_x)
        evals -= 1
        # Check if best
        if prop_y > best_y + tol:
            best_y = prop_y
            best_x = prop_x
        # Reduce delta
        delta *= ratio
    # Return found maximizer
    return best_x
