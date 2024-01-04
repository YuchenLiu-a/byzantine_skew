from __future__ import annotations

import torch
import torch.nn.functional as F

from typing import Tuple

@torch.no_grad()
def flatten_update(update: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, list[Tuple[str, torch.Size]]]:
    flat_param_updates = []
    structure = []
    for name, param_update in update.items():
        flat_param_updates.append(param_update.view(-1))
        structure.append((name, param_update.size()))
    flat_update = torch.cat(flat_param_updates)

    return flat_update, structure

@torch.no_grad()
def flatten_updates(updates: dict[str, dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, dict]:
    flat_updates = []
    client_idxs = []
    update_structure = []
    for client_idx in updates:
        client_idxs.append(client_idx)
        update = updates[client_idx]
        flat_update, update_structure = flatten_update(update)
        flat_updates.append(flat_update)
    if len(flat_updates) == 0:
        flat_updates = torch.Tensor()
    else:
        flat_updates = torch.stack(flat_updates)
    structure = {
        'client_idxs': client_idxs,
        'update_structure': update_structure,
    }
    return flat_updates, structure

@torch.no_grad()
def unflatten_update(flat_update: torch.Tensor, structure: dict) -> dict[str, torch.Tensor]:
    update_structure: list[Tuple[str, torch.Size]] = structure['update_structure']
    update = {}
    tensor_idx = 0
    for name, size in update_structure:
        num = size.numel()
        update[name] = flat_update[tensor_idx : tensor_idx + num].view(size)
        tensor_idx += num
    return update

@torch.no_grad()
def eval_ben_updates(benign_client_messages: dict[str, dict], eps: float=1e-7) -> dict:
    if len(benign_client_messages) == 0:
        return {}
    else:
        benign_updates = {idx: benign_client_messages[idx]['update'] for idx in benign_client_messages}
        flat_benign_updates, _ = flatten_updates(benign_updates)
        mean = flat_benign_updates.mean(dim=0)
        # dimension-wise
        
        # principle compoment
        q = len(flat_benign_updates)
        U, S, V = torch.pca_lowrank(flat_benign_updates, q=q, center=True)
        principle_coordinates = U[:, 0]
        pca_mean = principle_coordinates.mean().item()
        # pca_median = principle_coordinates.median().item()
        # pca_median_shift = pca_median - pca_mean
        pca_std = principle_coordinates.std(unbiased=False).item()
        pca_moment3 = (principle_coordinates - pca_mean).pow(3).mean().item()
        pca_fisher = pca_moment3 / (pca_std ** 3 + eps)

        pca_dir = V[:, 0]
        ipm_dir = -mean
        pca_vs_ipm = F.cosine_similarity(pca_dir, ipm_dir, dim=0).item()
        skew_pca_dir = pca_fisher * pca_dir
        skew_pca_vs_ipm = F.cosine_similarity(skew_pca_dir, ipm_dir, dim=0).item()
        lie_dir = flat_benign_updates.std(dim=0)
        pca_vs_lie = F.cosine_similarity(pca_dir, lie_dir, dim=0).item()
        

        ben_stats = {
            'principle_coordinates': principle_coordinates.tolist(),

            'benign_avg_norm': mean.norm().item(),
            'singular_values': S.tolist(),
            # 'pca_std': pca_std,

            'pca_vs_ipm': pca_vs_ipm, 
            'skew_pca_vs_ipm': skew_pca_vs_ipm, 
            'pca_vs_lie': pca_vs_lie, 

            'pca_fisher': pca_fisher,
            # 'pca_median_shift': pca_median_shift,
        }

        return ben_stats

def eval_agg_dev(agg_update: dict[str, torch.Tensor], ben_client_msgs: dict[str, dict]):
    if len(ben_client_msgs) == 0:
        return {}
    else:
        flat_agg_update, _ = flatten_update(agg_update)

        benign_updates = {idx: ben_client_msgs[idx]['update'] for idx in ben_client_msgs}
        flat_benign_updates, _ = flatten_updates(benign_updates)
        flat_benign_avg = flat_benign_updates.mean(dim=0)

        renamed_dev_stats = {}
        dev_stats = eval_dev(flat_agg_update, flat_benign_avg)
        for key, val in dev_stats.items():
            renamed_dev_stats['agg_'+key] = val
        
        return renamed_dev_stats

def eval_byz_dev(byz_client_msgs: dict[str, dict], ben_client_msgs: dict[str, dict]):
    if len(byz_client_msgs) == 0 or len(ben_client_msgs) == 0:
        return {}
    else:
        # flat_byz_avg
        byz_updates = {idx: msg['update'] for idx, msg in byz_client_msgs.items()}
        flat_byz_updates, _ = flatten_updates(byz_updates)
        flat_byz_avg = flat_byz_updates.mean(dim=0)
        # flat_ben_avg
        ben_updates = {idx: msg['update'] for idx, msg in ben_client_msgs.items()}
        flat_ben_updates, _ = flatten_updates(ben_updates)
        flat_ben_avg = flat_ben_updates.mean(dim=0)

        renamed_dev_stats = {}
        dev_stats = eval_dev(flat_byz_avg, flat_ben_avg)
        for key, val in dev_stats.items():
            renamed_dev_stats['byz_'+key] = val
        
        return renamed_dev_stats

def eval_dev(target_update: torch.Tensor, ref_update: torch.Tensor, eps: float=1e-7):
    dev = target_update - ref_update

    dev_norm: float = dev.norm().item()
    ref_norm: float = ref_update.norm().item()
    cosine = dev.dot(ref_update).item() / (dev_norm + eps) / (ref_norm + eps)

    dev_stats = {
        'dev_cos': cosine,
        'dev_norm': dev_norm,
        # 'ref_norm': ref_norm,
    }

    return dev_stats