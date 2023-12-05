from __future__ import annotations

import torch
from .. import utils

def nnm(client_msgs: dict[str, dict], knowledge: dict):
    # flatten
    n_clients = len(client_msgs)
    n_byz = knowledge['n_byz_updates']
    updates = {idx: client_msg['update'] for idx, client_msg in client_msgs.items()}
    flat_updates, structure = utils.flatten_updates(updates)
    # compute distance
    dists = torch.cdist(flat_updates, flat_updates)
    # find nearest neighbors
    nearest_neighbor_idxs = dists.topk(k=n_clients - n_byz + 1, largest=False, sorted=True)[1][..., 1:]
    # mix
    mixed_cl_messages = {}
    for idx, nearest_neighbor_idxs_per_cl in zip(client_msgs, nearest_neighbor_idxs):
        flat_mixed_update = flat_updates[nearest_neighbor_idxs_per_cl].mean(dim=0)
        mixed_update = utils.unflatten_update(flat_mixed_update, structure)
        mix_cl_message = {
                'update': mixed_update,
                'n_data': client_msgs[idx]['n_data'],
        }
        mixed_cl_messages[idx] = mix_cl_message
    return mixed_cl_messages, knowledge