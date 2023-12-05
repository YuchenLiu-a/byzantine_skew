from __future__ import annotations

import math
import random
from .. import utils

def bucket(client_msgs: dict[str, dict], s: int, knowledge: dict):
    if s == 0:
        return client_msgs, knowledge
    else:
        client_idxs = list(client_msgs.keys())
        random.shuffle(client_idxs)

        bucket_client_msgs = {}
        n_clients = len(client_msgs)
        n_buckets = math.ceil(n_clients / s)
        for bucket_idx in range(n_buckets):
            bucket_client_idxs = client_idxs[bucket_idx * s : (bucket_idx + 1) * s]
            bucket_id = '_'.join(bucket_client_idxs)
            # update
            updates = {idx: client_msgs[idx]['update'] for idx in bucket_client_idxs}
            flat_updates, structure = utils.flatten_updates(updates)
            flat_avg_update = flat_updates.mean(dim=0)
            bucket_update = utils.unflatten_update(flat_avg_update, structure)
            # n_data
            n_datas = [client_msgs[idx]['n_data'] for idx in bucket_client_idxs]
            bucket_n_data = sum(n_datas)
            # pack
            bucket_client_msg = {
                'update': bucket_update,
                'n_data': bucket_n_data,
            }
            bucket_client_msgs[bucket_id] = bucket_client_msg
        
        n_byz_updates = knowledge['n_byz_updates']
        bucket_n_byz_updates = math.ceil(n_byz_updates / s)
        bucket_knowledge = {
            'n_byz_updates': bucket_n_byz_updates,
        }
        return bucket_client_msgs, bucket_knowledge