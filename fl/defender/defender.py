from __future__ import annotations

from argparse import Namespace

from .bucket import bucket
from .nnm import nnm

def get_aggregator(args: Namespace, byz_client_idxs: set[str]):
    if args.aggregator == 'aksel':
        from .aksel import Aksel
        return Aksel(args, byz_client_idxs)
    elif args.aggregator == 'bulyan':
        from .bulyan import Bulyan
        return Bulyan(args, byz_client_idxs)
    elif args.aggregator == 'cc':
        from .centeredclipping import CenteredClipping
        return CenteredClipping(args, byz_client_idxs)
    elif args.aggregator == 'clamp':
        from .clamp import Clamp
        return Clamp(args, byz_client_idxs)
    elif args.aggregator == 'dnc':
        from .dnc import Dnc
        return Dnc(args, byz_client_idxs)
    elif args.aggregator == 'krum':
        from .krum import Krum
        return Krum(args, byz_client_idxs)
    elif args.aggregator == 'mda':
        from .mda import Mda
        return Mda(args, byz_client_idxs)
    elif args.aggregator == 'mean':
        from .mean import Mean
        return Mean(args, byz_client_idxs)
    elif args.aggregator == 'median':
        from .median import Median
        return Median(args, byz_client_idxs)
    elif args.aggregator == 'rfa':
        from .rfa import Rfa
        return Rfa(args, byz_client_idxs)
    elif args.aggregator == 'trmean':
        from .trmean import Trmean
        return Trmean(args, byz_client_idxs)
    else:
        raise Exception('invalid aggregator')

class Defender:
    def __init__(self, args: Namespace, byz_client_idxs: set[str]) -> None:
        self.args = args
        self.aggregator = get_aggregator(args, byz_client_idxs)

    def defend(self, client_msgs: dict[str, dict], knowledge: dict):
        if self.args.nnm:
            client_msgs, knowledge = nnm(client_msgs, knowledge)
        bucket_client_msgs, bucket_knowledge = bucket(client_msgs, self.args.bucket_s, knowledge)
        agg_update, verbose_log = self.aggregator(bucket_client_msgs, bucket_knowledge)
        return agg_update, verbose_log