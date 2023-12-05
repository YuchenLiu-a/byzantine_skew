from __future__ import annotations

from argparse import Namespace

from ..client import Client
from .bitflip import BitFlip
from .ipm import Ipm
from .lie import Lie
from .mimic import Mimic
from .minmax import MinMax
from .minsum import MinSum
from .omn import Omn
from .skew import Skew

def get_attacker(byz_clients: dict[str, Client], args: Namespace):
    if args.attack == 'bitflip':
        return BitFlip(byz_clients, args)
    elif args.attack == 'ipm':
        return Ipm(byz_clients, args)
    elif args.attack == 'lie':
        return Lie(byz_clients, args)
    elif args.attack == 'mimic':
        return Mimic(byz_clients, args)
    elif args.attack == 'minmax':
        return MinMax(byz_clients, args)
    elif args.attack == 'minsum':
        return MinSum(byz_clients, args)
    elif args.attack == 'omn':
        return Omn(byz_clients, args)
    elif args.attack == 'skew':
        return Skew(byz_clients, args)
    else:
        raise Exception('invalid attack')

__all__ = [
    'get_attacker'
]