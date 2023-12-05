from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
import fl

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'femnist', 'imagenet12' ], required=True, )
    parser.add_argument('--data_root', type=str, )
    parser.add_argument('--n_classes', type=int, )
    # data partition
    parser.add_argument('--n_clients', type=int, )
    parser.add_argument('--partition', type=str, choices=['iid', 'dirichlet', ], )
    parser.add_argument('--dirichlet_beta', default=0.5, type=float, )
    parser.add_argument('--dirichlet_min_n_data', default=10, type=int, )

    # attack
    parser.add_argument('--byz_client_ratio', type=float, required=True, )
    parser.add_argument('--ben_known_ratio', type=float, default=0, )
    parser.add_argument('--attack', type=str, choices=['bitflip', 'ipm', 'lie', 'mimic', 'minmax', 'minsum', 'omn', 'skew', ])
    parser.add_argument('--ipm_epsilon', default=0.1, type=float, )
    parser.add_argument('--lie_z', default=1.5, type=float, )
    deviation_choices = ['unit_vec', 'sign', 'skew_mode', 'skew_std', 'neg_skew_mode', 'neg_skew_std', 'neg_std', 'ran_std', 'std' ]
    parser.add_argument('--minmax_deviation', default='std', type=str, choices=deviation_choices, )
    parser.add_argument('--minsum_deviation', default='std', type=str, choices=deviation_choices, )
    parser.add_argument('--omn_deviation', default='std', type=str, choices=deviation_choices, )
    parser.add_argument('--omn_evals', default=8, type=int, )
    parser.add_argument('--skew_lambda', default=1, type=float, )

    # server
    parser.add_argument('--architecture', type=str, choices=['alexnet', 'resnet18', 'squeezenet', ], )
    parser.add_argument('--loss', type=str, choices=['cross_entropy', ], )
    parser.add_argument('--framework', type=str, choices=['fedavg', ], required=True, )
    parser.add_argument('--server_n_epochs', type=int, )
    parser.add_argument('--n_sampled_clients', type=int, )
    parser.add_argument('--client_sample_ratio', type=float, )
    # aggregator
    parser.add_argument('--nnm', action='store_true', default=False, )
    parser.add_argument('--bucket_s', default=1, type=int)
    parser.add_argument('--aggregator', type=str, choices=['aksel', 'bulyan', 'cc', 'dnc', 'krum', 'mean', 'median', 'rfa', 'trmean', 'mda', ], required=True, )
    parser.add_argument('--cc_l', default=1, type=int, )
    parser.add_argument('--cc_tau', default=10, type=float, )
    parser.add_argument('--dnc_n_iters', default=1, type=int, )
    parser.add_argument('--dnc_b', default=1000, type=int, help='dimension of subsamples')
    parser.add_argument('--dnc_c', default=1.0, type=float, help='filtering fraction', )
    parser.add_argument('--rfa_budget', default=8, type=int, )
    parser.add_argument('--rfa_eps', default=1e-7, type=float, )

    # client
    parser.add_argument('--client_batch_size', type=int, )
    parser.add_argument('--client_optimizer', type=str, choices=['sgd', ], )
    parser.add_argument('--client_lr', type=float, )
    parser.add_argument('--client_momentum', type=float, )
    parser.add_argument('--client_weight_decay', type=float, )
    parser.add_argument('--client_n_epochs', type=int, )
    parser.add_argument('--clip_max_norm', default=None, type=float)

    # random
    parser.add_argument('--seed', type=int, required=True, )

    # efficiency
    parser.add_argument('--pin_memory', action='store_true', default=False, )
    parser.add_argument('--n_workers', default=1, type=int, )
    parser.add_argument('--te_batch_size', default=128, type=int, )

    # wandb
    parser.add_argument('--wandb', action='store_true', default=False, )

    # gpu
    parser.add_argument('--devices', type=str, required=True)
    parser.add_argument('--nice', action='store_true', default=False, )

    args = parser.parse_args()

    # set default config
    if args.dataset == 'femnist':
        from configs import femnist_config as config
    elif args.dataset == 'cifar10':
        from configs import cifar10_config as config
    elif args.dataset == 'cifar100':
        from configs import cifar100_config as config
    elif args.dataset == 'imagenet12':
        from configs import imagenet12_config as config
    else:
        raise Exception('invalid dataset')
    for name, val in vars(config).items():
        if getattr(args, name, None) is None:
            setattr(args, name, val)

    return args

def get_nondefault_setups(argv: list[str]):
    features = []
    for arg in argv[1:]:
        if not arg.startswith('--'):
            features.append(arg)
    return features

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # parse arguments
    args = parse_args()
    # set gitversion
    git_version = __file__.split('/')[-2]
    setattr(args, 'gitversion', git_version)
    # get non-default setups
    
    # initialize wandb
    if args.wandb:
        import wandb
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project='byzantine_skew')
        wandb.config.update(args)
        nondefault_setups = get_nondefault_setups(sys.argv)
        run_name = '_'.join([args.gitversion] + nondefault_setups)
        wandb.run.name = run_name # type: ignore
    fl.fl_train(args)