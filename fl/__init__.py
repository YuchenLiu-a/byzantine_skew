from __future__ import annotations

import math
import numpy.random as random
import time
from . import utils

from torch.utils.data.dataloader import DataLoader
from .defender import Defender
from .attacker import get_attacker
from .client import Client

def update_args(args, n_clients: int):
    setattr(args, 'n_clients', n_clients)
    # compute number of byzantine clients
    setattr(args, 'n_byz_clients', math.ceil(args.n_clients * args.byz_client_ratio))

    # sychronize n_sampled_clients and client_sample_ratio
    if args.n_sampled_clients is None:
        setattr(args, 'n_sampled_clients', math.ceil(args.n_clients * args.client_sample_ratio))
    else:
        setattr(args, 'client_sample_ratio', args.n_sampled_clients / args.n_clients)

    return args

def fl_train(args):
    utils.setup_seed(args.seed)
    # load data
    client_datas, te_data = utils.load_data(args.dataset, args.data_root, args.n_clients, args.partition, args.dirichlet_beta, args.dirichlet_min_n_data)
    update_args(args, len(client_datas))
    # record arguments
    print(f'args: {args}')
    # initialize client-side
    clients = {str(idx): Client(client_datas[idx], args) for idx in range(len(client_datas))}
    # initialize attacker
    byz_client_idxs = {str(idx) for idx in range(args.n_byz_clients)}
    byz_clients = {idx: clients[idx] for idx in byz_client_idxs}
    attacker = get_attacker(byz_clients, args=args)
    # initialize server-side
    global_model = utils.get_model(args.architecture, args.n_classes)
    defender = Defender(args, byz_client_idxs)
    # test
    te_dataloader = DataLoader(dataset=te_data, batch_size=args.te_batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=args.pin_memory)
    loss_fun = utils.get_loss(args.loss)

    # start
    for epoch in range(args.server_n_epochs):
        tic = time.time()
        # sample clients
        sampled_client_idxs: set[str] = set(str(idx) for idx in random.choice(args.n_clients, args.n_sampled_clients, replace=False).tolist())
        # distribute
        server_message = {'model_state': global_model.state_dict()}
        # benign clients perform local update
        sampled_ben_client_idxs = sampled_client_idxs.difference(byz_client_idxs)
        ben_client_msgs = {idx: clients[idx].local_update(server_message) for idx in sampled_ben_client_idxs}
        # attack
        num_sampled_bex = len(ben_client_msgs)
        num_known_benign = int(args.ben_known_ratio * num_sampled_bex)
        known_benign_subidx = random.choice(num_sampled_bex, num_known_benign, replace=False).tolist()
        sampled_ben_client_idxs_lt = list(sampled_ben_client_idxs)
        known_benign_client_idxs = [sampled_ben_client_idxs_lt[i] for i in known_benign_subidx]
        sampled_byz_client_idxs = sampled_client_idxs.intersection(byz_client_idxs)
        defender_knowledge = {'n_byz_updates': len(sampled_byz_client_idxs)}
        attacker_knowledge = {
            'benign_client_messages': ben_client_msgs,
            'known_benign_client_idxs': known_benign_client_idxs,
            'defender': defender, 
            'defender_knowledge': defender_knowledge, 
        }
        byz_client_msgs, byz_verbose_log = attacker.attack(sampled_byz_client_idxs, server_message, attacker_knowledge)
        # mergetorch slice
        sampled_client_msgs = {**ben_client_msgs, **byz_client_msgs}
        # aggregate
        agg_update, agg_verbose_log = defender.defend(sampled_client_msgs, defender_knowledge)
        # step
        utils.step(agg_update, global_model)
        # evaluate
        ben_stats = utils.eval_ben_updates(ben_client_msgs)
        byz_dev_stats = utils.eval_byz_dev(byz_client_msgs, ben_client_msgs)
        agg_dev_stats = utils.eval_agg_dev(agg_update, ben_client_msgs)
        perf_stats = utils.eval_perf(global_model, te_dataloader, loss_fun)
        # wandb log
        if args.wandb:
            import wandb
            log_data = {
                'sampled_client_idxs': list(sampled_client_idxs),
                **ben_stats,
                **byz_verbose_log,
                **byz_dev_stats,
                **agg_verbose_log,
                **agg_dev_stats,
                **perf_stats,
            }
            wandb.log(log_data) # type: ignore
        if not args.nice and epoch == 0:
            utils.block(args.devices)
        toc = time.time()
        # output
        print(f'==================== epoch {epoch} time {toc - tic: .2f} ====================')
        print(f'sampled clients: {sampled_client_idxs}')
        utils.print_dict(ben_stats, 'ben_updates stats')
        utils.print_dict(byz_verbose_log, args.attack)
        utils.print_dict(byz_dev_stats, 'byz_update dev stats')
        utils.print_dict(agg_verbose_log, args.aggregator)
        utils.print_dict(agg_dev_stats, 'agg_update dev stats')
        utils.print_dict(perf_stats, f'performance {epoch}')
        

    print('==================== end of training ====================')

__all__ = [
    'fl_train',
]