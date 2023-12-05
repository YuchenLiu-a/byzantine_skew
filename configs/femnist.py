from argparse import Namespace
femnist_config = Namespace(
    data_root = 'data/leaf/data/femnist',
    # n_clients = 50,
    n_classes = 62,

    architecture = 'femnistnet',
    loss = 'cross_entropy',
    server_n_epochs = 800,
    n_sampled_clients = 10,

    client_batch_size = 128,
    client_optimizer = 'sgd',
    client_lr = 0.5,
    client_momentum = 0.5,
    client_weight_decay = 1e-4,
    client_n_epochs = 1,
    clip_max_norm = 2.0,
)

if __name__ == '__main__':
    print(vars(femnist_config))
