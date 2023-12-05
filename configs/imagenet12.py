from argparse import Namespace
imagenet12_config = Namespace(
    data_root = 'data/imagenet12',
    n_clients = 50,
    n_classes = 12,

    architecture = 'resnet18',
    loss = 'cross_entropy',
    server_n_epochs = 200,
    n_sampled_clients = 50,

    client_batch_size = 128,
    client_optimizer = 'sgd',
    client_lr = 0.1,
    client_momentum = 0.9,
    client_weight_decay = 1e-4,
    client_n_epochs = 1,
    clip_max_norm = 2.0,
)

if __name__ == '__main__':
    print(vars(imagenet12_config))
