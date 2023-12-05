from __future__ import annotations

import json
import numpy as np
import numpy.random
import os
import torch
import torchvision.io as io
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder
from torchvision.datasets.folder import default_loader
from typing import Any, Callable, Optional, Tuple


class ImageNet12(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        train=True):
        split = 'train' if train else 'val'
        root = os.path.join(root, split)

        super().__init__(root, transform, target_transform, loader, is_valid_file)

class Femnist(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None) -> None:
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.n_classes = 62
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
        if self.transform is not None:
            data = self.transform(self.data[index])
        else:
            data = self.data[index]
        
        if self.target_transform is not None:
            target = self.target_transform(self.targets[index])
        else:
            target = self.targets[index]
        
        return data, target

def load_femnist(root: str, transform=None, target_transform=None)->Tuple[list[Femnist], Femnist]:
    if root.startswith('~'):
        root = os.path.expanduser(root)
    if not os.path.exists(root):
        raise Exception('invalid data root')

    client_datas = []

    tr_path = os.path.join(root, 'data/train')
    rel_paths = os.listdir(tr_path)
    for rel_path in rel_paths:
        abs_path = os.path.join(tr_path, rel_path)
        with open(abs_path, 'r') as f:
            json_data = json.load(f)
            for user in json_data['users']:
                x = torch.Tensor(json_data['user_data'][user]['x']).reshape(-1, 1, 28, 28)
                y = torch.Tensor(json_data['user_data'][user]['y']).long()
                client_data = Femnist(x, y, transform, target_transform)
                client_datas.append(client_data)
    
    te_xs = []
    te_ys = []
    te_path = os.path.join(root, 'data/test')
    rel_paths = os.listdir(te_path)
    for rel_path in rel_paths:
        abs_path = os.path.join(te_path, rel_path)
        with open(abs_path, 'r') as f:
            json_data = json.load(f)
            for user in json_data['users']:
                te_xs += json_data['user_data'][user]['x']
                te_ys += json_data['user_data'][user]['y']
    te_xs = torch.Tensor(te_xs).reshape(-1, 1, 28, 28)
    te_ys = torch.Tensor(te_ys).long()
    te_data = Femnist(te_xs, te_ys, transform, target_transform)

    return client_datas, te_data

def load_data(
    dataset: str, root: str, 
    n_clients: int, partition: str, 
    dirichlet_beta: float, dirichelt_min_n_data: int, 
) -> Tuple[list[ClientDataset] | list[Femnist], CIFAR10 | CIFAR100 | ImageNet12 | Femnist]:
    if dataset == 'cifar10':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
            ]
        )
        tr_data = CIFAR10(root=root, train=True, transform=transform, download=True)
        te_data = CIFAR10(root=root, train=False, transform=transform, download=True)
        client_datas = partition_data(tr_data, n_clients, partition, dirichlet_beta=dirichlet_beta, dirichelt_min_n_data=dirichelt_min_n_data)
    elif dataset == 'cifar100':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
            ]
        )
        tr_data = CIFAR100(root=root, train=True, transform=transform, download=True)
        te_data = CIFAR100(root=root, train=False, transform=transform, download=True)
        client_datas = partition_data(tr_data, n_clients, partition, dirichlet_beta=dirichlet_beta, dirichelt_min_n_data=dirichelt_min_n_data)
    elif dataset == 'imagenet12':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tr_data = ImageNet12(root=root, train=True, transform=transform)
        te_data = ImageNet12(root=root, train=False, transform=transform)
        client_datas = partition_data(tr_data, n_clients, partition, dirichlet_beta=dirichlet_beta, dirichelt_min_n_data=dirichelt_min_n_data)
    elif dataset == 'femnist':
        transform = None
        target_transform = None
        client_datas, te_data = load_femnist(root, transform, target_transform)
    else:
        raise Exception('invalid dataset')

    print(f'datalens: {[len(client_data) for client_data in client_datas]}')

    return client_datas, te_data

class ClientDataset(Dataset):
    def __init__(self, total_data: CIFAR10 | CIFAR100 | ImageNet12, client_data_idxs: np.ndarray) -> None:
        super().__init__()
        self.total_data = total_data
        self.client_data_idxs = client_data_idxs
    
    def __len__(self):
        return len(self.client_data_idxs)
    
    def __getitem__(self, idx):
        return self.total_data[self.client_data_idxs[idx]]
    
    def print_cls_distribution(self):
        targets_arr = np.array(self.total_data.targets)[self.client_data_idxs]
        n_labels = targets_arr.max() + 1
        n_cls_samples = []
        for c in range(n_labels):  # type: ignore
            n_samples = np.sum(targets_arr==c)
            n_cls_samples.append(n_samples)
        print(f'class distribution: {n_cls_samples}')

def partition_data(
    data: CIFAR10 | CIFAR100 | ImageNet12,
    n_clients: int, partition: str, 
    dirichlet_beta: float=0, dirichelt_min_n_data: int=0, 
    # n_shards: int=0,
) -> list[ClientDataset]:
    if partition == 'iid':
        n_data = len(data)  # type: ignore
        n_main_data = n_data // n_clients
        n_tail_data = n_data % n_clients

        client_datas: list[ClientDataset] = []
        permuted_data_idxs: np.ndarray = numpy.random.permutation(n_data)
        dataidx_idx = 0
        for client_idx in range(n_clients):
            n_client_data = n_main_data + int(client_idx < n_tail_data)
            client_data_idxs: np.ndarray = permuted_data_idxs[dataidx_idx : dataidx_idx + n_client_data]
            client_data = ClientDataset(total_data=data, client_data_idxs=client_data_idxs)
            client_datas.append(client_data)
            dataidx_idx += n_client_data
    elif partition == 'dirichlet':
        min_n_data = -1
        labels: np.ndarray = np.array(data.targets)  # type: ignore
        n_classes = labels.max() + 1
        while min_n_data < dirichelt_min_n_data:
            total_client_data_idxs = [np.array([], dtype=np.int64) for _ in range(n_clients)]
            # C * N
            proportions = numpy.random.dirichlet(alpha=[dirichlet_beta for _ in range(n_clients)], size=n_classes)
            proportions = proportions / np.sum(proportions, axis=-1, keepdims=True)
            proportions = np.cumsum(proportions, axis=-1)
            for class_idx in range(n_classes):  # type: ignore
                class_data_idxs: np.ndarray = np.where(labels == class_idx)[0]  # type: ignore
                n_class_data = len(class_data_idxs)
                split_pos = (proportions[class_idx] * n_class_data).astype(np.int64)
                numpy.random.shuffle(class_data_idxs)
                client_class_data_idxs = np.split(class_data_idxs, split_pos)
                for client_idx in range(n_clients):
                    total_client_data_idxs[client_idx] = np.concatenate([total_client_data_idxs[client_idx], client_class_data_idxs[client_idx]])
            min_n_data = min([len(client_data_idx) for client_data_idx in total_client_data_idxs])
        client_datas: list[ClientDataset] = []
        for client_idx in range(n_clients):
            numpy.random.shuffle(total_client_data_idxs[client_idx])  # type: ignore
            client_data = ClientDataset(total_data=data, client_data_idxs=total_client_data_idxs[client_idx])  # type: ignore
            client_datas.append(client_data)
    else:
        raise Exception('invalid partition')
    
    for client_data in client_datas:
        client_data.print_cls_distribution()
        
    return client_datas