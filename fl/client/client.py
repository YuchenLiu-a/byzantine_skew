import torch

from argparse import Namespace
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import time

from ..utils import get_loss, get_model, get_optimizer

class Client:
    def __init__(self, data: Dataset, args: Namespace):
        self.data = data
        self.args = args
        self.local_model = get_model(self.args.architecture, self.args.n_classes)
        self.loss = get_loss(self.args.loss)
        self.dataloader = DataLoader(self.data, batch_size=self.args.client_batch_size, shuffle=True, num_workers=self.args.n_workers, pin_memory=self.args.pin_memory)

    def local_update(self, server_message: dict):
        global_state = server_message['model_state']
        self.local_model.load_state_dict(global_state)

        optimizer = get_optimizer(self.args.client_optimizer, self.local_model.parameters(), self.args.client_lr, self.args.client_momentum, self.args.client_weight_decay)

        tic = time.time()
        self.local_model.cuda()
        self.local_model.train()
        for _ in range(self.args.client_n_epochs):
            for (inputs, targets) in self.dataloader:
                # drop batch of size 1 when there is batch normalization
                if self.args.architecture in ['resnet18', 'squeezenet']:
                    batch_size = len(inputs)
                    if batch_size == 1:
                        continue

                inputs, targets = inputs.cuda(), targets.cuda()
                
                optimizer.zero_grad()

                outputs = self.local_model(inputs)
                loss = self.loss(outputs, targets)
                
                loss.backward()
                if self.args.clip_max_norm is not None:
                    clip_grad_norm_(self.local_model.parameters(), max_norm=self.args.clip_max_norm)
                optimizer.step()
            self.local_model.cpu()
        
        with torch.no_grad():
            self.state_update = {}
            local_state = self.local_model.state_dict()
            for key in local_state:
                self.state_update[key] = local_state[key] - global_state[key]
        
        client_message = self.pack_other_message()
        client_message['update'] = self.state_update
        toc = time.time()

        return client_message
    
    def pack_other_message(self) -> dict:
        uploaded_message = {
            'n_data': len(self.data),  # type: ignore
        }

        return uploaded_message