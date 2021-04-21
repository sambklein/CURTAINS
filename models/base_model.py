from abc import ABC, abstractmethod
import torch
import numpy as np
import torch.nn as nn


class base_model(ABC, nn.Module):
    """
    Within the scope of this project this is the basic definition of a model
    """

    def __init__(self, transformer, device, exp_name, dir='test'):
        super(base_model, self).__init__()
        self.transformer = transformer
        self.device = device
        self.exp_name = exp_name
        self.dir = dir
        self.set_loss_names()

    @abstractmethod
    def compute_loss(self, data, batch_size):
        return 0

    @abstractmethod
    def set_loss_names(self):
        return 0

    @abstractmethod
    def transform_to_data(self, data_l, data_h):
        """
        Transform to the mass range in data_h from the features/mass in data_l
        """
        return 0

    def save(self, path):
        torch.save(self.transformer.state_dict(), path)

    def load(self, path):
        self.transformer.load_state_dict(torch.load(path))

    def get_numpy(self, x):
        if torch.is_tensor(x):
            x = x.detach()
            if self.device != 'cpu':
                x = x.cpu()
            x = x.numpy()
        return x

    def my_round(self, x, nsf):
        if x:
            x = round(x, nsf - (int(np.floor(np.log10(abs(x)))) - 1))
        return x

    def get_loss_state(self, nsf=10):
        return {key: self.my_round(value.item(), nsf) for key, value in self.loss_dict.items()}

    def set_loss_dict(self, loss):
        # Given a list of losses, ordered in the same way as the models
        self.loss_dict = {self.loss_names[i]: loss for i, loss in enumerate(loss)}
