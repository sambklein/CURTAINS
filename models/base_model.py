from abc import ABC, abstractmethod
import torch
import numpy as np
import torch.nn as nn


class base_model(ABC, nn.Module):
    """
    Within the scope of this project this is the basic definition of a model
    """

    def __init__(self, transformer, device, exp_name, dir='test', mass_sampler=None):
        super(base_model, self).__init__()
        self.transformer = transformer
        self.device = device
        self.exp_name = exp_name
        self.dir = dir
        self.mass_sampler = mass_sampler
        self.set_loss_names()

    @abstractmethod
    def compute_loss(self, data, batch_size):
        return 0

    @abstractmethod
    def set_loss_names(self):
        return 0

    @abstractmethod
    def transform_to_mass(self, features, lm, hm):
        """
        Transform features to [lm, hm], return transform, log_prob
        """
        return 0

    def sample_mass(self, mass, n_sample=None):
        if n_sample is None:
            n_sample = len(mass)
        return self.mass_sampler.sample(n_sample, limits=(mass.min().item(), mass.max().item())).to(mass.device)

    def transform_to_data(self, dl, dh, batch_size=None):
        """Transform features in dl to the masses given in dh"""
        batch_size = None
        if batch_size is not None:
            raise Exception('Batch size arg not implemented.')
            # n_full = int(dl.shape[0] // batch_size)
            # nfit = n_full * batch_size
            # # The last feature is the mass or resonant feature (lm = low mass, hm = high mass)
            # lm = dl[:nfit, -1].view(-1, batch_size, 1)
            # hm = dh[:nfit, -1].view(-1, batch_size, 1)
            # low_mass_features = dl[:nfit, :-1].view(-1, batch_size, self.take - 1)
            # nbatches = lm.size[1]
            # sampled_features = torch.empty_like(low_mass_features)
            # for i in range(nbatches):
            #     sampled_features[i] = self.transform_to_mass(low_mass_features[i], lm[i], hm[i])
            # sampled_features = sampled_features.view(-1, self.take - 1)
            # return torch.cat(
            #     (sampled_features, self.transform_to_mass(low_mass_features[nfit:, :-1], dl[nfit:, -1], dh[nfit:, -1])))
        else:
            # The last feature is the mass or resonant feature (lm = low mass, hm = high mass)
            lm = dl[:, -1].view(-1, 1)
            hm = dh[:, -1].view(-1, 1)
            if self.mass_sampler is not None:
                hm_test = self.sample_mass(hm, n_sample=lm.shape[0])
                # if hm.shape[0] != lm.shape[0]:
                #     print(lm.shape)
                #     print(hm.shape)
                #     mass = hm
                #     print(mass.min().item(), mass.max().item())
                # hm[:len(hm_test)] = hm_test
                hm = hm_test
            low_mass_features = dl[:, :-1]
            return self.transform_to_mass(low_mass_features, lm, hm)

    @abstractmethod
    def inverse_transform_to_mass(self, features, lm, hm):
        """
        Transform features to mass range [lm, hm], return transform, log_prob
        """
        return 0

    # Transform to ... given data
    def inverse_transform_to_data(self, dl, dh, batch_size=None):
        """Transform features in dh to the masses given in dl"""
        batch_size = None
        # The last feature is the mass or resonant feature (lm = low mass, hm = high mass)
        lm = dl[:, -1].view(-1, 1)
        hm = dh[:, -1].view(-1, 1)
        if self.mass_sampler is not None:
            lm_test = self.sample_mass(lm, n_sample=hm.shape[0])
            lm = lm_test
        high_mass_features = dh[:, :-1]
        return self.inverse_transform_to_mass(high_mass_features, lm, hm)

    def save(self, path):
        torch.save(self.transformer.state_dict(), path)

    def load(self, path):
        self.transformer.load_state_dict(torch.load(path))

    def get_numpy(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return x

    def my_round(self, x, nsf):
        if x:
            try:
                x = round(x, nsf - (int(np.floor(np.log10(abs(x)))) - 1))
            except Exception as e:
                print(e)
        return x

    def get_loss_state(self, nsf=10):
        return {key: self.my_round(value.item(), nsf) for key, value in self.loss_dict.items()}

    def set_loss_dict(self, loss):
        # Given a list of losses, ordered in the same way as the models
        self.loss_dict = {self.loss_names[i]: loss for i, loss in enumerate(loss)}
