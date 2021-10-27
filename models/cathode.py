import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import models.cathode_flows as fnn
from models.base_model import base_model
import numpy as np

class DensityEstimator:
    class Unknown(Exception):
        """ Error to raise for unkown DensityEstimator model """

    @classmethod
    def get_all_subclasses(cls):
        """ Get all the subclasses recursively of this class. """
        return DensityEstimator.__subclasses__()
        # for subclass in cls.__subclasses__():
        #     yield from subclass.get_all_subclasses()
        #     yield subclass

    @classmethod
    def _get_name(cls, name):
        return name.upper()

    @classmethod
    def _parse_yaml(cls, filename, latent_dim):
        with open(filename, 'r') as stream:
            cls.params = yaml.safe_load(stream)
            cls.params['num_inputs'] = latent_dim

        name = cls.params['ModelType']
        if name != 'MDN':
            if cls.params['Transform'] == 'RQS':
                name = name + '_RQS'

        return name

    def __new__(cls, *args, **kwargs):
        ld = args[0]
        filename = 'models/cathode_model.yml'
        name = cls._parse_yaml(filename, ld)
        name = cls._get_name(name)
        # for subclass in DensityEstimator.__subclasses__():
        for subclass in cls.get_all_subclasses():
            if subclass.name == name:
                # Using "object" base class methods avoid recursion here.
                return object.__new__(subclass)
        raise DensityEstimator.Unknown(f'Unknown model "{name}" requested')

    def __init__(self, *args, eval_mode=False, load_path=None,
                 device=torch.device("cpu"), verbose=False, **kwargs):
        # with open(filename, 'r') as stream:
        #     params = yaml.safe_load(stream)

        self.bound = False

        self.build(self.params, eval_mode, load_path, device, verbose)

    def build(self, params, eval_mode, load_path, device, verbose):
        """
        Used for building a flow based density estimator model
        from a yaml config file.
        """
        modules = []
        for i in range(params['num_blocks']):
            self.build_block(i, modules, params)

        if self.bound:
            self.model = fnn.FlowSequentialUniformBase(*modules)
            # modules += [fnn.InfiniteToFinite(to_finite=False)]
        else:
            self.model = fnn.FlowSequential(*modules)

        self.model = fnn.FlowSequential(*modules)
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)
        # Workaround bug in flow.py
        self.model.num_inputs = params['num_inputs']

        self.finalize_build(params, eval_mode, device, verbose, load_path)

    def load_model(self, load_path):
        if load_path is not None:
            print(f"Loading model parameters from {load_path}")
            self.model.load_state_dict(torch.load(load_path,
                                                  map_location='cpu'))

    def build_optimizer(self, params):
        optimizer = optim.__dict__[params['optimizer']['name']]
        optimizer_kwargs = params['optimizer']
        del optimizer_kwargs['name']
        self.optimizer = optimizer(self.model.parameters(),
                                   **optimizer_kwargs)

    def finalize_build(self, params, eval_mode, device, verbose, load_path):
        self.model.to(device)
        if verbose:
            print(self.model)
        total_parameters = sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad)
        print(f"DensityEstimator has {total_parameters} parameters")

        self.load_model(load_path)

        # Get the requested for optimizer
        self.build_optimizer(params)

        if eval_mode:
            self.model.eval()

    def build_block(self, modules, params):
        raise NotImplementedError


class Cathode(DensityEstimator):
    name = "MAF"

    def __init__(self, latent_dim, exp_name, device, *args, dir='test', **kwargs):
        super().__init__(*args, device=device, **kwargs)
        self.device = device
        self.exp_name = exp_name
        self.dir = dir
        self.set_loss_names()
        self.latent_dim = latent_dim

    def build_block(self, i, modules, params):
        modules += [
            fnn.MADE(params['num_inputs'], params['num_hidden'],
                     params['num_cond_inputs'],
                     act=params['activation_function'],
                     pre_exp_tanh=params['pre_exp_tanh']),
        ]
        if params['batch_norm']:
            modules += [fnn.BatchNormFlow(params['num_inputs'],
                                          momentum=params['batch_norm_momentum'])]
        modules += [fnn.Reverse(params['num_inputs'])]


    def set_loss_names(self):
        self.loss_names = ['mle']

    def sample(self, num, context=None):
        return self.model.sample(num_samples=num, cond_inputs=context)

    def log_prob(self, data):
        joint_samples = data[:, :self.latent_dim + 1]
        # joint_samples, s2 = data.split(int(data.shape[1] / 2), 1)
        # joint_samples = torch.cat([s1, s2], 0)
        return self.model.log_probs(joint_samples[:, :-1], cond_inputs=joint_samples[:, -1].view(-1, 1))

    def compute_loss(self, data, batch_size):
        self.mle = -self.log_prob(data).mean()
        self.set_loss_dict([self.mle])
        return self.mle

    # This isn't really a transformation, but other than the semantics this makes sense
    def transform_to_mass(self, features, dl, dh):
        sample = self.sample(dl.shape[0], context=dh[:, -1].view(-1, 1))
        lp = torch.zeros(sample.shape[0])
        return sample, lp

    def inverse_transform_to_mass(self, features, dl, dh):
        return self.transform_to_mass(features, dh, dl)


    def transform_to_data(self, dl, dh, batch_size=None):
        """Transform features in dl to the masses given in dh"""
        batch_size = None
        if batch_size is not None:
            raise Exception('Batch size arg not implemented.')
        else:
            # The last feature is the mass or resonant feature (lm = low mass, hm = high mass)
            lm = dl[:, -1].view(-1, 1)
            hm = dh[:, -1].view(-1, 1)
            low_mass_features = dl[:, :-1]
            return self.transform_to_mass(low_mass_features, lm, hm)

    # Transform to ... given data
    def inverse_transform_to_data(self, dl, dh, batch_size=None):
        """Transform features in dh to the masses given in dl"""
        # TODO: Need to implement the batch size, should take previous implementation and make it generic
        batch_size = None
        # The last feature is the mass or resonant feature (lm = low mass, hm = high mass)
        lm = dl[:, -1].view(-1, 1)
        hm = dh[:, -1].view(-1, 1)
        high_mass_features = dh[:, :-1]
        return self.inverse_transform_to_mass(high_mass_features, lm, hm)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        # self.transformer.load_state_dict(torch.load(path))
        self.load_model(path)

    def get_numpy(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
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

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()


def test():
    print([cls.__name__ for cls in DensityEstimator.__subclasses__()])
    # model = DE_MAF()
    # model.optimizer
    model = Cathode('test', torch.device('cpu'))
    model.optimizer


if __name__ == '__main__':
    test()
