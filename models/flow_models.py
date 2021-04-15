import torch.nn as nn
import torch


# TODO: sort out the inheritance so this takes from base_model

class flow_builder(nn.Module):
    def __init__(self, flow, base_dist, device, exp_name, dir='INN_test'):
        super(flow_builder, self).__init__()
        self.flow = flow
        self.base_dist = base_dist
        self.device = device
        self.exp_name = exp_name
        self.loss_names = ['mle']
        self.dir = dir

    # Semantically within the scope of this project this should return the latent space.
    def forward(self, data):
        return self.encode(data)

    def encode(self, x):
        return self.flow.transform_to_noise(x)

    def decode(self, x):
        return self.forward(x)

    def save(self, path):
        torch.save(self.flow.state_dict(), path)

    def load(self, path):
        self.flow.load_state_dict(torch.load(path))

    def sample(self, num):
        return self.flow.sample(num)

    # TODO: should be a utility
    def get_numpy(self, x):
        if torch.is_tensor(x):
            x = x.detach()
            if self.device != 'cpu':
                x = x.cpu()
            x = x.numpy()
        return x

    def log_prob(self, data):
        return self.flow.log_prob(data)

    def compute_loss(self, data, batch_size):
        self.mle = -self.log_prob(data).mean()
        return self.mle

    def get_loss_state(self, nsf=10):
        return {'mle': self.mle.item()}


class contextual_flow(flow_builder):
    def __init__(self, flow, base_dist, device, exp_name, dir='INN_test'):
        super(contextual_flow, self).__init__(flow, base_dist, device, exp_name, dir=dir)

    def log_prob(self, data):
        return self.flow.log_prob(data[:, :-1], context=data[:, -1].view(-1, 1))

    def sample(self, num, context):
        return self.flow.sample(num, context=context)