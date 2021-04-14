import torch.nn as nn
import torch


# TODO: sort out the inheritance so you don't have the following code duplication, who knows how far this will go in the future

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

    def autoencode(self, data):
        print('Encoding and decoding is irrelevant for invertible models')
        return data

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


class curtains_transformer(flow_builder):
    # TODO: instead of a base dist pass a pseudo sampler for the data dist?
    def __init__(self, flow, base_dist, device, exp_name, dist_measure, nfeatures, dir='INN_test'):
        self.dist_measure = dist_measure
        self.take = nfeatures + 1
        super(curtains_transformer, self).__init__(flow, base_dist, device, exp_name, dir=dir)

    def transform_to_mass(self, features, lm, hm):
        return self.flow.transform_to_noise(features, context=torch.cat((lm, hm), 1))

    def transform_to_data(self, dl, dh):
        lm = dl[:, -1].view(-1, 1)
        hm = dh[:, -1].view(-1, 1)
        low_mass_features = dl[:, :-1]
        return self.transform_to_mass(low_mass_features, lm, hm)

    def compute_loss(self, data, batch_size):
        dl = data[:, :self.take]
        dh = data[:, self.take:]
        transformed = self.transform_to_data(dl, dh)
        high_mass_features = dh[:, :-1]
        self.loss = self.dist_measure(transformed, high_mass_features)
        return self.loss

    def get_loss_state(self, nsf=10):
        return {'Distance': self.loss.item()}
