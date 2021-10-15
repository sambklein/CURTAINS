import torch.nn as nn
import torch
from .base_model import base_model


class flow_builder(base_model):
    def __init__(self, flow, base_dist, device, exp_name, dir=dir):
        super(flow_builder, self).__init__(flow, device, exp_name, dir=dir)
        # TODO: saving the flow in two places, also labelled transformer..
        self.flow = flow
        self.base_dist = base_dist
        self.latent_dim = self.base_dist.sample(10).shape[1]

    def set_loss_names(self):
        self.loss_names = ['mle']

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

    def sample(self, num, context=None):
        return self.flow.sample(num, context=context)

    def log_prob(self, data):
        # joint_samples = data.view(-1, self.latent_dim + 1)
        joint_samples = data[:, :self.latent_dim + 1]
        return self.flow.log_prob(joint_samples[:, :-1])

    def compute_loss(self, data, batch_size):
        self.mle = -self.log_prob(data).mean()
        self.set_loss_dict([self.mle])
        return self.mle

    # This isn't really a transformation, but other than the semantics this makes sense
    def transform_to_mass(self, features, dl, dh):
        return self.sample_and_log_prob(dl.shape[0], dh[:, -1].view(-1, 1))

    def inverse_transform_to_mass(self, features, dl, dh):
        return self.sample_and_log_prob(dh.shape[0], dl[:, -1].view(-1, 1))


class contextual_flow(flow_builder):
    def __init__(self, flow, base_dist, device, exp_name, dir='INN_test'):
        super(contextual_flow, self).__init__(flow, base_dist, device, exp_name, dir=dir)

    def log_prob(self, data):
        joint_samples = data.view(-1, self.latent_dim + 1)
        return self.flow.log_prob(joint_samples[:, :-1], context=joint_samples[:, -1].view(-1, 1))

    def transform_to_mass(self, features, lm, hm):
        samples, log_prob = self.flow.sample_and_log_prob(1, context=hm.view(-1, 1))
        return samples.squeeze(), log_prob

    def inverse_transform_to_mass(self, features, lm, hm):
        return self.transform_to_mass(features, hm, lm)


class flow_for_flow(flow_builder):
    def __init__(self, mtm_flow, dtm_flow, base_dist, device, exp_name, dir='INN_test'):
        super(flow_for_flow, self).__init__(mtm_flow, base_dist, device, exp_name, dir=dir)
        self.dtm_flow = dtm_flow
        self.take = self.base_dist.sample(10).shape[1] + 1

    # def set_loss_names(self):
    #     self.loss_names = ['mle_base', 'mle_transformer']
    def set_loss_names(self):
        self.loss_names = ['mle_transformer']

    def log_prob(self, data, context=None):
        # return self.transformer.log_prob(data[:, :-1], context=data[:, -1].view(-1, 1))
        return self.transformer_log_prob(data[:, :-1], data[:, -1].view(-1, 1), context)

    # This is a custom wrapper for the transformer flow, so that we can provide the right context
    def transformer_log_prob(self, inputs, context, bdist_context=None):
        embedded_context = self.transformer._embedding_net(context)
        noise, logabsdet = self.transformer._transform(inputs, context=embedded_context)
        # We don't want to condition the base distribution on the input mass
        # TODO: discuss with Seb
        log_prob = self.transformer._distribution.log_prob(noise, context=bdist_context)
        return log_prob + logabsdet

    def compute_loss(self, data, batch_size):
        dl = data[:, :self.take]
        dh = data[:, self.take:]
        # self.mle = -self.log_prob(dh, dl[:, -1].view(-1, 1)).mean()
        self.mle = -self.log_prob(dh).mean()
        # self.mle_lm = -self.dtm_flow.log_prob(dl[:, :-1], context=dl[:, -1].view(-1, 1)).mean()
        # Concatenate the low and high mass to learn the full distribution
        # self.mle_lm = -self.dtm_flow.log_prob(torch.cat((dl[:, :-1], dh[:, :-1]), 0),
        #                                       context=torch.cat((dl[:, -1], dh[:, -1]), 0).view(-1, 1)).mean()
        loss_vals = [self.mle]
        self.set_loss_dict(loss_vals)
        return loss_vals

    def sample(self, num, context):
        return self.transformer.sample(num, context=context)

    def sample_lm(self, num, context=None):
        return self.dtm_flow.sample(num, context=context)

    def transform_to_data(self, dl, dh):
        return self.transformer._transform.inverse(dl[:, :-1], context=dh[:, -1].view(-1, 1))[0]
