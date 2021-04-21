import torch
from .base_model import base_model


class curtains_transformer(base_model):
    def __init__(self, INN, device, exp_name, dist_measure, nfeatures, dir='INN_test'):
        # nfeatures that we want to learn, plus the context feature
        self.take = nfeatures + 1
        super(curtains_transformer, self).__init__(INN, device, exp_name, dir=dir)
        self.dist_measure = dist_measure

    def set_loss_names(self):
        self.loss_names = ['OT distance']

    # Transform to ... given mass
    def transform_to_mass(self, features, lm, hm):
        # Providing the context means we condition on the masses
        # This function will return both the transformation and the log_prob (which we can't use),
        # so we take the first arg
        return self.transformer(features, context=torch.cat((lm, hm), 1))[0]

    # Transform to ... given data
    def transform_to_data(self, dl, dh):
        # The last feature is the mass or resonant feature (lm = low mass, hm = high mass)
        lm = dl[:, -1].view(-1, 1)
        hm = dh[:, -1].view(-1, 1)
        low_mass_features = dl[:, :-1]
        return self.transform_to_mass(low_mass_features, lm, hm)

    def inverse_transform_to_mass(self, features, lm, hm):
        # Providing the context means we condition on the masses
        # This function will return both the transformation and the log_prob (which we can't use),
        # so we take the first arg
        return self.transformer.inverse(features, context=torch.cat((lm, hm), 1))[0]

    # Transform to ... given data
    def inverse_transform_to_data(self, dl, dh):
        # The last feature is the mass or resonant feature (lm = low mass, hm = high mass)
        lm = dl[:, -1].view(-1, 1)
        hm = dh[:, -1].view(-1, 1)
        high_mass_features = dh[:, :-1]
        return self.inverse_transform_to_mass(high_mass_features, lm, hm)

    def compute_loss(self, data, batch_size):
        # The data is passed with concatenated pairs of low mass and high mass features
        # The first #self.take are the low mass samples (dl = data low)
        dl = data[:, :self.take]
        # The next #self.take are the high mass samples (dl = data low)
        dh = data[:, self.take:]
        # This returns the transformation we are after
        transformed = self.transform_to_data(dl, dh)
        # Drop the mass feature from the high mass sample
        high_mass_features = dh[:, :-1]
        # Calculate the distance between the transformation and the high mass
        dists = self.dist_measure(transformed, high_mass_features)
        self.set_loss_dict([dists])
        return self.loss_dict[self.loss_names[0]]


class two_way_curtains_transformer(curtains_transformer):
    def __init__(self, INN, device, exp_name, dist_measure, nfeatures, dir='INN_test'):
        super(two_way_curtains_transformer, self).__init__(INN, device, exp_name, dist_measure, nfeatures, dir=dir)
        self.set_loss_names()

    def set_loss_names(self):
        self.loss_names = ['Forward distance', 'Inverse distance']

    def compute_loss(self, data, batch_size):
        # The data is passed with concatenated pairs of low mass and high mass features
        # The first #self.take are the low mass samples (dl = data low)
        dl = data[:, :self.take]
        # The next #self.take are the high mass samples (dl = data low)
        dh = data[:, self.take:]
        # This returns the transformation from low mass to high mass
        transformed_hm = self.transform_to_data(dl, dh)
        # This returns the transformation from high mass to low mass
        transformed_lm = self.inverse_transform_to_data(dl, dh)
        # Drop the mass from the feature sample
        high_mass_features = dh[:, :-1]
        low_mass_features = dl [:, :-1]
        # Calculate the distance between the transformation and truth
        forward_dists = self.dist_measure(transformed_hm, high_mass_features)
        inverse_dists = self.dist_measure(transformed_lm, low_mass_features)
        self.set_loss_dict([forward_dists, inverse_dists])
        return sum([self.loss_dict[nm] for nm in self.loss_names])
