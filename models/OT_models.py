import torch
from .base_model import base_model


class curtains_transformer(base_model):
    def __init__(self, INN, device, exp_name, dist_measure, nfeatures, dir='INN_test', det_beta=0, direction=0, **kwargs):
        # nfeatures that we want to learn, plus the context feature
        self.take = nfeatures + 1
        self.det_beta = det_beta
        super(curtains_transformer, self).__init__(INN, device, exp_name, dir=dir, **kwargs)
        self.dist_measure = dist_measure
        self.direction = direction

    def set_loss_names(self):
        self.loss_names = ['OT distance']
        if self.det_beta > 0:
            self.loss_names += ['detJ']

    # Transform to ... given mass
    def transform_to_mass(self, features, lm, hm):
        # Providing the context means we condition on the masses
        return self.transformer(features, context=torch.cat((lm, hm), 1))

    def inverse_transform_to_mass(self, features, lm, hm):
        # Providing the context means we condition on the masses
        return self.transformer.inverse(features, context=torch.cat((lm, hm), 1))

    def compute_loss(self, data, batch_size):
        # The data is passed with concatenated pairs of low mass and high mass features
        # The first #self.take are the low mass samples (dl = data low)
        dl = data[:, :self.take]
        # The next #self.take are the high mass samples (dl = data low)
        dh = data[:, self.take:]
        # This returns the transformation we are after
        if self.direction == 0:
            transformed, detJ = self.transform_to_data(dl, dh)
        elif self.direction == -1:
            transformed, detJ = self.inverse_transform_to_data(dl, dh)
        # Drop the mass feature from the high mass sample
        features = dh[:, :-1]
        # Calculate the distance between the transformation and the high mass
        dists = self.dist_measure(transformed, features)
        losses = [dists]
        mean_detJ = detJ.mean()
        if self.det_beta:
            losses += [mean_detJ]
        self.set_loss_dict(losses)
        return dists - self.det_beta * detJ.mean()


class delta_curtains_transformer(curtains_transformer):
    def __init__(self, INN, device, exp_name, dist_measure, nfeatures, dir='INN_test', **kwargs):
        super(delta_curtains_transformer, self).__init__(INN, device, exp_name, dist_measure, nfeatures, dir=dir,
                                                         **kwargs)

    def transform_to_mass(self, features, lm, hm):
        return self.transformer(features, context=hm - lm)

    def inverse_transform_to_mass(self, features, lm, hm):
        return self.transformer.inverse(features, context=hm - lm)


class delta_mass_curtains_transformer(curtains_transformer):
    def __init__(self, INN, device, exp_name, dist_measure, nfeatures, dir='INN_test', **kwargs):
        super(delta_mass_curtains_transformer, self).__init__(INN, device, exp_name, dist_measure, nfeatures, dir=dir,
                                                              **kwargs)

    def transform_to_mass(self, features, lm, hm):
        return self.transformer(features, context=torch.cat((lm, hm), 1))

    def inverse_transform_to_mass(self, features, lm, hm):
        return self.transformer.inverse(features, context=torch.cat((lm, hm), 1))

    # TODO: this ie) negative delta trainings
    # def compute_loss(self, data, batch_size):
    #     # The data is passed with concatenated pairs of low mass and high mass features
    #     # The first #self.take are the low mass samples (dl = data low)
    #     dl = data[:, :self.take]
    #     # The next #self.take are the high mass samples (dl = data low)
    #     dh = data[:, self.take:]
    #     # This returns the transformation we are after
    #     transformed = self.transform_to_data(dl, dh)
    #     # TODO: train with this
    #     transformed = self.transform_to_data(dh, dl)
    #     # Drop the mass feature from the high mass sample
    #     high_mass_features = dh[:, :-1]
    #     # Calculate the distance between the transformation and the high mass
    #     dists = self.dist_measure(transformed, high_mass_features)
    #     self.set_loss_dict([dists])
    #     return self.loss_dict[self.loss_names[0]]


class tucan(curtains_transformer):
    """Two way curtain transformer = tucan"""

    def __init__(self, INN, device, exp_name, dist_measure, nfeatures, dir='INN_test', det_beta=0, **kwargs):
        super(tucan, self).__init__(INN, device, exp_name, dist_measure, nfeatures, dir=dir, det_beta=det_beta,
                                    **kwargs)
        self.iter = 0
        self.set_loss_names()

    def set_loss_names(self):
        self.loss_names = ['Forward distance', 'Inverse distance']
        if self.det_beta > 0:
            self.loss_names += ['Forward detJ', 'Inverse DetJ']

    def compute_loss(self, data, batch_size):
        # The data is passed with concatenated pairs of low mass and high mass features
        # The first #self.take are the low mass samples (dl = data low)
        dl = data[:, :self.take]
        # The next #self.take are the high mass samples (dl = data low)
        dh = data[:, self.take:]

        # This returns the transformation from high mass to low mass
        transformed_lm, detJ_lm = self.inverse_transform_to_data(dl, dh)
        # This returns the transformation from low mass to high mass
        transformed_hm, detJ_hm = self.transform_to_data(dl, dh)
        # Drop the mass from the feature sample
        high_mass_features = dh[:, :-1]
        low_mass_features = dl[:, :-1]
        # Calculate the distance between the transformation and truth
        forward_dists = self.dist_measure(transformed_hm, high_mass_features)
        inverse_dists = self.dist_measure(transformed_lm, low_mass_features)
        losses = [forward_dists, inverse_dists]
        detJ_hm = detJ_hm.mean()
        detJ_lm = detJ_lm.mean()
        if self.det_beta > 0:
            losses += [detJ_hm, detJ_lm]
        self.set_loss_dict(losses)
        if self.iter:
            self.iter = 0
            return forward_dists - self.det_beta * detJ_hm
        else:
            self.iter = 1
            return inverse_dists - self.det_beta * detJ_lm


class delta_tucan(delta_curtains_transformer, tucan):
    def __init__(self, INN, device, exp_name, dist_measure, nfeatures, dir='INN_test', **kwargs):
        super(delta_tucan, self).__init__(INN, device, exp_name, dist_measure, nfeatures,
                                          dir=dir, **kwargs)


class delta_mass_tucan(delta_mass_curtains_transformer, tucan):
    def __init__(self, INN, device, exp_name, dist_measure, nfeatures, dir='INN_test', **kwargs):
        super(delta_mass_tucan, self).__init__(INN, device, exp_name, dist_measure, nfeatures,
                                               dir=dir, **kwargs)
