import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, base_model, nfeatures, nclasses, exp_name, loss_object=F.binary_cross_entropy,
                 loss_name='Classification Error', directory='BasicClassifier', activation=nn.Identity()):
        super(Classifier, self).__init__()
        self.base_model = base_model(nfeatures, nclasses)
        self.loss_object = loss_object
        self.loss_name = loss_name
        self.exp_name = exp_name
        self.directory = directory
        self.activation = activation

    def forward(self, data):
        return self.predict(data)

    def get_scores(self, data):
        return self.base_model(data)

    def predict(self, data):
        return self.activation(self.get_scores(data))

    def device(self):
        return next(self.parameters()).device

    def compute_loss(self, data):
        inputs, target, weight = data
        if inputs.isnan().any():
            raise Exception('Inputs are NaNs.')
        device = self.device()
        prediction = self.predict(inputs.to(device))
        if prediction.isnan().any():
            raise Exception('Classifier has diverged.')
        self.loss = self.loss_object(prediction, target.to(device), weight=weight.to(device))
        return self.loss

    def save(self, path):
        torch.save(self.base_model.state_dict(), path)

    def load(self, path):
        self.base_model.load_state_dict(torch.load(path))

    def round_sf(self, x, nsf):
        if x:
            x = round(x, nsf - (int(np.floor(np.log10(abs(x)))) - 1))
        return x

    def get_loss_state(self, nsf=10):
        return {self.loss_name: self.round_sf(self.loss.item(), nsf)}

    # Useful when making predictions that would ordinarily go out of memory
    def batch_predict(self, data_array, encode=False):
        # TODO: this should accept both batch and unbatched data and maybe pass large batches through a dataloader
        store = []
        for data in data_array:
            if encode:
                store += [torch.cat(self.encode(data), 1)]
            else:
                store += [self(data)]
        return torch.cat(store)
