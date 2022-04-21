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
        if isinstance(self.base_model, nn.ModuleList):
            preds = []
            for model in self.base_model:
                preds += [model(data)]
            preds = torch.cat(preds, 1)
            return preds.mean(1).view(-1, 1)
            # Select classifiers at random to
            # inds = np.arange(len(preds)) * preds.shape[1] + np.random.randint(0, 5, len(preds))
            # return preds.flatten()[inds].view(-1, 1)
        else:
            return self.base_model(data)

    def predict(self, data):
        return self.activation(self.get_scores(data))

    def device(self):
        return next(self.parameters()).device

    def weighted_loss(self, prediction, target, weight):
        wl = target * -prediction.log() * weight + (1 - target) * -(1 - prediction).log()
        return wl.mean()

    def compute_loss(self, data, return_pred=False):
        inputs, target, weight = data
        if inputs.isnan().any():
            raise Exception('Inputs are NaNs.')
        device = self.device()
        prediction = self.predict(inputs.to(device))
        if prediction.isnan().any():
            raise Exception('Classifier has diverged.')
        self.loss = self.loss_object(prediction, target.to(device), weight=weight.to(device))
        # self.loss = self.weighted_loss(prediction, target.to(device), 0.8)
        if return_pred:
            return self.loss, prediction
        else:
            return self.loss

    def save(self, path):
        torch.save(self.base_model.state_dict(), path)

    def load(self, path):
        if isinstance(path, list):
            # TODO: this is a patently bad idea
            if isinstance(self.base_model, nn.ModuleList):
                self.base_model = self.base_model[0]
            self.base_model = nn.ModuleList(self.base_model for i in range(len(path)))
            [model.load_state_dict(torch.load(p)) for model, p in zip(self.base_model, path)]
        else:
            self.base_model.load_state_dict(torch.load(path))

    def round_sf(self, x, nsf):
        if x:
            x = round(x, nsf - (int(np.floor(np.log10(abs(x)))) - 1))
        return x

    def get_loss_state(self, nsf=10):
        return {self.loss_name: self.round_sf(self.loss.item(), nsf)}

    # Useful when making predictions that would ordinarily go out of memory
    def batch_predict(self, data_array, encode=False):
        store = []
        for data in data_array:
            if encode:
                store += [torch.cat(self.encode(data), 1)]
            else:
                store += [self(data)]
        return torch.cat(store)
