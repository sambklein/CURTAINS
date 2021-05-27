from torch.utils.data import Dataset
import torch
import numpy as np

from utils.io import on_cluster


class BasePhysics(Dataset):

    def __init__(self, data, scale=None):
        super(BasePhysics, self).__init__()
        self.data = data
        self.num_points = data.shape[0]
        self.scale_norm = 1
        self.normed = False
        self.set_scale(scale)
        # This isn't dynamic, so if you want to modify the size of the dataset, should have another method
        # TODO: what is the clean way to deal with this?
        self.shape = self.data.shape

    def set_scale(self, scale):
        if scale is None:
            # If no scaling variable is passed then this is the train set, so find the scaling vars
            self.max_vals = []
            self.min_vals = []
            for train_feature in self.data.t():
                self.max_vals += [train_feature.max()]
                self.min_vals += [train_feature.min()]
        else:
            self.max_vals = scale[0]
            self.min_vals = scale[1]

    def scale(self, scale_fact):
        # This will keep track of multiple scalings
        self.scale_norm *= scale_fact
        self.data *= scale_fact
        self.scaled = True

    def unscale(self, data_in=None):
        if data_in is None:
            data = self.data
        else:
            data = data_in
        data /= self.scale_norm
        if data_in is None:
            self.scale_norm = 1
        return data

    def normalize(self):
        for i, train_feature in enumerate(self.data.t()):
            min_val = self.min_vals[i]
            max_val = self.max_vals[i]
            zo = (train_feature - min_val) / (max_val - min_val)
            self.data.t()[i] = (zo - 0.5) * 2
        self.normed = True

    def unnormalize(self, data_in=None):

        data = self.unscale(data_in)

        if self.normed or (data_in is not None):
            temp = torch.empty_like(data)
            for i, train_feature in enumerate(data.t()):
                min_val = self.min_vals[i]
                max_val = self.max_vals[i]
                zo = train_feature / 2 + 0.5
                temp.t()[i] = zo * (max_val - min_val) + min_val
            data = temp
            if data_in is None:
                self.normed = False
        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]


class Curtains(BasePhysics):
    def __init__(self, df, norm=None, dtype=torch.float32):
        self.df = df
        data, feature_nms = self.get_features(df)
        self.feature_nms = feature_nms
        super(Curtains, self).__init__(torch.tensor(data).type(dtype), scale=norm)

    @staticmethod
    def get_features(df):
        if on_cluster():
            # TODO: better handling of this for outputting names and selecting features in one place - list of lists?
            nfeatures = 4
            data = np.zeros((df.shape[0], nfeatures + 1))
            # The last data feature is always the context
            # 'pt', 'eta', 'phi', 'mass', 'tau1', 'tau2', 'tau3', 'd12', 'd23', 'ECF2', 'ECF3'
            data[:, 0] = df['tau2'] / df['tau1']
            data[:, 1] = df['tau3'] / df['tau2']
            data[:, 2] = np.log(df['d23'] + 1)
            data[:, 3] = np.log(df['d12'] + 1)
            data[:, 4] = df['mass']
            return data, ['tau2s/taus', 'tau3s/tau2s', 'd23', 'd12', 'mass']
        else:
            nfeatures = 4
            data = np.zeros((df.shape[0], nfeatures + 1))
            data[:, 0] = df['Qws']
            data[:, 1] = df['tau2s'] / df['taus']
            data[:, 2] = df['tau3s'] / df['tau2s']
            data[:, 3] = df['d34s']
            data[:, 4] = df['m']
            return data, ['Qws', 'tau2s/taus', 'tau3s/tau2s', 'd34s', 'm']

    def get_quantile(self, quantile):
        # Returns a numpy array of the training features, plus the context feature on the end
        features = self.data[:, :-1]
        mx = self.get_quantile_mask(quantile)
        return features[mx]

    def get_quantile_mask(self, quantile):
        mx = self.df['mass_q{}'.format(quantile)]
        return np.array(mx, dtype='bool')


class CurtainsTrainSet(Dataset):

    def __init__(self, data1, data2, mix_qs=False, stack=False):
        self.data1 = data1
        self.data2 = data2
        self.s1 = self.data1.shape[0]
        self.s2 = self.data2.shape[0]
        self.ndata = min(self.s1, self.s2)
        self.mix_qs = mix_qs
        self.stack = stack
        self.data = self.get_data()
        self.shape = [self.ndata, *self.data1.shape[1:]]

    def get_data(self):
        if self.stack:
            # This method keeps the high mass and low mass regions separated
            data = torch.cat((self.data1.data, self.data2.data), 0)
            return data[torch.randperm(self.s1 + self.s2, device='cpu')]
        else:
            if self.mix_qs:
                # This method will shuffle samples between classes
                # With this you also learn to map within the same class
                d1 = self.data1[torch.randperm(self.s1, device='cpu')]
                d2 = self.data2[torch.randperm(self.s2, device='cpu')]
                data = torch.cat((d1[:self.ndata].data, d2[:self.ndata].data), 0)
                data_shuffled = data[torch.randperm(data.shape[0], device='cpu')]
                data = torch.cat((data_shuffled[:self.ndata], data_shuffled[self.ndata:]), 1)
                # Sort so that map goes from low mass to high mass
                ndf = self.data1.data.shape[1]
                data2 = torch.cat((data_shuffled[self.ndata:], data_shuffled[:self.ndata]), 1)
                data = torch.where(data[:, ndf - 1] < data[:, -1], data.t(), data2.t()).t()
                return data
            else:
                # This method keeps the high mass and low mass regions separated
                d1 = self.data1[torch.randperm(self.s1, device='cpu')]
                d2 = self.data2[torch.randperm(self.s2, device='cpu')]
                return torch.cat((d1[:self.ndata].data, d2[:self.ndata].data), 1)

    def set_norm_fact(self, scale):
        self.norm_fact = scale
        self.data1.set_scale(scale)
        self.data2.set_scale(scale)

    def set_and_get_norm_facts(self):
        # Set the scale for each feature using the combined datasets
        scale = [self.data1.max_vals, self.data1.min_vals]
        scale1 = [self.data2.max_vals, self.data2.min_vals]
        upperbound = np.where([s[0] < s[1] for s in zip(scale[0], scale1[0])], scale1[0], scale[0])
        lowerbound = np.where([s[0] < s[1] for s in zip(scale[1], scale1[1])], scale[1], scale1[1])

        def glist(array):
            return [torch.tensor(i) for i in array]

        s = [glist(upperbound), glist(lowerbound)]
        self.set_norm_fact(s)
        return s

    def scale(self, sf):
        self.data1.scale(sf)
        self.data2.scale(sf)
        self.data = self.get_data()

    def normalize(self):
        self.data1.normalize()
        self.data2.normalize()
        self.data = self.get_data()

    def shuffle(self):
        device = self.data.device
        self.data = self.get_data().to(device)

    def copy_construct(self, inds):
        # At present this does not need to be more detailed, we don't care about the scaling properties while training
        # TODO should add a copyconstruct to the base model and the Curtains method to make this easier
        dataset = CurtainsTrainSet(Curtains(self.data1.df.iloc[inds]), Curtains(self.data2.df.iloc[inds]))
        dataset.set_norm_fact(self.norm_fact)
        if self.data1.normed:
            dataset.normalize()
        # Manually place on the same device
        dataset.data = dataset.data.to(self.data.device)
        return dataset

    def get_valid(self, inds_valid, inds_train):
        return self.copy_construct(inds_train), self.copy_construct(inds_valid)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]


class WrappingCurtains():
    """
    This is just for wrapping three curtains datasets into one for ease of use, handling them all, and scaling etc
    """

    def __init__(self, trainset, signalset, validationset, bins, scale=None):
        self.scale = scale
        self.trainset = trainset
        self.signalset = signalset
        self.validationset = validationset
        self.scale_data()
        self.bins = bins

        self.ndata = self.trainset.data.shape[0]
        # The last feature of the dataset is the context
        self.nfeatures = self.signalset.shape[1] - 1
        self.feature_nms = self.signalset.feature_nms

    def scale_data(self):
        if self.scale:
            self.trainset.scale(self.scale)
            self.signalset.scale(self.scale)
            self.validationset.scale(self.scale)

    def normalize(self):
        self.trainset.normalize()
        self.signalset.normalize()
        self.validationset.normalize()

    def unnormalize(self):
        self.trainset.unnormalize()
        self.signalset.unnormalize()
        self.validationset.unnormalize()


# TODO use the base class definition here as well
class JetsDataset(Dataset):
    # TODO: for the time being this is just the leading and subleading jet four momenta

    def __init__(self, lo_obs, nlo_obs, lo_const, nlo_const, scale=None):
        self.data = torch.cat((torch.tensor(lo_obs), torch.tensor(nlo_obs)), 1)
        self.lo_obs = torch.tensor(lo_obs)
        self.nlo_obs = torch.tensor(nlo_obs)
        self.lo_const = torch.tensor(lo_const)
        self.nlo_const = torch.tensor(nlo_const)
        self.num_points = lo_obs.shape[0]
        if scale == None:
            # If no scaling variable is passed then this is the train set, so find the scaling vars
            self.max_vals = []
            self.min_vals = []
            for train_feature in self.data.t():
                self.max_vals += [train_feature.max()]
                self.min_vals += [train_feature.min()]
        else:
            self.max_vals = scale[0]
            self.min_vals = scale[1]

    def normalize(self):
        for i, train_feature in enumerate(self.data.t()):
            min_val = self.min_vals[i]
            max_val = self.max_vals[i]
            zo = (train_feature - min_val) / (max_val - min_val)
            self.data.t()[i] = (zo - 0.5) * 2

    def unnormalize(self, data=None):

        if data == None:
            data = self.data

        for i, train_feature in enumerate(data.t()):
            min_val = self.min_vals[i]
            max_val = self.max_vals[i]
            zo = train_feature * 2 + 0.5
            data.t()[i] = zo * (max_val - min_val) + min_val
        return data

    def __len__(self):
        return self.num_points

    def __getitem__(self, item):
        return self.data[item]
