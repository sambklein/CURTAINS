from torch.utils.data import Dataset
import torch
import numpy as np


class HepmassDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # TODO: should probably save a reference to this?
        # self.dataframe = dataframe
        self.data = torch.tensor(dataframe.to_numpy(dtype='float32'))
        self.transform = transform
        self.dimension = len(dataframe.columns)
        self.num_points = len(dataframe.index)

    def __len__(self):
        return self.num_points

    def __getitem__(self, item):
        return self.data[item]


class BasePhysics(Dataset):
    # TODO: need to sort out scaling and normalizing

    def __init__(self, data, scale=None):
        self.data = data
        self.num_points = data.shape[0]
        self.scale = scale
        self.set_scale(scale)

    def set_scale(self, scale):
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


class Curtains(BasePhysics):
    def __init__(self, df, dtype=torch.float32):
        # TODO: memory footprint of this?
        self.df = df
        data = self.get_features(df)
        super(Curtains, self).__init__(torch.tensor(data).type(dtype))

    @staticmethod
    def get_features(df):
        nfeatures = 5
        data = np.zeros((df.shape[0], nfeatures + 1))
        # The last data feature is always the context, TODO: this could/should be handled by the data class?
        data[:, 0] = df['tau3s'] / df['taus']
        data[:, 1] = df['tau3s'] / df['tau2s']
        data[:, 2] = df['Qws']
        data[:, 3] = df['d34s']
        data[:, 5] = df['m']
        return data


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

        self.ndata = self.trainset.shape[0]
        # The last feature of the dataset is the context
        self.nfeatures = self.trainset.shape[1] - 1

    # TODO: Careful with the scaling/unscaling and normalize/unnormalize
    def scale_data(self):
        if self.scale:
            self.trainset.data *= self.scale
            self.signalset.data *= self.scale
            self.validationset.data *= self.scale

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
