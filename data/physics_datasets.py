from torch.utils.data import Dataset
import torch


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
