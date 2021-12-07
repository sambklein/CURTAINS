import warnings

from torch.utils.data import Dataset
import torch
import numpy as np

from utils.io import on_cluster


class BaseData(Dataset):
    def deal_with_nans(self):
        nan_mask = self.data.isnan().any(1)
        num_nans = nan_mask.sum()
        if num_nans > 0:
            self.data = self.data[~nan_mask]
            warnings.warn(f'There are {num_nans} samples with NaN values. These have been dropped.');
        # This isn't dynamic, so if you want to modify the size of the dataset this must be updated as well
        self.shape = self.data.shape

    def normalize(self):
        for i, train_feature in enumerate(self.data.t()):
            min_val = self.min_vals[i]
            max_val = self.max_vals[i]
            zo = (train_feature - min_val) / (max_val - min_val)
            self.data.t()[i] = (zo - 0.5) * 2
        self.normed = True
        self.deal_with_nans()

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


class BasePhysics(BaseData):

    def __init__(self, data, scale=None):
        super(BasePhysics, self).__init__()
        self.data = data
        self.num_points = data.shape[0]
        self.scale_norm = 1
        self.normed = False
        self.deal_with_nans()
        self.set_scale(scale)

    def set_scale(self, scale):
        if scale is None:
            # If no scaling variable is passed then this is the train set, so find the scaling vars
            self.max_vals = self.data.quantile(0.99, 0)
            self.min_vals = self.data.quantile(0.01, 0)
            # self.max_vals = []
            # self.min_vals = []
            # for train_feature in self.data.t():
            #     # self.max_vals += [train_feature.max()]
            #     # self.min_vals += [train_feature.min()]
            #     self.max_vals += [train_feature.quantile(0.9999)]
            #     self.min_vals += [train_feature.quantile(0.0001)]
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
        return df.to_numpy(), list(df.keys())
        # if features == 0:
        #     # TODO: better handling of this for outputting names and selecting features in one place - list of lists?
        #     nfeatures = 4
        #     data = np.zeros((df.shape[0], nfeatures + 1))
        #     # The last data feature is always the context
        #     # 'pt', 'eta', 'phi', 'mass', 'tau1', 'tau2', 'tau3', 'd12', 'd23', 'ECF2', 'ECF3'
        #     data[:, 0] = df['tau2'] / df['tau1']
        #     data[:, 1] = df['tau3'] / df['tau2']
        #     data[:, 2] = np.log(df['d23'] + 1)
        #     data[:, 3] = np.log(df['d12'] + 1)
        #     data[:, 4] = df['mass']
        #     return data, [r'$\tau_{21}$', r'$\tau_{32}$', r'$d_{23}$', r'$d_{12}$', 'mass']
        #
        # else:
        #     # nfeatures = 9
        #     # data = np.zeros((df.shape[0], nfeatures + 1))
        #     # # The last data feature is always the context
        #     # # 'pt', 'eta', 'phi', 'mass', 'tau1', 'tau2', 'tau3', 'd12', 'd23', 'ECF2', 'ECF3'
        #     # data[:, 0] = df['pt'] + df['nlo_pt']
        #     # data[:, 1] = df['mass'] + df['nlo_mass']
        #     # data[:, 2] = df['pt']
        #     # data[:, 3] = df['nlo_pt']
        #     # data[:, 4] = df['mass']
        #     # data[:, 5] = df['nlo_mass']
        #     # phi_1 = df['phi']
        #     # phi_2 = df['nlo_phi']
        #     # delPhi = np.arctan2(np.sin(phi_1 - phi_2), np.cos(phi_1 - phi_2))
        #     # data[:, 6] = ((df['eta'] - df['nlo_eta']) ** 2 + delPhi ** 2) ** (0.5)
        #     # data[:, 7] = df['ptjj']
        #     # data[:, 8] = df['ptjj'] * data[:, 6]
        #     # # data[:, 2] = np.log(df['d23'] + 1)
        #     # # data[:, 3] = np.log(df['d12'] + 1)
        #     # data[:, 9] = df['mjj']
        #     # return data, [r'$p_t + p_t$', r'$m + m$', r'$p_t$', r'$nlo p_t$', 'mass', 'nlo mass',
        #     #                r'$dR_{jj}$', r'$p_{t, jj}$', r'$p_{t, jj} * dR_{jj}$', r'$m_{JJ}$']
        #     nfeatures = 4
        #     data = np.zeros((df.shape[0], nfeatures + 1))
        #     # The last data feature is always the context
        #     # 'pt', 'eta', 'phi', 'mass', 'tau1', 'tau2', 'tau3', 'd12', 'd23', 'ECF2', 'ECF3'
        #     data[:, 0] = df['pt'] + df['nlo_pt']
        #     data[:, 1] = df['pt']
        #     data[:, 2] = df['nlo_pt']
        #     phi_1 = df['phi']
        #     phi_2 = df['nlo_phi']
        #     delPhi = np.arctan2(np.sin(phi_1 - phi_2), np.cos(phi_1 - phi_2))
        #     data[:, 3] = ((df['eta'] - df['nlo_eta']) ** 2 + delPhi ** 2) ** (0.5)
        #     data[:, 4] = df['mjj']
        #     return data, [r'$p_t + p_t$', r'$p_t$', r'$nlo p_t$',
        #                   r'$dR_{jj}$',  r'$m_{JJ}$']

    def get_quantile(self, quantile):
        # Returns a numpy array of the training features, plus the context feature on the end
        features = self.data[:, :-1]
        mx = self.get_quantile_mask(quantile)
        return features[mx]

    def get_quantile_mask(self, quantile):
        mx = self.df['mass_q{}'.format(quantile)]
        return np.array(mx, dtype='bool')

    def unnorm_mass(self, mass):
        not_tensor = not torch.is_tensor(mass)
        if not_tensor:
            mass = torch.tensor(mass)
        # mass = torch.tensor(mass)
        min_val = self.min_vals[-1]
        max_val = self.max_vals[-1]
        zo = mass / 2 + 0.5
        mass = zo * (max_val - min_val) + min_val
        if not_tensor:
            mass = mass.detach().cpu().numpy()
        return mass

    def norm_mass(self, mass):
        not_tensor = not torch.is_tensor(mass)
        if not_tensor:
            dtype = type(mass)
            mass = torch.tensor(mass)
        min_val = self.min_vals[-1]
        max_val = self.max_vals[-1]
        zo = (mass - min_val) / (max_val - min_val)
        mass = (zo - 0.5) * 2
        if not_tensor:
            mass = dtype(mass)
        return mass


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
            if self.mix_qs == 1:
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

            elif self.mix_qs == 2:
                def mix_side_bands(band):
                    masses = band[:, -1]
                    max_mass = masses.max()
                    min_mass = masses.min()
                    split = min_mass + (max_mass - min_mass) / 2
                    # split = masses.mean()
                    # split = masses.median()
                    lm = band[masses < split]
                    hm = band[masses > split]
                    ntake = min(len(lm), len(hm))
                    return torch.cat((lm[:ntake], hm[:ntake]), 1)

                d1 = self.data1[torch.randperm(self.s1, device='cpu')]
                d2 = self.data2[torch.randperm(self.s2, device='cpu')]
                data = torch.cat((mix_side_bands(d1), mix_side_bands(d2)), 0)
                self.mix_qs = 0
            elif self.mix_qs == 0:
                # This method keeps the high mass and low mass regions separated
                d1 = self.data1[torch.randperm(self.s1, device='cpu')]
                d2 = self.data2[torch.randperm(self.s2, device='cpu')]
                data = torch.cat((d1[:self.ndata].data, d2[:self.ndata].data), 1)
                self.mix_qs = 2
            
            elif self.mix_qs == 3:
                # This method keeps the high mass and low mass regions separated
                d1 = self.data1[torch.randperm(self.s1, device='cpu')]
                d2 = self.data2[torch.randperm(self.s2, device='cpu')]
                data = torch.cat((d1[:self.ndata].data, d2[:self.ndata].data), 1)
            return data

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
        dataset = CurtainsTrainSet(Curtains(self.data1.df.iloc[inds]),
                                   Curtains(self.data2.df.iloc[inds]),
                                   mix_qs=self.mix_qs, stack=self.stack)
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

    def __init__(self, trainset, signalset, validationset, validationset_lm, bins, scale=None):
        self.scale = scale
        self.trainset = trainset
        self.signalset = signalset
        self.validationset = validationset
        self.validationset_lm = validationset_lm
        self.scale_data()
        self.bins = bins
        self.mass_bins = torch.tensor(bins)

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
        self.validationset_lm.normalize()
        self.mass_bins = self.validationset_lm.norm_mass(torch.tensor(self.bins))

    def unnormalize(self):
        self.trainset.unnormalize()
        self.signalset.unnormalize()
        self.validationset.unnormalize()
        self.validationset_lm.unnormalize()
        self.mass_bins = torch.tensor(self.bins)


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
