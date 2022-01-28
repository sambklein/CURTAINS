import torch

from utils.plotting import hist_features, get_windows_plot, add_contour, kde_plot, plot_delta_mass
from utils.torch_utils import tensor2numpy, shuffle_tensor
from .physics_datasets import JetsDataset, WrappingCurtains, Curtains, CurtainsTrainSet

import glob
import os
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob

# Taken from https://github.com/bayesiains/nsf/blob/master/data/base.py
from utils.io import get_top_dir, on_cluster, make_slim


def load_jets(sm='QCD', split=0.1, normalize=True, dtype='float32'):
    """
    This will return a training and a test dataset for the JETS dataset generated for RODEM.
    :param sm: The type of the jet, QCD, WZ or tt
    :param split: The fraction of the events to take as a test set
    :return:
    """

    dir = '/srv/beegfs/scratch/groups/rodem/AnomalyDetection/HEP/jets/'

    with h5py.File(dir + "{}dijet100k.h5".format(sm), 'r') as readfile:
        # with h5py.File(dir + "output.h5", 'r') as readfile:
        lo_obs = np.array(readfile["objects/jets/lo_obs"][:], dtype=dtype)
        nlo_obs = np.array(readfile["objects/jets/nlo_obs"][:], dtype=dtype)
        lo_const = np.array(readfile["objects/jets/lo_constituents"][:], dtype=dtype)
        nlo_const = np.array(readfile["objects/jets/nlo_constituents"][:], dtype=dtype)

    if split == 0:
        return JetsDataset(lo_obs, nlo_obs, lo_const, nlo_const)
    nevents = lo_obs.shape[0]
    nsample = int(split * nevents)
    indices = np.random.choice(nevents, size=nsample, replace=False)
    objects = (lo_obs, nlo_obs, lo_const, nlo_const)
    trainset = JetsDataset(*[np.delete(array, indices, 0) for array in objects])
    testset = JetsDataset(*[array[indices, :] for array in objects], scale=[trainset.max_vals, trainset.min_vals])

    if normalize:
        trainset.normalize()
        testset.normalize()
        print(trainset.data.max())
        print(testset.data.max())

    return trainset, testset


# TODO: this is stupid code duplication from the pytorch-utils repo in the anomaly tools.
def fill_array(to_fill, obj, dtype):
    arr = np.array(obj, dtype=dtype)
    to_fill[:len(arr)] = arr


def convert_from_cylindrical(data):
    data = data.to_numpy()
    constPx = data[:, 0] * np.cos(data[:, 2])
    constPy = data[:, 0] * np.sin(data[:, 2])
    constPz = data[:, 0] * np.sinh(data[:, 1])

    three_vector = np.column_stack((constPx, constPy, constPz))
    jetP2 = np.sum(three_vector ** 2, axis=1)
    constE = np.sqrt(jetP2 + data[:, 3] ** 2)

    JetinExyz = np.hstack((constE.reshape(-1, 1), three_vector))
    return JetinExyz


def calculate_mass(four_vector):
    return (four_vector[:, 0] ** 2 - np.sum(four_vector[:, 1:4] ** 2, axis=1)) ** 0.5


def get_dijet_features(df):
    fv1 = convert_from_cylindrical(df.iloc[:, :4])
    fv2 = convert_from_cylindrical(df.iloc[:, 11:15])
    tot_fv = fv1 + fv2
    mjj = calculate_mass(tot_fv)
    ptjj = (tot_fv[:, 1] ** 2 + tot_fv[:, 2] ** 2) ** (0.5)
    return mjj, ptjj


def load_curtains_pd(sm='QCDjj_pT', dtype='float32', extraStats=False, feature_type=0):
    if on_cluster() and (feature_type < 2):

        directory = '/srv/beegfs/scratch/groups/rodem/anomalous_jets/data/'

        if extraStats:
            if sm[:3] == 'QCD':
                files = glob.glob(directory + f"20210629_{sm}_450_1200_nevents_10M/*.h5")
            else:
                files = glob.glob(directory + f'20210430_{sm}_450_1200_nevents_1M/*.h5')
        else:
            files = glob.glob(directory + f'20210430_{sm}_450_1200_nevents_1M/*.h5')

        nchunks = len(files)
        lo_obs = np.empty((nchunks, 224570, 11))
        nlo_obs = np.empty((nchunks, 224570, 11))

        for i in range(nchunks):
            with h5py.File(files[i], 'r') as readfile:
                fill_array(lo_obs[i], readfile["objects/jets/jet1_obs"][:], dtype)
                fill_array(nlo_obs[i], readfile["objects/jets/jet2_obs"][:], dtype)

        low_level_names = ['pt', 'eta', 'phi', 'mass', 'tau1', 'tau2', 'tau3', 'd12', 'd23', 'ECF2', 'ECF3']
        lo_obs = np.vstack(lo_obs)
        mx = lo_obs[:, 0] != 0
        df = pd.DataFrame(np.hstack((lo_obs[mx], np.vstack(nlo_obs)[mx])),
                          columns=low_level_names + ['nlo_' + nm for nm in low_level_names])

        slim_file = get_top_dir() + f'/data/slims/{sm}.csv'
        if not os.path.isfile(slim_file):
            if not os.path.isfile(slim_file):
                df_sv = df.take(list(range(10000)))
                os.makedirs(get_top_dir() + '/data/slims/', exist_ok=True)
                df_sv.to_csv(slim_file, index=False)

        mjj, ptjj = get_dijet_features(df)
        df['mjj'] = mjj
        df['ptjj'] = ptjj
        # return df

    elif feature_type < 2:
        slim_dir = get_top_dir() + '/data/slims'
        filename = f'{sm}.csv'
        slim_file = slim_dir + '/' + filename
        df = pd.read_csv(slim_file)
        mjj, ptjj = get_dijet_features(df)
        df['mjj'] = mjj
        df['ptjj'] = ptjj
        # return df

    if feature_type == 0:
        data = pd.DataFrame()
        data[r'$\tau_{21}$'] = df['tau2'] / df['tau1']
        data[r'$\tau_{32}$'] = df['tau3'] / df['tau2']
        data[r'$d_{23}$'] = df['tau3'] / df['tau2']
        data[r'$d_{12}$'] = np.log(df['d12'] + 1)
        data[r'$d_{32}$'] = np.log(df['d23'] + 1)
        data['mass'] = np.log(df['d12'] + 1)

    elif feature_type == 1:
        data = pd.DataFrame()
        data[r'$p_t + p_t$'] = df['pt'] + df['nlo_pt']
        data[r'$p_t$'] = df['pt']
        data[r'$nlo p_t$'] = df['nlo_pt']
        phi_1 = df['phi']
        phi_2 = df['nlo_phi']
        delPhi = np.arctan2(np.sin(phi_1 - phi_2), np.cos(phi_1 - phi_2))
        data[r'$dR_{jj}$'] = ((df['eta'] - df['nlo_eta']) ** 2 + delPhi ** 2) ** (0.5)
        data[r'$m_{JJ}$'] = df['mjj']

    elif feature_type >= 2:
        if on_cluster():
            directory = '/srv/beegfs/scratch/groups/rodem/LHCO'
        else:
            directory = 'data/downloads'
        lhco_filename = 'events_anomalydetection_v2.features.h5'
        df = pd.read_hdf(f'{directory}/{lhco_filename}')
        # make_slim(df, directory, lhco_filename)

        # Reorder the features such that the jets are ordered according to their invariant masses
        jet_order_mask = df['mj1'] < df['mj2']
        inverted_keys = ['pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2', 'tau2j2', 'tau3j2', 'pxj1', 'pyj1', 'pzj1', 'mj1',
                         'tau1j1', 'tau2j1', 'tau3j1', 'label']
        proper_order = df.loc[jet_order_mask]
        improper_order = df.loc[~jet_order_mask]
        improper_order.columns = inverted_keys
        df = pd.concat((proper_order, improper_order))

        if sm == 'QCDjj_pT':
            df = df.loc[df['label'] == 0]
        else:
            df = df.loc[df['label'] == 1]

        for jet in ['j1', 'j2']:
            df[f'pt{jet}'] = np.sqrt(df[f'px{jet}'] ** 2 + df[f'py{jet}'] ** 2)
            df[f'eta{jet}'] = np.arcsinh(df[f'pz{jet}'] / df[f'pt{jet}'])
            df[f'phi{jet}'] = np.arctan2(df[f'py{jet}'], df[f'px{jet}'])
            df[f'p{jet}'] = np.sqrt(df[f'pz{jet}'] ** 2 + df[f'pt{jet}'] ** 2)
            df[f'e{jet}'] = np.sqrt(df[f'm{jet}'] ** 2 + df[f'p{jet}'] ** 2)

        # TODO: remove this?
        fig, axs_ = plt.subplots(2, 4, figsize=(22, 12))
        tplot = df[['ptj1', 'etaj1', 'phij1', 'ej1', 'ptj2', 'etaj2', 'phij2', 'ej2']]
        for i, ax in enumerate(fig.axes):
            ax.hist(tplot.iloc[:, i], alpha=0.5, density=True, bins=50, histtype='step')
            ax.set_title(tplot.keys()[i])
        fig.savefig('test')

        data = df[['mj1', 'mj2']].copy()
        # data['mj2-mj1'] = abs(df['mj2'] - df['mj1'])
        # data = pd.DataFrame(np.sort(df[['mj1', 'mj2']].to_numpy(), 1), columns=['mj1', 'mj2'], index=df.index)
        data['mj2-mj1'] = data['mj2'] - data['mj1']
        data[r'$\tau_{21}^{j_1}$'] = df['tau2j1'] / df['tau1j1']
        data[r'$\tau_{32}^{j_1}$'] = df['tau3j1'] / df['tau2j1']
        data[r'$\tau_{21}^{j_2}$'] = df['tau2j2'] / df['tau1j2']
        data[r'$\tau_{32}^{j_2}$'] = df['tau3j2'] / df['tau2j2']
        # data = pd.DataFrame()
        data[r'$p_t^{j_1}$'] = df['ptj1']
        data[r'$p_t^{j_2}$'] = df['ptj2']
        phi_1 = df['phij1']
        phi_2 = df['phij2']
        delPhi = np.arctan2(np.sin(phi_1 - phi_2), np.cos(phi_1 - phi_2))
        data[r'$dR_{jj}$'] = ((df['etaj1'] - df['etaj2']) ** 2 + delPhi ** 2) ** (0.5)

        data['delPhi'] = abs(delPhi)
        data['delEta'] = abs(df['etaj1'] - df['etaj2'])

        data['mjj'] = calculate_mass(
            np.sum([df[[f'ej{i}', f'pxj{i}', f'pyj{i}', f'pzj{i}']].to_numpy() for i in range(1, 3)], 0))

        # TODO: This needs to be handled by the data class!! It needs to be undone for classification!!
        # def scale_features(data):
        #     eps = 1e-6
        #     mn = data.min() - eps
        #     mx = data.max() + eps
        #     data = (data - mn) / (mx - mn)
        #     data = np.log(data) - np.log(1 - data)
        #     return data
        def scale_features(data):
            return data

        data.iloc[:, :-1] = scale_features(data.iloc[:, :-1])

        if feature_type == 2:
            data = data[['mj1', 'mj2', r'$dR_{jj}$', 'mjj']]

        if feature_type == 10:
            data = data[['mj1', 'mj2-mj1', r'$\tau_{21}^{j_1}$', r'$\tau_{21}^{j_2}$', r'$dR_{jj}$', r'$p_t^{j_1}$',
                         r'$p_t^{j_2}$', 'mjj']]

        if feature_type == 11:
            data = data[['mj1', 'mj2-mj1', r'$\tau_{21}^{j_1}$', r'$\tau_{21}^{j_2}$', r'$dR_{jj}$', r'$p_t^{j_1}$',
                         r'$p_t^{j_2}$', 'delEta', 'mjj']]

        if feature_type == 12:
            data = data[['mj1', 'mj2-mj1', r'$\tau_{21}^{j_1}$', r'$\tau_{21}^{j_2}$', 'mjj']]

        if feature_type == 13:
            data = data[['mj1', 'mj2-mj1', r'$\tau_{21}^{j_1}$', r'$\tau_{21}^{j_2}$', 'mjj']]
            data['gauss'] = np.random.normal(size=data['mj1'].shape[0])

        if 3 <= feature_type <= 8:
            # Introduce successive spurious correlations
            mJJ = data.iloc[:, -1]
            mJJ_normed = (mJJ - min(mJJ)) / (max(mJJ) - min(mJJ))
            if 4 <= feature_type:
                data.iloc[:, 0] = data.iloc[:, 0] * mJJ
            if 5 <= feature_type:
                data.iloc[:, 1] = data.iloc[:, 1] / mJJ

            def scale_mJJ(feature):
                return mJJ_normed * (max(feature) - min(feature)) + min(feature)

            if 6 <= feature_type:
                data.iloc[:, 3] = data.iloc[:, 3] + scale_mJJ(data.iloc[:, 3]) / 2
            if 7 <= feature_type:
                data.iloc[:, 4] = data.iloc[:, 4] - scale_mJJ(data.iloc[:, 4])

            if feature_type <= 7:
                data = data[['mj1', 'mj2-mj1', r'$\tau_{21}^{j_1}$', r'$\tau_{21}^{j_2}$', r'$dR_{jj}$', 'mjj']]

        # Defines features 3-9

        # for feature in ['delPhi', 'delEta']:
        # # for feature in [r'$dR_{jj}$', 'delPhi', 'delEta']:
        #     fig, axs_ = plt.subplots(1, 2, figsize=(20, 8))
        #     axs_[0].hist(data[feature], alpha=0.5, density=True, bins=50, histtype='step')
        #     # axs_[1].hist2d(data[r'$dR_{jj}$'], data['mjj'], alpha=0.5, density=True, bins=50)
        #     sns.kdeplot(data[feature][:10000], y=data['mjj'], ax=axs_[1], alpha=0.4, levels=10, color='red', fill=True)
        #     fig.savefig(feature)

        # if not on_cluster():
        #     data = data.sample(frac=1)
        #     data = data.sample(100000)

    return data


def load_curtains():
    df = load_curtains_pd()
    return Curtains(df)


def drop_redundant_df(df, context_feature, cutoff):
    context_df = df[context_feature]
    df = df.loc[(context_df > cutoff)]
    return df


def get_bin(process, bin, trainset=None, normalize=True):
    df = load_curtains_pd(sm=process)
    context_feature = df['mass']
    df = df.loc[(context_feature < bin[1]) & (context_feature > bin[0])]
    if trainset is None:
        return df['mass']
        # return df
    else:
        data = Curtains(df, norm=[trainset.max_vals, trainset.min_vals])
        if normalize:
            data.normalize()
        return data


def dope_dataframe(undoped, anomaly_data, doping):
    n = int(len(anomaly_data) * doping)
    if len(anomaly_data) < n:
        raise Exception('Not enough anomalies in this region for this level of doping.')
    if len(anomaly_data) != 0:
        anomaly_data = anomaly_data.sample(frac=1)
        mixing_anomalies = anomaly_data.iloc[:n]
        anomaly_data = anomaly_data.iloc[n:]
        df = pd.concat((mixing_anomalies, undoped), 0)
        df = df.sample(frac=1)
        return anomaly_data, mixing_anomalies, df
    else:
        pass


def mask_dataframe(df, context_feature, bins, indx, doping=0., anomaly_data=None):
    def mx_data(data):
        context_df = data[context_feature]
        mx = (context_df >= bins[indx[0]]) & (context_df < bins[indx[1]])
        return data.loc[mx]

    undoped_df = mx_data(df)
    anomaly_data = mx_data(anomaly_data)

    remaining_anomalies, mixed_anomalies, df = dope_dataframe(undoped_df, anomaly_data, doping)

    return remaining_anomalies, mixed_anomalies, df, len(mixed_anomalies) / len(undoped_df)


def filter_mix_data(df_bg, df_anomaly, edges, context):
    u_context_df = df_bg[context]
    u_mx = ((u_context_df > edges[0]) & (u_context_df < edges[1]))

    a_context_df = df_anomaly[context]
    a_mx = ((a_context_df > edges[0]) & (a_context_df < edges[1]))

    mix = pd.concat([df_bg.loc[u_mx], df_anomaly.loc[a_mx]])
    mix = mix.sample(frac=1)
    return mix, len(df_anomaly.loc[a_mx]) / len(df_bg.loc[u_mx])


def binwise_mixing(df, anomaly, context, bins):
    reg = ["OB1", "SB1", "SR", "SB2", "OB2"]
    data = {}
    for edge1, edge2, window in zip(bins, bins[1:], reg):
        data[window] = filter_mix_data(df, anomaly, (edge1, edge2), context)
    return data


def dope_dataframe_new(undoped, anomaly_data, doping, bins, context):
    '''
    returns the mixed in anomalies, holdout anomalies, and doped dataframe and sig frac in OB1-SB1-SR-SB2-OB2
    
    args:
        undoped -> pandas df for the bg spectra.
        anomaly_data -> pandas df for the sig spectra.
        doping -> integer for how many signal samples to take and mix in.
        bins -> list for bin edge definition for ob1 through ob2.
        context -> df key for the context feature, Usually mjj or mass.
    '''

    anomaly_data = anomaly_data.sample(frac=1)
    mixed_in_anomaly = anomaly_data.iloc[:doping]
    holdout_anomaly = anomaly_data.iloc[doping:]

    package_data = binwise_mixing(undoped, mixed_in_anomaly, context, bins)

    return holdout_anomaly, mixed_in_anomaly, package_data


def get_data(dataset, sv_nm, bins=None, normalize=True, mix_qs=False, flow=False,
             anomaly_process='WZ_allhad_pT', doping=0, extraStats=True, feature_type=0):
    # Using bins and quantiles to separate semantics between separating base on self defined mass bins and quantiles
    if dataset == 'curtains':
        df = load_curtains_pd(extraStats=extraStats, feature_type=feature_type)
    else:
        raise NotImplementedError('The loader of this dataset has not been implemented yet.')

    cutoff = 2700
    if bins:
        # Split the data into different datasets based on the binning
        if feature_type == 0:
            context_feature = 'mass'
            woi = [40, 150]
        else:
            context_feature = 'mjj'
            woi = [cutoff, 5000]

        anomaly_data = load_curtains_pd(sm=anomaly_process, feature_type=feature_type)

        signal_anomalies, mixed, package = dope_dataframe_new(df, anomaly_data, doping, bins, context_feature)

        lm, sigfrac_lm = package["SB1"]
        hm, sigfrac_hm = package["SB2"]
        ob1, sigfrac_ob1 = package["OB1"]
        ob2, sigfrac_ob2 = package["OB2"]
        signal, sigfrac_sr = package["SR"]

        '''
        plotting the windows:
        df['mass'] will be the bg - mention region of interest ?
        lm_mixed, hm_mixed, ob1_mixed, ob2_mixed, signal_mixed are the ones that enter the whole bg.
        '''
        # mixed = pd.concat([lm_mixed, hm_mixed, ob1_mixed, ob2_mixed, signal_mixed])
        anomaly_mixed_mass = mixed[context_feature]
        bg_mass = df[context_feature]

        get_windows_plot(bg_mass, anomaly_mixed_mass, woi, bins, sv_nm,
                         frac=[sigfrac_ob1, sigfrac_lm, sigfrac_sr, sigfrac_hm, sigfrac_ob2])

        lm = Curtains(lm)
        hm = Curtains(hm)

        # Take a look at the input features prior to scaling
        nfeatures = len(lm.feature_nms) - 1
        fig, axs = plt.subplots(1, nfeatures, figsize=(5 * nfeatures + 2, 5))
        hist_features(lm, hm, nfeatures, axs, axs_nms=lm.feature_nms, labels=['SB1', 'SB2'], legend=False)
        fig.savefig(sv_nm + '_inital_features.png')

        training_data = CurtainsTrainSet(lm, hm, mix_qs=mix_qs, stack=flow)

        # Set the normalization factors for the other datasets
        scale = training_data.set_and_get_norm_facts()
        validation_data_lm = Curtains(ob1, norm=scale)
        validation_data = Curtains(ob2, norm=scale)
        signal_data = Curtains(signal, norm=scale)
        signal_anomalies = Curtains(signal_anomalies, norm=scale)

        # Plot some correlations
        fig, axs = plt.subplots(1, nfeatures, figsize=(5 * nfeatures + 2, 10))
        n_percent = 0.1

        def sample_tensor(tensor):
            tensor = shuffle_tensor(tensor)
            n = int(tensor.shape[0] * n_percent)
            return tensor[:n]

        drape = WrappingCurtains(training_data, signal_data, validation_data, validation_data_lm, bins)

    else:
        return Curtains(df)

    if normalize:
        drape.normalize()
        signal_anomalies.normalize()

        # And prior to being scaled
        nfeatures = len(lm.feature_nms) - 1
        fig, axs = plt.subplots(1, nfeatures, figsize=(5 * nfeatures + 2, 5))
        hist_features(lm, hm, nfeatures, axs, axs_nms=lm.feature_nms, labels=['SB1', 'SB2'], legend=False)
        fig.savefig(sv_nm + '_scaled_features.png')

    ntake = min(len(lm), len(signal_data))
    # TODO: saving in experiment name
    plot_delta_mass(training_data.data[:, 3] - training_data.data[:, -1],
                    true_deltas=lm[:ntake, -1] - signal_data.data[:ntake, -1], name=None, bins=100)

    return drape, signal_anomalies


def get_koala_data(bins=None, anomaly_process='WZ_allhad_pT', doping=0., extraStats=True, dtype=torch.float32,
                   feature_type=0):
    df = load_curtains_pd(extraStats=extraStats, feature_type=feature_type)

    # Split the data into different datasets based on the binning
    if feature_type == 0:
        context_feature = 'mass'
    else:
        context_feature = 'mjj'

    anomaly_data = load_curtains_pd(sm=anomaly_process, feature_type=feature_type)

    _, lm_mixed, lm = mask_dataframe(df, context_feature, bins, [1, 2], doping, anomaly_data)
    _, hm_mixed, hm = mask_dataframe(df, context_feature, bins, [3, 4], doping, anomaly_data)
    _, ob1_mixed, ob1 = mask_dataframe(df, context_feature, bins, [0, 1], doping, anomaly_data)
    _, ob2_mixed, ob2 = mask_dataframe(df, context_feature, bins, [4, 5], doping, anomaly_data)
    signal_anomalies, signal_mixed, signal = mask_dataframe(df, context_feature, bins, [2, 3], doping, anomaly_data)

    background = pd.concat((lm, hm), 0)

    return torch.tensor(background.values, dtype=dtype), torch.tensor(signal.values, dtype=dtype), torch.tensor(
        signal_anomalies.values, dtype=dtype)


def main():
    # import argparse
    # import matplotlib.pyplot as plt
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--download', type=int, default=0,
    #                     help='Choose the base output directory')
    #
    # args = parser.parse_args()
    #
    # if args.download:
    #     load_hepmass(mass='1000')
    #     load_hepmass(mass='all')
    #     load_hepmass(mass='not1000')
    #
    # data_train, data_test = load_hepmass(mass='1000', slim=True)
    #
    # fig, axs_ = plt.subplots(9, 3, figsize=(5 * 3 + 2, 5 * 9 + 2))
    # axs = fig.axes
    # for i, data in enumerate(data_train.data.t()):
    #     axs[i].hist(data.numpy())
    # fig.savefig(get_top_dir() + '/images/hepmass_features.png')

    return 0


if __name__ == '__main__':
    main()
