import torch

from utils.plotting import hist_features, get_windows_plot, plot_delta_mass
from .physics_datasets import WrappingCurtains, Curtains, CurtainsTrainSet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.io import on_cluster, make_slim


def calculate_mass(four_vector):
    return (four_vector[:, 0] ** 2 - np.sum(four_vector[:, 1:4] ** 2, axis=1)) ** 0.5


def load_curtains_pd(sm='QCDjj_pT', dtype='float32', extraStats=False, feature_type=0):
    if on_cluster():
        directory = '/srv/beegfs/scratch/groups/rodem/LHCO'
    else:
        directory = 'data/downloads'
    lhco_filename = 'events_anomalydetection_v2.features.h5'
    df = pd.read_hdf(f'{directory}/{lhco_filename}')
    make_slim(df, directory, lhco_filename)

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

    data = data[['mj1', 'mj2-mj1', r'$\tau_{21}^{j_1}$', r'$\tau_{21}^{j_2}$', r'$dR_{jj}$', 'mjj']]

    return data


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

        # Make a meta object to wrap the data class
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

    return drape, signal_anomalies


def main():
    return 0


if __name__ == '__main__':
    main()
