# Density ratio estimation
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight

from models.classifier import Classifier
from models.nn.networks import dense_net

# Make a class to wrap the dataset and use with torch's dataloader
from torch.utils.data import Dataset
import torch

from utils import hyperparams
from utils.plotting import plot_rates_dict, plot_losses, add_classifier_outputs
from utils.torch_utils import sample_data
from utils.training import Timer


class SupervisedDataClass(Dataset):
    def __init__(self, data, labels, weights=None, dtype=torch.float32, noise_generator=None, bg_labels=None,
                 standardise=1):
        super(SupervisedDataClass, self).__init__()
        self.dtype = dtype
        if isinstance(data, torch.Tensor):
            self.data = data.type(dtype)
        else:
            self.data = torch.tensor(data, dtype=dtype)
        self.targets = torch.tensor(labels, dtype=dtype)
        self.bg_labels = bg_labels

        self.nfeatures = self.data.shape[1]
        self.noise_generator = noise_generator
        self.normed = False
        self.base_data = data
        self.base_labels = labels
        self.weights = torch.ones_like(self.targets)
        self.update_weights = False
        self.standardise = standardise
        if noise_generator is not None:
            self.update_data()
        if weights == 'balance':
            self.update_weights = True
            self.weights = class_weight.compute_sample_weight('balanced', y=self.targets.reshape(-1)).reshape(-1, 1)
        elif weights is not None:
            self.weights = weights

    def __getitem__(self, item):
        return self.data[item], self.targets[item], self.weights[item]

    def __len__(self):
        return self.data.shape[0]

    def get_and_set_norm_facts(self, normalize=False):
        if self.standardise:
            self.max_vals, self.min_vals = list(torch.std_mean(self.data, dim=0))
        else:
            self.max_vals = self.data.max(0)[0]
            self.min_vals = self.data.min(0)[0]
        if normalize:
            self.normalize()
        return self.max_vals, self.min_vals

    def set_norm_facts(self, facts):
        self.max_vals, self.min_vals = facts

    def update_data(self):
        if self.noise_generator is not None:
            data, targets = self.noise_generator(self.base_data, self.base_labels)
            self.data, self.targets = torch.tensor(data, dtype=self.dtype), torch.tensor(targets, dtype=self.dtype)
            if self.normed:
                self.normed = False
                self.normalize()
            if self.update_weights:
                self.weights = class_weight.compute_sample_weight('balanced', y=self.targets.reshape(-1)).reshape(-1, 1)
            else:
                self.weights = torch.ones_like(self.targets)

    def _normalize(self, data_in):
        if self.standardise == 1:
            return (data_in - self.min_vals) / (self.max_vals + 1e-8)
        else:
            return (data_in - self.min_vals) / (self.max_vals - self.min_vals)

    def _unnormalize(self, data_in):
        if self.standardise == 1:
            stds, means = self.max_vals, self.min_vals
            data_out = data_in * (stds + 1e-8) + means
        else:
            data_out = (data_in - self.min_vals) / (self.max_vals - self.min_vals)
        return data_out

    def normalize(self, data_in=None, facts=None):
        data_passed = data_in is not None
        if not data_passed:
            data_in = self.data
        if facts is not None:
            self.set_norm_facts(facts)
        if self.normed:
            data_out = data_in
        else:
            # data_out = (data_in - self.min_vals) / (self.max_vals - self.min_vals)
            data_out = self._normalize(data_in)
            self.normed = True
        if data_passed:
            return data_out
        else:
            self.data = data_out

    def unnormalize(self, data_in=None):
        data_passed = data_in is not None
        if self.normed or data_passed:
            if not data_passed:
                data_in = self.data
            # data_in = data_in * (self.max_vals - self.min_vals) + self.min_vals
            data_in = self._unnormalize(data_in)
            if not data_passed:
                self.data = data_in
        return data_in


def get_net(batch_norm=False, layer_norm=False, width=32, depth=2, dropout=0.0, cf_activ=torch.relu):
    def net_maker(nfeatures, nclasses):
        return dense_net(nfeatures, nclasses, layers=[width] * depth, batch_norm=batch_norm, layer_norm=layer_norm,
                         drp=dropout, context_features=None, int_activ=cf_activ)

    return net_maker


def fit_classifier(classifier, train_data, valid_data, optimizer, batch_size, n_epochs, device, sv_dir, plot=True,
                   load_best=False, scheduler=None, pure_noise=False, fold=0):
    # Initialize timer class, this is useful on the cluster as it will say if you have enough time to run the job
    timer = Timer('irrelevant', 'irrelevant', print_text=False)

    # Make an object to load training data
    n_workers = 0
    data_obj = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    data_valid = torch.utils.data.DataLoader(valid_data, batch_size=1000, shuffle=True, num_workers=n_workers)
    n_train = int(np.ceil(len(train_data) / batch_size))
    n_valid = int(np.ceil(len(valid_data) / 1000))
    if pure_noise:
        # Get the data bounding values
        l1 = train_data.data.min()
        l2 = train_data.data.max()

    train_loss = np.zeros(n_epochs)
    valid_loss = np.zeros(n_epochs)
    max_sic_tpr = np.zeros(n_epochs)
    max_sic = np.zeros(n_epochs)
    scheduler_bool = scheduler is not None
    classifier_dir = os.path.join(sv_dir, f'classifier_{fold}')
    os.makedirs(classifier_dir, exist_ok=True)
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        # Start the timer
        timer.start()

        train_data.update_data()
        valid_data.update_data()
        if pure_noise:
            # Replace the signal data with random samples from a uniform distribution
            mx = (train_data.targets == 0).view(-1)
            train_data.data[mx] = (l1 - l2) * torch.rand_like(train_data.data[mx]) + l2
            data_obj = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                   num_workers=n_workers)

        running_loss = np.zeros(n_train)
        for i, data in enumerate(data_obj, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # Get the model loss
            data = [dt.to(device) for dt in data]
            loss = classifier.compute_loss(data)
            # Propogate the loss
            loss.backward()
            # Update the parameters
            optimizer.step()
            if scheduler_bool:
                scheduler.step()

            # Get statistics
            running_loss[i] = loss.item()

        # Save loss info for the epoch
        train_loss[epoch] = np.mean(running_loss)

        # Validate
        running_loss = np.zeros(n_valid)
        classifier.eval()
        store = []
        with torch.no_grad():
            for i, data in enumerate(data_valid, 0):
                # Get the model loss
                loss, pred = classifier.compute_loss(data, return_pred=True)
                running_loss[i] = loss.item()
                store += [np.concatenate([data[1].cpu(), pred.cpu()], 1)]
        lbls = np.concatenate(store)
        fpr, tpr, _ = roc_curve(lbls[:, 0], lbls[:, 1])
        fpr_mx = fpr != 0
        sic = tpr[fpr_mx] / fpr[fpr_mx] ** 0.5
        max_ind = np.argmax(sic)
        max_sic[epoch] = sic[max_ind]
        max_sic_tpr[epoch] = tpr[fpr_mx][max_ind]
        valid_loss[epoch] = np.mean(running_loss)
        classifier.save(f'{classifier_dir}/{epoch}')

        # Stop the timer
        classifier.train()
        timer.stop()

    # Save the validation and training loss metrics
    np.save(f'{classifier_dir}/valid_loss.npy', valid_loss)
    np.save(f'{classifier_dir}/train_loss.npy', train_loss)
    np.save(f'{classifier_dir}/max_sic.npy', max_sic)
    np.save(f'{classifier_dir}/max_sic_tpr.npy', max_sic_tpr)

    if plot:
        # Plot loss development and sic development
        fig, ax = plt.subplots(1, 3, figsize=(3 * 5 + 2, 5))
        ax[0].plot(train_loss, label='Train')
        ax[0].plot(valid_loss, label='Validation')
        ax[0].legend()
        ax[0].set_title('Classifier Training')
        ax[1].plot(max_sic, label='Max SIC')
        ax[1].legend()
        ax[2].plot(max_sic_tpr, label='Max SIC tpr')
        ax[2].legend()
        [ax[i].set_xlabel('epochs') for i in range(3)]
        fig.savefig(f'{classifier_dir}/training_{fold}.png')

    if load_best:
        best_epoch = np.argmin(valid_loss)
        # Index is counted from zero so add one to get the best epoch
        print(f'Best epoch: {best_epoch + 1} loaded')
        classifier.load(f'{classifier_dir}/{best_epoch}')

    classifier.eval()

    return train_loss, valid_loss


def dope_data(truth, anomaly_data, beta):
    n = int(len(truth) * (1 - beta))
    n1 = len(truth) - n
    n_anomalies = int(len(anomaly_data))
    if n1 > n_anomalies:
        print(f"Can't dope higher than {beta}")
        n1 = n_anomalies
        n = int(len(truth)) - n1
    truth = sample_data(truth, n)
    anomaly_data, anomaly_data_train = sample_data(anomaly_data, n1, split=True)
    try:
        truth = torch.cat((anomaly_data_train, truth), 0)
    except TypeError:
        truth = np.concatenate((anomaly_data_train.detach().cpu(), truth), 0)
        np.random.shuffle(truth)
    return anomaly_data, truth


def get_datasets(train_index, valid_index, eval_index, false_signal, X, y, beta_add_noise, use_weights,
                 bg_truth_labels, standardise):
    def index_data(index):
        return X[index], y[index]

    X_train, y_train = index_data(train_index)
    X_val, y_val = index_data(valid_index)
    X_eval, y_eval = index_data(eval_index)

    # These will only be used at evaluation
    if bg_truth_labels is not None:
        bg_truth_val = bg_truth_labels[valid_index]
        bg_truth_eval = bg_truth_labels[eval_index]
    else:
        bg_truth_val = None
        bg_truth_eval = None

    if false_signal in [1, 2]:
        # Append a dummy noise sample to the
        n_features = X_train.shape[1]

        def add_noise(data, labels):
            n_sample = int(data.shape[0] * beta_add_noise)
            if false_signal == 1:
                data = np.concatenate(
                    (data, np.random.multivariate_normal([0] * n_features, np.eye(n_features), n_sample)))
            else:
                l1 = np.quantile(data, 0.01, axis=0)
                l2 = np.quantile(data, 0.99, axis=0)
                data = np.concatenate(
                    (data, ((l1 - l2) * np.random.rand(*data.shape) + l2)[:n_sample]))
            labels = np.concatenate((labels, np.ones((n_sample, 1))))
            return data, labels
    elif false_signal == 3:
        def add_noise(data, labels):
            n_sample = int(data.shape[0] * beta_add_noise)
            eps = 1e-2
            scale_factor = eps * np.std(data, axis=0)
            new_data = data[-n_sample:] + scale_factor * np.random.rand(n_sample, data.shape[1])
            data = np.concatenate((data[:-n_sample], new_data))
            return data, labels
    else:
        add_noise = None

    weights = 'balance' if use_weights else None
    train_data = SupervisedDataClass(X_train, y_train, weights=weights, noise_generator=add_noise,
                                     standardise=standardise)
    valid_data = SupervisedDataClass(X_val, y_val, weights=weights, noise_generator=add_noise,
                                     bg_labels=bg_truth_val, standardise=standardise)
    eval_data = SupervisedDataClass(X_eval, y_eval, weights=weights, noise_generator=None, bg_labels=bg_truth_eval,
                                    standardise=standardise)

    return train_data, valid_data, eval_data


def run_classifiers(bg_template, sr_samples, directory, name, anomaly_data=None, bg_truth_labels=None, mass_incl=True,
                    sup_title='', load=False, return_rates=False, false_signal=1, normalize=True, batch_size=1000,
                    nepochs=100,
                    lr=0.0001, wd=0.001, drp=0.0, width=32, depth=3, batch_norm=False, layer_norm=False,
                    use_scheduler=True,
                    use_weights=True, thresholds=None, beta_add_noise=0.1, pure_noise=False, nfolds=5,
                    data_unscaler=None, cf_activ='relu'):
    """
    bg_truth_labels 0 = known anomaly, 1 = known background, -1 = unknown/sampled/transformed sample
    """
    if thresholds is None:
        thresholds = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999]
    if bg_truth_labels is None:
        bg_truth_labels = torch.cat((torch.zeros(len(bg_template)),
                                     torch.ones(len(sr_samples))))

    tpr_c, fpr_c = None, None

    def prepare_data(data):
        data = data.detach().cpu()
        if data_unscaler is not None:
            data = data_unscaler(data)
        if mass_incl:
            mass = data[:, -1]
            data = data[:, :-1]
        else:
            mass = None
        return data, mass

    bg_template, bg_mass = prepare_data(bg_template)
    sr_samples, sr_mass = prepare_data(sr_samples)

    # Classifier hyperparameters
    if drp > 0:
        width = int(width / drp)

    sv_dir = os.path.join(directory, name)
    anomaly_bool = anomaly_data is not None
    if anomaly_bool:
        anomaly_data, ad_mass = prepare_data(anomaly_data)

    X, y = torch.cat((bg_template, sr_samples), 0).cpu().numpy(), torch.cat(
        (torch.zeros(len(bg_template)), torch.ones(len(sr_samples))), 0).view(-1, 1).cpu().numpy()

    if mass_incl:
        masses = torch.cat((bg_mass, sr_mass), 0)
    else:
        masses = None

    # Setting random_state to an integer means repeated calls yield the same result
    kfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=1)

    # Define a custom generator for getting train, validation, test splits
    def kfold_gen(scikit_generator):
        """
        A generator that, given a scikit learn generator, will return groups of train, validation, test indicies
        """
        n = scikit_generator.get_n_splits()
        indicies = np.array([arr[1] for arr in scikit_generator.split(X, y)], dtype=object)
        count = 0
        while count < n:
            yield np.concatenate(indicies[0:3]).flatten().astype(np.int32), \
                  indicies[3].astype(np.int32), \
                  indicies[4].astype(np.int32)
            count += 1
            indicies = np.roll(indicies, 1, 0)

    split_inds = kfold_gen(kfold)

    store_losses = []

    # Train the model
    for fold, (train_index, valid_index, eval_index) in enumerate(split_inds):

        # The validation data will not be touched in the training loop
        train_data, valid_data, _ = get_datasets(train_index, valid_index, eval_index, false_signal, X, y,
                                                 beta_add_noise, use_weights, bg_truth_labels, normalize)

        if normalize:
            facts = train_data.get_and_set_norm_facts(normalize=True)
            valid_data.normalize(facts=facts)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = get_net(batch_norm=batch_norm, layer_norm=layer_norm, width=width, depth=depth, dropout=drp,
                      cf_activ=hyperparams.activations[cf_activ])
        classifier = Classifier(net, train_data.nfeatures, 1, name, directory=directory,
                                activation=torch.sigmoid).to(device)

        # Make an optimizer object
        if (wd is None) or (wd == 0.):
            optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
            # optimizer = torch.optim.NAdam(classifier.parameters(), lr=lr)
        else:
            optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=wd)
        if use_scheduler:
            max_step = int(nepochs * np.ceil(len(train_data.data) / batch_size))
            # TODO: pass this, set to one by default
            periods = 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_step / periods, 0)
        else:
            scheduler = None

        # Train or load a classifier
        def load_classifier():
            classifier.load(os.path.join(sv_dir, f'classifier_{fold}', f'{nepochs - 1}'))
            classifier_dir = os.path.join(sv_dir, f'classifier_{fold}')
            valid_loss = np.load(f'{classifier_dir}/valid_loss.npy')
            train_loss = np.load(f'{classifier_dir}/train_loss.npy')
            return train_loss, valid_loss

        def fit_the_classifier():
            losses = fit_classifier(classifier, train_data, valid_data, optimizer, batch_size, nepochs,
                                    device, sv_dir, scheduler=scheduler, pure_noise=pure_noise, fold=fold,
                                    load_best=False)
            return losses

        if load == 1:
            losses = load_classifier()
        # Load a classifier if it has been run, otherwise train one
        elif load == 2:
            try:
                losses = load_classifier()
            except:
                losses = fit_the_classifier()
        else:
            losses = fit_the_classifier()

        # store the losses from the training
        store_losses += [losses]

    # From the saved losses pick the best epoch
    n_av = 5

    # Take every second because the first is the training loss
    losses = np.concatenate(store_losses)
    train_loss = losses[0::2]
    val_loss = losses[1::2]
    plot_losses(train_loss, val_loss, sv_dir)
    # eval_epoch = np.argsort(val_loss.mean(0))[:n_av]
    # eval_epoch = [nepochs - 1, nepochs - 2]
    eval_epoch = [nepochs - 1]
    # print(f'Best epoch: {eval_epoch}. \nLoading and evaluating now.')
    split_inds = kfold_gen(kfold)

    info_dict = defaultdict(list)
    rates_dict = {}
    # Evaluate each model at the selected epoch
    # Plot the classifier output distributions
    fig, ax = plt.subplots(1, nfolds, figsize=(5 * nfolds + 2, 7))
    for fold, (train_index, valid_index, eval_index) in enumerate(split_inds):

        # # TODO: this was set to the last epoch for 'paper' trainings
        # eval_epoch = np.argsort(val_loss[fold])[:n_av]
        print(f'Best epoch: {eval_epoch}. \nLoading and evaluating now.')

        # The classifier object does not need to be reinitialised here, only loaded
        models_to_load = [os.path.join(sv_dir, f'classifier_{fold}', f'{e}') for e in eval_epoch]
        classifier.load(models_to_load)

        # The training data is needed to get the scaling information
        train_data, _, eval_data = get_datasets(train_index, valid_index, eval_index, 0, X, y, 0.0, use_weights,
                                                bg_truth_labels, normalize)
        eval_masses = masses[eval_index].reshape(-1, 1)

        if normalize:
            facts = train_data.get_and_set_norm_facts(normalize=True)
            eval_data.normalize(facts=facts)

        with torch.no_grad():
            y_scores = classifier.predict(eval_data.data.to(device)).cpu().numpy()
        info_dict['y_scores'] += [y_scores]
        labels_test = eval_data.targets
        info_dict['labels_test'] += [labels_test.cpu().numpy()]

        # Plot the classifier distribution
        if anomaly_bool:
            with torch.no_grad():
                ad = SupervisedDataClass(anomaly_data, np.ones(len(anomaly_data)))
                if normalize:
                    ad.normalize(facts=facts)
                ad = ad.data
                anomaly_scores = classifier.predict(ad.to(device)).cpu().numpy()
            if eval_data.bg_labels is not None:
                lbls_bg = eval_data.bg_labels.cpu().numpy()
                lbls = lbls_bg
                bg_scores = y_scores[:, 0]
            else:
                bg_scores = y_scores[eval_data.targets == 0]
                # bg_scores = eval_data.targets.cpu().numpy()
                lbls_bg = np.zeros(len(bg_scores))
                lbls = lbls_bg
            # Get the background vs signal AUC if that is available
            data_mx = lbls_bg != -1
            info_dict['y_labels_1'] += [np.concatenate((np.ones(len(anomaly_scores)), lbls_bg[data_mx]))]
            info_dict['y_scores_1'] += [np.concatenate((anomaly_scores.reshape(-1, 1),
                                                        bg_scores[data_mx].reshape(-1, 1)))[:, 0]]
            # info_dict['y_labels_1'] += [lbls_bg[data_mx]]
            # info_dict['y_scores_1'] += [bg_scores[data_mx]]
            # Get the background only AUC if that information is available
            info_dict['y_labels_2'] += [eval_data.targets.cpu().numpy()[lbls == 0]]
            info_dict['y_scores_2'] += [y_scores[lbls == 0]]

            info_dict['masses_folds'] += [eval_masses.cpu().numpy()]
            info_dict['bg_labels'] += [lbls_bg]

            # Plot classifier outputs per fold
            d1 = info_dict['y_scores_1'][-1]
            s1 = info_dict['y_scores'][-1][info_dict['labels_test'][-1] == 0]
            lbls_mx = info_dict['labels_test'][-1] == 1
            s2 = info_dict['y_scores'][-1][lbls_mx]
            # Mask out the anomalies
            s2 = s2[info_dict['bg_labels'][-1].reshape(-1, 1)[lbls_mx] == 0]
            add_classifier_outputs(ax[fold], d1, s1, s2)

            # Calculate and plot some AUCs for the epoch
            fpr, tpr, _ = roc_curve(info_dict['y_labels_1'][-1], info_dict['y_scores_1'][-1])
            rates_dict[f'{fold}'] = [fpr, tpr]

            # Get the sic value
            fpr_nz = fpr[fpr != 0]
            tpr_nz = tpr[fpr != 0]
            sic = tpr_nz / fpr_nz ** 0.5
            print(f'Max SIC: {np.max(sic)}')

            # Calculate the expected and real counts that pass a certain threshold of the classifier
            if return_rates:
                # Count the number of events that are left in the signal region after a certain cut on the background
                # template
                count = []
                count_bg = []
                expected_count = []
                store_masses = []
                store_masses_labels = []
                mx = eval_data.targets == 0
                for i, at in enumerate(thresholds):
                    threshold = np.quantile(y_scores[mx], at)
                    expected_count += [sum(y_scores[mx] >= threshold)]
                    count += [sum(y_scores[eval_data.targets == 1] >= threshold)]
                    count_bg += [np.sum(y_scores[eval_data.bg_labels == 1] >= threshold)]
                    ms_mx = eval_data.targets[:, 0] == 1
                    store_masses += [eval_masses[ms_mx][y_scores[ms_mx] >= threshold]]
                    store_masses_labels += [eval_data.bg_labels.view(-1, 1)[ms_mx][y_scores[ms_mx] >= threshold]]
                    if anomaly_bool:
                        signal_pass_rate = np.sum(
                            info_dict['y_scores_1'][-1][info_dict['y_labels_1'][-1] == 1] >= threshold) / np.sum(
                            info_dict['y_labels_1'][-1] == 1)
                        bg_pass_rate = np.sum(
                            info_dict['y_scores_1'][-1][info_dict['y_labels_1'][-1] == 0] >= threshold) / np.sum(
                            info_dict['y_labels_1'][-1] == 0)
                        info_dict['pass_rates'] += [np.array((signal_pass_rate, bg_pass_rate))]
                info_dict['counts'] += [np.array(count)]
                info_dict['masses'] += [store_masses]
                info_dict['masses_labels'] += [store_masses_labels]
                info_dict['expected_counts'] += [np.array(expected_count)]
                info_dict['sig_counts'] += [np.array(count_bg).flatten()]

    fig.legend()
    fig.savefig(os.path.join(sv_dir, 'folds_classifier_outputs.png'))

    plot_rates_dict(sv_dir, rates_dict, 'folds')

    keys_to_cat = ['y_scores', 'labels_test', 'y_labels_1', 'y_scores_1', 'y_labels_2', 'y_scores_2', 'masses_folds',
                   'bg_labels', 'sig_counts']
    for key in keys_to_cat:
        if key in info_dict.keys():
            info_dict[key] = np.concatenate(info_dict[key])

    # # Calculate the expected and real counts that pass a certain threshold of the classifier
    # if return_rates:
    #     # Count the number of events that are left in the signal region after a certain cut on the background
    #     # template
    #     count = []
    #     count_signal = []
    #     expected_count = []
    #     store_masses = []
    #     store_masses_labels = []
    #     mx = info_dict['labels_test'] == 0
    #     y_scores = info_dict['y_scores']
    #     for i, at in enumerate(thresholds):
    #         threshold = np.quantile(y_scores[mx], at)
    #         expected_count += [np.sum(y_scores[mx] >= threshold)]
    #         count += [np.sum(y_scores[info_dict['labels_test'] == 1] >= threshold)]
    #         # count_signal += [np.sum(y_scores[info_dict['bg_labels'] == 1] >= threshold)]
    #         ms_mx = info_dict['labels_test'] == 1
    #         store_masses += [info_dict['masses_folds'][ms_mx][y_scores[ms_mx] >= threshold]]
    #         store_masses_labels += [info_dict['bg_labels'].reshape(-1, 1)[ms_mx][y_scores[ms_mx] >= threshold]]
    #         if anomaly_bool:
    #             signal_pass_rate = np.sum(info_dict['y_scores_1'][info_dict['y_labels_1'] == 1] >= threshold) / np.sum(
    #                 info_dict['y_labels_1'] == 1)
    #             bg_pass_rate = np.sum(info_dict['y_scores_1'][info_dict['y_labels_1'] == 0] >= threshold) / np.sum(
    #                 info_dict['y_labels_1'] == 0)
    #             info_dict['pass_rates'] += [np.array((signal_pass_rate, bg_pass_rate))]
    #     info_dict['counts'] += [np.array(count)]
    #     info_dict['masses'] += [store_masses]
    #     info_dict['expected_counts'] += [np.array(expected_count)]
    #     info_dict['sig_counts'] += [np.array(count_signal)]

    # fpr, tpr, _ = roc_curve(labels_test, y_scores)
    fpr, tpr, _ = roc_curve(info_dict['labels_test'], info_dict['y_scores'])
    roc_auc = auc(fpr, tpr)

    # # Plot the classifier output distributions
    # fig, ax = plt.subplots()
    # d1 = info_dict['y_scores_1']
    # bins = get_bins(d1, nbins=80)
    # plt_kwargs = {'bins': bins, 'alpha': 0.8, 'histtype': 'step'}
    #
    # def get_weight(fact):
    #     return np.ones_like(fact) / np.sum(fact)
    #
    # ax.hist(d1, label='Anomalies', weights=get_weight(d1), **plt_kwargs)
    # s1 = info_dict['y_scores'][info_dict['labels_test'] == 0]
    # ax.hist(s1, weights=get_weight(s1), label='Train Label 0', **plt_kwargs)
    # lbls_mx = info_dict['labels_test'] == 1
    # s2 = info_dict['y_scores'][lbls_mx]
    # # Mask out the anomalies
    # s2 = s2[info_dict['bg_labels'].reshape(-1, 1)[lbls_mx] == 0]
    # ax.hist(s2, weights=get_weight(s2), label='Train Label 1', **plt_kwargs)
    # ax.set_yscale('log')
    # fig.legend()
    # fig.savefig(os.path.join(sv_dir, 'classifier_outputs.png'))

    if anomaly_bool:
        lmx = np.isfinite(info_dict['y_scores_1'])
        fpr1, tpr1, _ = roc_curve(info_dict['y_labels_1'][lmx], info_dict['y_scores_1'][lmx])
        if tpr_c is not None:
            # Plot a roc curve
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(fpr1, tpr1, label='Curtains', linewidth=2)
            key = list(fpr_c.keys())[0]
            ax.plot(fpr_c[key], tpr_c[key], label='CATHODE', linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xticks(np.arange(0, 1.1, 0.1))
            ax.set_xlabel('False positive rate')
            ax.set_ylabel('True positive rate')
            ax.legend()
            fig.savefig(sv_dir + 'roc_compare.png')

        lmx = np.isfinite(info_dict['y_scores_2'])
        fpr2, tpr2, _ = roc_curve(info_dict['y_labels_2'][lmx], info_dict['y_scores_2'][lmx])
        roc_auc_anomalies = auc(fpr1, tpr1)

    if return_rates:
        counts = np.array(info_dict['counts'])
        measured = np.sum(counts, 0)
        print(f'Expected {np.sum(info_dict["expected_counts"], 0)}.\n Measured {measured}.')
        print(f'Signal counts {np.sum(info_dict["sig_counts"], 0)}.')
        counts = {'counts': measured, 'counts_sep': info_dict['counts'],
                  'expected_counts': info_dict["expected_counts"],
                  'pass_rates': info_dict['pass_rates'], 'masses': info_dict['masses'],
                  'masses_labels': info_dict['masses_labels']}

    # Plot a roc curve
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    if anomaly_bool:
        ax.set_title(f'SR vs Transforms {roc_auc:.2f} \n SR anomalies vs SR QCD {roc_auc_anomalies:.2f}')
    else:
        ax.set_title(f'{sup_title} {roc_auc:.2f}')
    fig.savefig(sv_dir + 'roc.png')

    print(f'ROC AUC {roc_auc}')

    if return_rates:
        if anomaly_bool:
            return roc_auc, [fpr, tpr], [fpr1, tpr1], [fpr2, tpr2], counts, rates_dict
        else:
            return roc_auc, [fpr, tpr]
    else:
        return roc_auc
