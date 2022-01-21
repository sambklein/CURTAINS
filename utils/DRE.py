import os
import pdb
from copy import deepcopy

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight

from models.classifier import Classifier
from models.nn.networks import dense_net

# Make a class to wrap the dataset and use with torch's dataloader
from torch.utils.data import Dataset
import torch

from utils import CATHODE_classifier
from utils.plotting import add_error_hist, get_bins, hist_features
from utils.torch_utils import sample_data
from utils.training import Timer


class SupervisedDataClass(Dataset):
    def __init__(self, data, labels, weights=None, dtype=torch.float32, noise_generator=None, bg_labels=None):
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
        # self.max_vals = self.data.max(0)[0]
        # self.min_vals = self.data.min(0)[0]
        self.max_vals, self.min_vals = list(torch.std_mean(self.data, dim=0))
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

    def _normalize(self, data_in):
        # return (data_in - self.min_vals) / (self.max_vals - self.min_vals)
        return (data_in - self.min_vals) / (self.max_vals + 1e-8)

    def _unnormalize(self, data_in):
        # return (data_in - self.min_vals) / (self.max_vals - self.min_vals)
        stds, means = self.max_vals, self.min_vals
        return data_in * (stds + 1e-8) + means

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


def get_net(batch_norm=False, layer_norm=False, width=32, depth=2, dropout=0.0):
    def net_maker(nfeatures, nclasses):
        return dense_net(nfeatures, nclasses, layers=[width] * depth, batch_norm=batch_norm, layer_norm=layer_norm,
                         drp=dropout, context_features=None)

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
        fig.savefig(sv_dir + f'Training_{fold}.png')

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


def get_datasets(train_index, test_index, false_signal, X, y, beta_add_noise, use_weights, bg_truth_labels):
    X_train = X[train_index]
    y_train = y[train_index]
    X_val = X[test_index]
    y_val = y[test_index]
    # These will only be used at evaluation
    if bg_truth_labels is not None:
        bg_truth_labels = bg_truth_labels[test_index]

    if false_signal > 0:
        # Append a dummy noise sample to the
        n_features = X_train.shape[1]

        def add_noise(data, labels):
            n_sample = int(data.shape[0] * beta_add_noise)
            if false_signal == 1:
                data = np.concatenate(
                    (data, np.random.multivariate_normal([0] * n_features, np.eye(n_features), n_sample)))
            else:
                # l1 = data.min(0)
                # l2 = data.max(0)
                # The data is normalised, so we take this as the support for the 1+eps
                l1 = -1
                l2 = 1
                data = np.concatenate(
                    (data, ((l1 - l2) * np.random.rand(*data.shape) + l2)[:n_sample]))
            labels = np.concatenate((labels, np.zeros((n_sample, 1))))
            return data, labels
    else:
        add_noise = None

    weights = 'balance' if use_weights else None
    train_data = SupervisedDataClass(X_train, y_train, weights=weights, noise_generator=add_noise)
    valid_data = SupervisedDataClass(X_val, y_val, weights=weights, noise_generator=add_noise,
                                     bg_labels=bg_truth_labels)

    return train_data, valid_data


def get_auc(bg_template, sr_samples, directory, name, anomaly_data=None, bg_truth_labels=None, mass_incl=True,
            sup_title='', load=False, return_rates=False, false_signal=1, normalize=True, batch_size=1000, nepochs=100,
            lr=0.0001, wd=0.001, drp=0.0, width=32, depth=3, batch_norm=False, layer_norm=False, use_scheduler=True,
            use_weights=True, thresholds=None, beta_add_noise=0.1, pure_noise=False, nfolds=5, data_unscaler=None):
    """
    bg_truth_labels 0 = known anomaly, 1 = known background, -1 = unknown/sampled/transformed sample
    """
    if thresholds is None:
        thresholds = [0, 0.5, 0.8, 0.9, 0.95, 0.99]

    if anomaly_data is not None:
        CATHODE_classifier.get_auc(bg_template, sr_samples, directory, name, anomaly_data=anomaly_data,
                               data_unscaler=data_unscaler, mass_incl=mass_incl, bg_truth_labels=bg_truth_labels,
                               batch_size=batch_size, normalize=normalize, nepochs=nepochs, load=load)

    def prepare_data(data):
        data = data.detach().cpu()
        if data_unscaler is not None:
            data = data_unscaler(data)
        if mass_incl:
            data = data[:, :-1]
        return data

    bg_template = prepare_data(bg_template)
    sr_samples = prepare_data(sr_samples)

    # Classifier hyperparameters
    if drp > 0:
        width = int(width / drp)

    sv_dir = os.path.join(directory, name)
    anomaly_bool = anomaly_data is not None
    if anomaly_bool:
        anomaly_data = prepare_data(anomaly_data)

    X, y = torch.cat((bg_template, sr_samples), 0).cpu().numpy(), torch.cat(
        (torch.ones(len(bg_template)), torch.zeros(len(sr_samples))), 0).view(-1, 1).cpu().numpy()

    # Setting random_state to an integer means repeated calls yield the same result
    kfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=1)
    # Cast to a list to iterate over object twice, safer than recalculating and not memory intensive
    split_inds = kfold.split(X, y)

    store_losses = []

    # Train the model
    for fold, (train_index, test_index) in enumerate(split_inds):

        train_data, valid_data = get_datasets(train_index, test_index, false_signal, X, y, beta_add_noise, use_weights,
                                              bg_truth_labels)

        if normalize:
            facts = train_data.get_and_set_norm_facts(normalize=True)
            valid_data.normalize(facts=facts)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = get_net(batch_norm=batch_norm, layer_norm=layer_norm, width=width, depth=depth, dropout=drp)
        classifier = Classifier(net, train_data.nfeatures, 1, name, directory=directory,
                                activation=torch.sigmoid).to(device)

        # Make an optimizer object
        if (wd is None) or (wd == 0.):
            optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        else:
            optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=wd)
        if use_scheduler:
            max_step = int(nepochs * np.ceil(len(train_data.data) / batch_size))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_step, 0)
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
    # TODO: number of epochs to take
    n_av = 5
    # Take every second because the first is the training loss
    losses = np.concatenate(store_losses)[1::2]
    eval_epoch = np.argsort(losses.mean(0))[:n_av]
    # eval_epoch = [nepochs - 1]
    print(f'Best epoch: {eval_epoch}. \nLoading and evaluating now.')
    models_to_load = [os.path.join(sv_dir, f'classifier_{fold}', f'{e}') for e in eval_epoch]
    split_inds = kfold.split(X, y)

    y_scores_s = []
    labels_test_s = []
    y_labels_1 = []
    y_scores_1 = []
    y_labels_2 = []
    y_scores_2 = []
    counts = []
    expected_counts = []
    pass_rates = []
    # Evaluate each model at the selected epoch
    for fold, (train_index, test_index) in enumerate(split_inds):

        # The classifier object does not need to be reinitialised here, only loaded
        classifier.load(models_to_load)

        train_data, valid_data = get_datasets(train_index, test_index, 0, X, y, 0.0, use_weights, bg_truth_labels)

        if normalize:
            facts = train_data.get_and_set_norm_facts(normalize=True)
            valid_data.normalize(facts=facts)

        with torch.no_grad():
            y_scores = classifier.predict(valid_data.data.to(device)).cpu().numpy()
        y_scores_s += [y_scores]
        labels_test = valid_data.targets
        labels_test_s += [labels_test]

        # Plot the classifier distribution
        if anomaly_bool:
            with torch.no_grad():
                ad = SupervisedDataClass(anomaly_data, np.ones(len(anomaly_data)))
                if normalize:
                    ad.normalize(facts=facts)
                ad = ad.data
                anomaly_scores = classifier.predict(ad.to(device)).cpu().numpy()
            if valid_data.bg_labels is not None:
                lbls = valid_data.bg_labels
                bg_scores = y_scores
            else:
                bg_scores = y_scores[valid_data.targets == 1]
                lbls = np.ones(len(bg_scores))
            # Get the background vs signal AUC if that is available
            y_labels_1 += [np.concatenate((np.zeros(len(anomaly_scores)), lbls))]
            y_scores_1 += [np.concatenate((anomaly_scores, bg_scores))]
            # Get the background only AUC if that information is available
            y_labels_2 += [valid_data.targets[lbls == 1]]
            y_scores_2 += [y_scores[lbls == 1]]

            # Calculate and plot some AUCs for the epoch
            fpr, tpr, _ = roc_curve(y_labels_1[-1], y_scores_1[-1])
            roc_auc_1 = roc_auc_score(y_labels_1[-1], y_scores_1[-1])
            roc_auc_2 = roc_auc_score(y_labels_2[-1], y_scores_2[-1])
            roc_auc_3 = roc_auc_score(labels_test, y_scores)
            with open(os.path.join(sv_dir, f'classifier_info_{fold}.txt'), 'w') as f:
                f.write(f'SR vs transformed {roc_auc_2} \n')
                f.write(f'SR QCD vs SR anomalies {roc_auc_1} \n')
                f.write(f'Classification on training {roc_auc_3} \n')

        # Calculate the expected and real counts that pass a certain threshold of the classifier
        if return_rates:
            # Count the number of events that are left in the signal region after a certain cut on the background
            # template
            count = []
            expected_count = []
            mx = valid_data.targets == 1
            for i, at in enumerate(thresholds):
                threshold = np.quantile(y_scores[mx], 1 - at)
                expected_count += [sum(y_scores[mx] <= threshold)]
                count += [sum(y_scores[valid_data.targets == 0] <= threshold)]
                if anomaly_bool:
                    signal_pass_rate = sum(anomaly_scores <= threshold) / len(anomaly_scores)
                    bg_pass_rate = sum(bg_scores[lbls == 1] <= threshold) / len(y_scores)
                    pass_rates += [np.concatenate((signal_pass_rate, bg_pass_rate))]
            counts += [np.array(count)]
            expected_counts += [np.array(expected_count)]

    y_scores = np.concatenate(y_scores_s)
    labels_test = np.concatenate(labels_test_s)

    fpr, tpr, _ = roc_curve(labels_test, y_scores)
    roc_auc = auc(fpr, tpr)

    if anomaly_bool:
        y_labels_1 = np.concatenate(y_labels_1)
        y_scores_1 = np.concatenate(y_scores_1)
        y_labels_2 = np.concatenate(y_labels_2)
        y_scores_2 = np.concatenate(y_scores_2)
        lmx = np.isfinite(y_scores_1)
        fpr1, tpr1, _ = roc_curve(y_labels_1[lmx[:, 0]], y_scores_1[lmx])
        lmx = np.isfinite(y_scores_2)
        fpr2, tpr2, _ = roc_curve(y_labels_2[lmx], y_scores_2[lmx])
        roc_auc_anomalies = auc(fpr1, tpr1)

    if return_rates:
        counts = np.array(counts)
        if anomaly_bool:
            pass_rates = np.array(pass_rates)
        counts = {'counts': counts, 'expected_counts': expected_counts, 'pass_rates': pass_rates}

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
            return roc_auc, [fpr, tpr], [fpr1, tpr1], [fpr2, tpr2], counts
        else:
            return roc_auc, [fpr, tpr]
    else:
        return roc_auc
