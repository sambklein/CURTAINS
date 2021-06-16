import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from models.classifier import Classifier
from models.nn.networks import dense_net

# Make a class to wrap the dataset and use with torch's dataloader
from torch.utils.data import Dataset
import torch

from utils.plotting import add_error_hist, get_bins
from utils.torch_utils import sample_data


class SupervisedDataClass(Dataset):
    def __init__(self, data, labels, dtype=torch.float32):
        super(SupervisedDataClass, self).__init__()
        self.data = torch.tensor(data, dtype=dtype)
        self.targets = torch.tensor(labels, dtype=dtype)
        self.nfeatures = self.data.shape[1]

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return self.data.shape[0]


def net(nfeatures, nclasses):
    return dense_net(nfeatures, nclasses, layers=[64, 32, 16])


def fit_classifier(classifier, train_data, valid_data, optimizer, batch_size, n_epochs, device, sv_dir, plot=True,
                   save=True):
    # Make an object to load training data
    data_obj = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    data_valid = torch.utils.data.DataLoader(valid_data, batch_size=1000, shuffle=True, num_workers=0)
    n_train = int(np.ceil(len(train_data) / batch_size))
    n_valid = int(np.ceil(len(valid_data) / 1000))

    train_loss = np.zeros(n_epochs)
    valid_loss = np.zeros(n_epochs)
    for epoch in range(n_epochs):  # loop over the dataset multiple times

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

            # Get statistics
            running_loss[i] = loss.item()

        # Save loss info for the epoch
        train_loss[epoch] = np.mean(running_loss)

        # Validate
        running_loss = np.zeros(n_valid)
        with torch.no_grad():
            for i, data in enumerate(data_valid, 0):
                # Get the model loss
                loss = classifier.compute_loss(data)
                running_loss[i] = loss.item()
        valid_loss[epoch] = np.mean(running_loss)

    if plot:
        plt.figure()
        plt.plot(train_loss, label='Train')
        plt.plot(valid_loss, label='Validation')
        plt.legend()
        plt.title('Classifier Training')
        plt.tight_layout()
        plt.savefig(sv_dir + 'Training.png')

    if save:
        classifier.save(sv_dir + 'classifier')

    print('Finished Training')


def get_auc(interpolated, truth, directory, name, split=0.5, anomaly_data=None, mass_incl=True, balance=True,
            beta=None, sup_title='', mscaler=None):
    if balance:
        n = min((len(interpolated), len(truth)))
        interpolated = sample_data(interpolated, n)
        truth = sample_data(truth, n)
    sv_dir = directory + f'/{name}'
    anomaly_bool = anomaly_data is not None
    beta_bool = beta is not None

    if beta_bool and anomaly_bool:
        if not anomaly_bool:
            print('No anomalies passed to DRE.')
        else:
            n = int(len(truth) * (1 - beta))
            n1 = len(truth) - n
            truth = sample_data(truth, n)
            anomaly_data_train = sample_data(anomaly_data, n1)

    if anomaly_bool:
        truth = torch.cat((anomaly_data_train, truth), 0)

    X, y = torch.cat((interpolated, truth), 0).cpu().numpy(), torch.cat(
        (torch.ones(len(interpolated)), torch.zeros(len(truth))), 0).view(-1, 1).cpu().numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=1, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1, stratify=y_train)

    if mass_incl:
        test_mass = X_test[:, -1]
        if mscaler is not None:
            test_mass = mscaler(test_mass)
        X_train, X_val, X_test = X_train[:, :-1], X_val[:, :-1], X_test[:, :-1]

    train_data = SupervisedDataClass(X_train, y_train)
    valid_data = SupervisedDataClass(X_val, y_val)
    test_data = SupervisedDataClass(X_test, y_test)

    batch_size = 100
    nepochs = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classifier = Classifier(net, train_data.nfeatures, 1, name, directory=directory,
                            activation=torch.sigmoid).to(device)

    # Make an optimizer object
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # Train
    fit_classifier(classifier, train_data, valid_data, optimizer, batch_size, nepochs, device, sv_dir)

    with torch.no_grad():
        y_scores = classifier.predict(test_data.data.to(device)).cpu().numpy()
    labels_test = test_data.targets.cpu().numpy()
    fpr, tpr, _ = roc_curve(labels_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot the classifier distribution
    mx = labels_test == 0
    fig, ax = plt.subplots(1, 1)

    def add_normed_hist(data, ax, label):
        total = len(data)
        ax.hist(data, weights=np.ones_like(data) / total, label=label, histtype='step')

    add_normed_hist(y_scores[mx], ax, 'BG')
    add_normed_hist(y_scores[~mx], ax, 'Signal')
    if anomaly_bool:
        with torch.no_grad():
            if mass_incl:
                ad = anomaly_data.data[:, :-1]
            else:
                ad = anomaly_data.data
            anomaly_scores = classifier.predict(ad.to(device)).cpu().numpy()
        add_normed_hist(anomaly_scores, ax, 'Anomalies')
        fpr1, tpr1, _ = roc_curve(np.concatenate((np.zeros(len(anomaly_scores)), np.ones(len(y_scores[~mx])))),
                                  np.concatenate((anomaly_scores[:, 0], y_scores[~mx])))
        roc_auc_anomalies = auc(fpr1, tpr1)
    fig.legend()
    fig.savefig(f'{sv_dir}_classifier_distribution_{name}.png')

    # Plot a roc curve
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    if anomaly_bool:
        ax.set_title(f'Signal vs background{roc_auc:.2f} \n Injected anomalies vs signal {roc_auc_anomalies:.2f}')
    else:
        ax.set_title(f'{roc_auc}')
    fig.savefig(sv_dir + 'roc.png')

    if mass_incl:
        # TODO: convert back to the orginial mass
        # Plot the mass distribution for different cuts on the classifier
        thresholds = [0, 0.5, 0.8, 0.95, 0.99]
        fig, ax = plt.subplots(1, len(thresholds), figsize=(5 * len(thresholds) + 2, 5))
        mx = labels_test == 0
        bins = None
        test_mass = test_mass[:, np.newaxis]
        for i, at in enumerate(thresholds):
            threshold = np.quantile(y_scores[~mx], 1 - at)
            bg_mass = test_mass[~mx][y_scores[~mx] < threshold]
            signal_mass = test_mass[mx][y_scores[mx] < threshold]
            if bins is None:
                bins = get_bins(bg_mass, nbins=10)
                norm_bg = np.sum(np.histogram(bg_mass, bins=bins)[0])
                norm_signal = np.sum(np.histogram(signal_mass, bins=bins)[0])
            add_error_hist(ax[i], bg_mass, bins, 'blue', error_bars=True, label='Background', norm=norm_bg)
            add_error_hist(ax[i], signal_mass, bins, 'red', error_bars=True, label='Signal', norm=norm_signal)
            ax[i].set_yscale('log')
            ax[i].set_title(f'BG rejection {at:.2f}')
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.suptitle(sup_title)
        fig.tight_layout()
        fig.savefig(f'{sv_dir}_mass_dist_{name}.png')

    print(f'ROC AUC {roc_auc}')

    return roc_auc
