import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.utils import class_weight

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml


class Classifier(nn.Module):
    def __init__(self, layers, n_inputs=5):
        super().__init__()

        self.layers = []
        for nodes in layers:
            self.layers.append(nn.Linear(n_inputs, nodes))
            self.layers.append(nn.ReLU())
            n_inputs = nodes
        self.layers.append(nn.Linear(n_inputs, 1))
        self.layers.append(nn.Sigmoid())
        self.model_stack = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model_stack(x)

    def predict(self, x):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        with torch.no_grad():
            self.eval()
            x = torch.tensor(x, device=device)
            prediction = self.forward(x).detach().cpu().numpy()
        return prediction


def build_classifier(filename, n_inputs=5):
    with open(filename, 'r') as stream:
        params = yaml.safe_load(stream)

    model = Classifier(params['layers'], n_inputs=n_inputs)
    if params['loss'] == 'binary_crossentropy':
        loss = F.binary_cross_entropy
    else:
        raise NotImplementedError

    if params['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=float(params['learning_rate']))
    else:
        raise NotImplementedError

    return model, loss, optimizer


def train_model(classifier_configfile, epochs, X_train, y_train, X_test, y_test,
                X_val=None, batch_size=128, supervised=False, save_model=None, verbose=True):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # training a single classifier model
    if supervised:
        assert X_val is not None, (
            "Validation data needs to be provided in order to run supervised training!")

    input_train = X_train[:, 1:-2]
    if X_val is not None:
        input_val = X_val[:, 1:-2]
    else:
        input_val = X_test[:, 1:-2]

    if supervised:
        print("Running a fully supervised training. Sig/bkg labels will be known!")
        input_train = input_train[y_train == 1]
        input_val = input_val[X_val[:, -2] == 1]
        label_train = X_train[y_train == 1][:, 6]
        label_val = X_val[X_val[:, -2] == 1][:, 6]
    else:
        label_train = y_train
        if X_val is not None:
            label_val = X_val[:, -2]
        else:
            label_val = y_test

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(label_train),
                                                      label_train)
    class_weights = dict(enumerate(class_weights))

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(input_train),
                                                   torch.tensor(label_train).reshape(-1, 1))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(input_val),
                                                 torch.tensor(label_val).reshape(-1, 1))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    train_loss = np.zeros(epochs)  ## add pre-training loss?
    val_loss = np.zeros(epochs)

    model, loss_func, optimizer = build_classifier(classifier_configfile, n_inputs=input_train.shape[1])
    model.to(device)
    for epoch in range(epochs):
        print("training epoch nr", epoch)
        epoch_train_loss = 0.
        epoch_val_loss = 0.

        model.train()
        for i, batch in enumerate(train_dataloader):
            if verbose:
                print("...batch nr", i)
            batch_inputs, batch_labels = batch
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

            if class_weights is not None:
                batch_weights = (torch.ones(batch_labels.shape, device=device)
                                 - batch_labels) * class_weights[0] \
                                + batch_labels * class_weights[1]
            else:
                batch_weights = None

            optimizer.zero_grad()
            batch_outputs = model(batch_inputs)
            batch_loss = loss_func(batch_outputs, batch_labels, weight=batch_weights)
            batch_loss.backward()
            optimizer.step()
            epoch_train_loss += batch_loss
            if verbose:
                print("...batch training loss:", batch_loss.item())

        epoch_train_loss /= (i + 1)
        print("training loss:", epoch_train_loss.item())

        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(val_dataloader):

                batch_inputs, batch_labels = batch
                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                if class_weights is not None:
                    batch_weights = (torch.ones(batch_labels.shape, device=device)
                                     - batch_labels) * class_weights[0] \
                                    + batch_labels * class_weights[1]
                else:
                    batch_weights = None

                batch_outputs = model(batch_inputs)
                batch_loss = loss_func(batch_outputs, batch_labels, weight=batch_weights)
                epoch_val_loss += batch_loss
            epoch_val_loss /= (i + 1)
        print("validation loss:", epoch_val_loss.item())

        train_loss[epoch] = epoch_train_loss
        val_loss[epoch] = epoch_val_loss

        if save_model is not None:
            torch.save(model, save_model + "_ep" + str(epoch))
    return train_loss, val_loss


def train_n_models(n_runs, classifier_configfile, epochs, X_train, y_train, X_test, y_test,
                   X_val=None, batch_size=128, supervised=False,
                   verbose=True, savedir=None, save_model=None):
    # Trains n models and records the resulting losses and test data predictions.
    #    The outputs are saved to files if a directory path is given to savedir.
    #    If supervised is set true, the classifier learns to distinguish sig and
    #    bkg according to their actual labels.

    loss = {}
    val_loss = {}

    for j in range(n_runs):
        print(f"Training model nr {j}...")
        if save_model is not None:
            current_save_model = save_model + "_run" + str(j)
        else:
            current_save_model = None
        loss[j], val_loss[j] = train_model(
            classifier_configfile, epochs, X_train, y_train, X_test, y_test,
            X_val=X_val, batch_size=batch_size,
            supervised=supervised, save_model=current_save_model,
            verbose=verbose)

    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)

    val_loss_matrix = np.zeros((n_runs, epochs))
    for j in range(n_runs):
        for i in range(epochs):
            val_loss_matrix[j, i] = val_loss[j][i]
    loss_matrix = np.zeros((n_runs, epochs))
    for j in range(n_runs):
        for i in range(epochs):
            loss_matrix[j, i] = loss[j][i]

    np.save(os.path.join(savedir, 'val_loss_matris.npy'), val_loss_matrix)
    np.save(os.path.join(savedir, 'loss_matris.npy'), loss_matrix)

    if save_model is None:
        raise NotImplementedError("Removed prediction saving.",
                                  "Please provide model name to save_model.")

    return loss_matrix, val_loss_matrix


def plot_classifier_losses(train_losses, val_losses, yrange=None, savefig=None, suppress_show=False):
    # plots the classifier training losses from loss array. The image is saved if a filename is
    # given to the savefig parameter
    avg_train_losses = (
                               train_losses[5:] + train_losses[4:-1] + train_losses[3:-2]
                               + train_losses[2:-3] + train_losses[1:-4]) / 5
    avg_val_losses = (val_losses[5:] + val_losses[4:-1] + val_losses[3:-2]
                      + val_losses[2:-3] + val_losses[1:-4]) / 5

    plt.plot(range(1, len(train_losses)), train_losses[1:], linestyle=":", color="blue")
    plt.plot(range(1, len(val_losses)), val_losses[1:], linestyle=":", color="orange")
    plt.plot(range(3, len(train_losses) - 2), avg_train_losses, label="Training", color="blue")
    plt.plot(range(3, len(val_losses) - 2), avg_val_losses, label="Validation", color="orange")
    plt.plot(np.nan, np.nan, linestyle="None", label=" ")
    plt.plot(np.nan, np.nan, linestyle=":", color="black", label="Per Epoch Value")
    plt.plot(np.nan, np.nan, linestyle="-", color="black", label="5-Epoch Average")

    if yrange is not None:
        plt.ylim(*yrange)
    plt.xlabel("Training Epoch")
    plt.ylabel("(Mean) Binary Cross Entropy Loss")
    plt.legend(loc="upper right", frameon=False)
    if savefig is not None:
        plt.savefig(savefig + ".pdf", bbox_inches="tight")
    plt.close()


def preds_from_models(model_path_list, X_test, save_dir, predict_on_samples=False, take_mean=True):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if predict_on_samples:
        test_data = X_test
    else:
        test_data = X_test[X_test[:, -2] == 1]

    run_predictions = []
    for i, model_paths in enumerate(model_path_list):  ## looping over runs
        model_list = [torch.load(model_path, map_location=device) for model_path in model_paths]
        for model in model_list:
            model.eval()
        epoch_predictions = []
        for model in model_list:  ## looping over epochs
            epoch_predictions.append(model.predict(test_data[:, 1:-2]).flatten())
        run_predictions.append(np.stack(epoch_predictions))

    preds_matrix = np.stack(run_predictions)
    if take_mean:
        preds_matrix = np.mean(preds_matrix, axis=1, keepdims=True)
    np.save(os.path.join(save_dir, "preds_matris.npy"), preds_matrix)
    return preds_matrix


# tprs and tprs extraction function for the signal vs background task
def tprs_fprs_sics(preds_matris, y_test, X_test):
    runs = preds_matris.shape[0]
    epchs = preds_matris.shape[1]
    tprs = {}
    fprs = {}
    sics = {}
    for j in range(runs):
        for i in range(epchs):
            fpr, tpr, thresholds = roc_curve(X_test[:, -1][y_test == 1],
                                             preds_matris[j, i][y_test == 1])
            fpr_nonzero = np.delete(fpr, np.argwhere(fpr == 0))
            tpr_nonzero = np.delete(tpr, np.argwhere(fpr == 0))
            tprs[j, i] = tpr_nonzero
            fprs[j, i] = fpr_nonzero
            sics[j, i] = tprs[j, i] / fprs[j, i] ** 0.5
    return tprs, fprs, sics


# loading classifier output data from a directory
def load_predictions(X_test, prediction_dir):
    preds_matris = np.load(os.path.join(prediction_dir, 'preds_matris.npy'))
    if X_test.shape[0] == preds_matris.shape[-1]:
        pass
    ## check if predictions are only done on data
    elif X_test[X_test[:, -2] == 1].shape[0] == preds_matris.shape[-1]:
        X_test = X_test[X_test[:, -2] == 1]
    else:
        raise RuntimeError("Data and prediction shapes don't match!")
    val_loss_matris = np.load(os.path.join(prediction_dir, 'val_loss_matris.npy'))
    y_test = X_test[:, -2]

    X_test_combined = X_test
    y_test_combined = y_test
    preds_combined = preds_matris

    return X_test_combined, y_test_combined, preds_combined, val_loss_matris


def minumum_validation_loss_ensemble(predictions_matrix, validation_loss_matrix, n_epochs=10):
    min_val_loss_epochs = np.zeros((validation_loss_matrix.shape[0], n_epochs), dtype=int)
    for i in range(validation_loss_matrix.shape[0]):
        min_val_loss_epochs[i, :] = np.argpartition(validation_loss_matrix[i, :], n_epochs) \
            [:n_epochs]

    min_val_loss_preds = np.zeros((predictions_matrix.shape[0], predictions_matrix.shape[-1]))
    for i in range(predictions_matrix.shape[0]):
        min_val_loss_preds[i, :] = np.mean(predictions_matrix[i, min_val_loss_epochs[i, :], :],
                                           axis=0)
    min_val_loss_preds = min_val_loss_preds.reshape(min_val_loss_preds.shape[0], 1,
                                                    min_val_loss_preds.shape[1])

    return min_val_loss_preds


def compare_on_various_runs(tprs_list, fprs_list, pick_epochs_list, labels_list, sic_lim=None, savefig=None,
                            only_median=False, continuous_colors=False, reduced_legend=False, suppress_show=False,
                            return_all=False):
    assert len(tprs_list) == len(fprs_list) == len(pick_epochs_list) == len(labels_list), (
        "the input lists need to be of the same length")

    picked_median_colors = ["navy", "darkred", "darkgreen", "darkorange"]
    picked_single_colors = ["skyblue", "salmon", "lightgreen", "navajowhite"]
    if not continuous_colors and len(tprs_list) > len(picked_median_colors):
        print("for a non continuous color palette, additional colors need to be added " + \
              "to incorporate that many run collections")
        raise NotImplementedError
    if continuous_colors and not only_median:
        print("currently only support continuous colors on only_median runs")
        raise NotImplementedError

    # interpolation
    tprs_manual_list = []
    roc_median_list = []
    sic_median_list = []
    for run_collection in zip(tprs_list, fprs_list, pick_epochs_list):
        tprs, fprs, pick_epoch = run_collection
        max_min_tpr = 0.
        min_max_tpr = 1.
        for tpr in tprs.values():
            if min(tpr) > max_min_tpr:
                max_min_tpr = min(tpr)
            if max(tpr) < min_max_tpr:
                min_max_tpr = max(tpr)
        tprs_manual = np.linspace(max_min_tpr, min_max_tpr, 1000)
        tprs_manual_list.append(tprs_manual)

        roc_interpol = []
        sic_interpol = []
        for j in range(len(pick_epoch)):
            roc_function = interp1d(tprs[j, pick_epoch[j]], 1 / fprs[j, pick_epoch[j]])
            sic_function = interp1d(tprs[j, pick_epoch[j]],
                                    tprs[j, pick_epoch[j]] / (fprs[j, pick_epoch[j]]) ** (0.5))
            roc_interpol.append(roc_function(tprs_manual))
            sic_interpol.append(sic_function(tprs_manual))
        roc_median_list.append(np.median(np.stack(roc_interpol), axis=0))
        sic_median_list.append(np.median(np.stack(sic_interpol), axis=0))

    # color map
    median_colors = cm.viridis(np.linspace(0., 0.95, len(tprs_list))) if continuous_colors \
        else picked_median_colors[:len(tprs_list)]
    if not only_median:
        single_colors = picked_single_colors[:len(tprs_list)]
    zorder_single = np.arange(0, 5 * len(tprs_list), 5)
    zorder_median = np.arange(5 * len(tprs_list) + 5, 10 * len(tprs_list) + 5, 5)

    # draw ROCs
    plt.subplot(1, 2, 1)
    for k, run_collection in enumerate(zip(tprs_list, fprs_list, pick_epochs_list, labels_list)):
        tprs, fprs, pick_epoch, label = run_collection
        full_label = "" if label == "" else ", " + label
        if not only_median:
            # for j in range(len(pick_epoch)):
            for j, picked_epoch in enumerate(pick_epoch):
                # plt.plot(tprs[j,pick_epoch[j]], 1/fprs[j,pick_epoch[j]], color=single_colors[k])
                plt.plot(tprs[j, picked_epoch], 1 / fprs[j, picked_epoch], color=single_colors[k])
            if not reduced_legend:
                plt.plot(np.nan, np.nan, color=single_colors[k],
                         label=f"{len(pick_epoch)} individual runs{full_label}",
                         zorder=zorder_single[k])
        plt.plot(tprs_manual_list[k], roc_median_list[k], color=median_colors[k],
                 label=f"median{full_label}", zorder=zorder_median[k])
    plt.plot(np.linspace(0.0001, 1, 300), 1 / np.linspace(0.0001, 1, 300),
             color="gray", linestyle=":", label="random")
    plt.title("Signal Region", loc="right", style='italic')
    plt.xlabel('Signal Efficiency (True Positive Rate)')
    plt.ylabel('Rejection (1/False Positive Rate)')
    plt.legend(loc='upper right')
    plt.yscale('log')

    # draw SICs
    plt.subplot(1, 2, 2)
    for k, run_collection in enumerate(zip(tprs_list, fprs_list, pick_epochs_list, labels_list)):
        tprs, fprs, pick_epoch, label = run_collection
        full_label = "" if label == "" else ", " + label
        if not only_median:
            # for j in range(len(pick_epoch)):
            for j, picked_epoch in enumerate(pick_epoch):
                # plt.plot(tprs[j,pick_epoch[j]],
                # tprs[j,pick_epoch[j]]/(fprs[j,pick_epoch[j]])**(0.5), color=single_colors[k])
                plt.plot(tprs[j, picked_epoch],
                         tprs[j, picked_epoch] / (fprs[j, picked_epoch]) ** (0.5),
                         color=single_colors[k])
            if not reduced_legend:
                plt.plot(np.nan, np.nan, color=single_colors[k],
                         label=f"{len(pick_epoch)} individual runs{full_label}",
                         zorder=zorder_single[k])
        plt.plot(tprs_manual_list[k], sic_median_list[k], color=median_colors[k],
                 label=f"median{full_label}", zorder=zorder_median[k])
    plt.plot(np.linspace(0.0001, 1, 300),
             np.linspace(0.0001, 1, 300) / np.linspace(0.0001, 1, 300) ** (0.5), color="gray",
             linestyle=":", label="random")
    plt.ylim(sic_lim)
    plt.title("Signal Region", loc="right", style='italic')
    plt.ylabel('Significance Improvement')
    plt.xlabel('Signal Efficiency (True Positive Rate)')
    plt.legend(loc='upper right')

    # save / display
    plt.subplots_adjust(right=2.0)
    # plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, bbox_inches="tight")

    if return_all:
        return tprs_manual_list, roc_median_list, sic_median_list, tprs_list, fprs_list
    else:
        return tprs_manual_list, roc_median_list, sic_median_list


def full_single_evaluation(prediction_dir, X_test, n_ensemble_epochs=10,
                           sic_range=(0, 20), savefig=None, suppress_show=False, return_all=False):
    X_test, y_test, predictions, val_losses = load_predictions(
        X_test, prediction_dir)
    if predictions.shape[1] == 1:  ## check if ensembling done already
        min_val_loss_predictions = predictions
    else:
        min_val_loss_predictions = minumum_validation_loss_ensemble(
            predictions, val_losses, n_epochs=n_ensemble_epochs)
    tprs, fprs, sics = tprs_fprs_sics(min_val_loss_predictions, y_test, X_test)

    return compare_on_various_runs(
        [tprs], [fprs], [np.zeros(min_val_loss_predictions.shape[0])], [""],
        sic_lim=sic_range, savefig=savefig, only_median=False, continuous_colors=False,
        reduced_legend=False, suppress_show=suppress_show, return_all=return_all)
