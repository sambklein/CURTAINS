from torch.nn.utils import clip_grad_norm_
from tqdm import trange
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os

from .io import get_top_dir, on_cluster

import time


class TimerError(Exception):
    """Timer exception."""


class Timer:
    def __init__(self, dir, nm, text="One cycle takes on average {:.1f} seconds", print_text=False):
        self.text = text
        self._start_time = None
        self.cnt = 0
        self.times = []
        sv_dir = get_top_dir() + '/logs/' + dir
        if not os.path.exists(sv_dir):
            os.makedirs(sv_dir)
        self.file_name = sv_dir + '/' + nm
        self.print_text = print_text

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self, sv=True):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        self.times += [elapsed_time]
        self.report()
        if sv:
            self.save_time()

    def save_time(self):
        self.report()
        with open(self.file_name, 'w') as f:
            f.write('{}\n'.format(self.mean_time))

    def report(self):
        self.mean_time = np.mean(self.times)
        if self.print_text:
            print(self.text.format(self.mean_time))


def get_vars(optim):
    return optim.param_groups[0]['params']


def get_grads(loss, vars):
    # This returns the gradients of loss wrt vars
    return torch.autograd.grad(loss, vars, retain_graph=True, allow_unused=True)
    # try:
    #     return torch.autograd.grad(loss, vars, retain_graph=True, allow_unused=True)
    # except:
    #     print('Gradient does not work')


def step_optimizers(grads, vars, optimizer, grad_norm_clip_value=None):
    # Set any gradients to zero before updating
    optimizer.zero_grad()
    for j, var in enumerate(vars):
        var.grad = grads[j]
    if grad_norm_clip_value is not None:
        clip_grad_norm_(vars, grad_norm_clip_value)
    optimizer.step()


def fit(model, optimizers, dataset, n_epochs, batch_size, writer, schedulers=None, regularizers=None,
        schedulers_epoch_end=None, gclip=None, monitor_interval=None, nsplit=5, plot_history=True,
        shuffle_epoch_end=False, load_best=False):
    """
    :param model: the model to be trained
    :param optimizers: a list of optimizers corresponding to the sub models of the passed model
    :param dataset: data to train on, instance of torch.utils.dataset
    :param n_epochs: number of epochs to train the model for
    :param batch_size: the batch size used to make the torch.utils.data.DataLoader
    :param writer: a SummaryWriter from tensorboardX
    :param schedulers: list of learning rate schedulers for the corresponding optimizers
    :param regularizers: list of functions that return regularizers for the params in the corresponding optimizers list
    :param monitor_interval: number of steps at which to print the training statistics
    :return: A model trained according to the defined procedure
    """

    # ndata = dataset.data.shape[0]
    kfold = KFold(n_splits=nsplit, shuffle=True)
    inds = kfold.split(dataset)
    # TODO: implement the full cross validation process if doing hyper parameter scans, will need to handle val_loss saving
    train_inds, val_inds = list(inds)[0]
    # If shuffling on epoch end the data must have this method
    if shuffle_epoch_end:
        trainset, valset = dataset.get_valid(val_inds, train_inds)
    else:
        trainset = dataset[train_inds]
        valset = dataset[val_inds]
    top_dir = get_top_dir()
    sv_dir = top_dir + '/images' + '/' + model.dir
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)

    # Initialize timer class, this is useful on the cluster as it will say if you have enough time to run the job
    timer = Timer(model.dir, '/timing_{}.txt'.format(model.exp_name))

    # n_work = 2 if on_cluster() else 0
    n_work = 0
    ntrain = trainset.data.shape[0]
    trainset.data = trainset.data[:ntrain - (ntrain % batch_size), :]
    training_data = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=n_work,
                                                pin_memory=False)
    val_data = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=n_work,
                                           pin_memory=False)

    # If a monitoring interval is not passed set a default
    if not monitor_interval:
        monitor_interval = 100

    tbar = trange(n_epochs, position=0, leave=True)
    trec = trange(n_epochs, desc='{desc}', position=2)
    tval = trange(n_epochs, desc='{desc}', position=3)
    # Get the names of the model losses
    loss_nms = model.loss_names
    # Make a pandas dataframe to store validation losses and one for train losses
    train_save = pd.DataFrame(columns=loss_nms)
    val_save = pd.DataFrame(columns=loss_nms)
    global_step = 0
    for epoch in tbar:
        timer.start()

        global_step += 1
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        if regularizers:
            if not isinstance(regularizers, list):
                regularizers = [regularizers]

        if schedulers:
            # Don't define a scheduler if None is passed!
            if not isinstance(schedulers, list):
                schedulers = [schedulers]

        running_loss = []
        for i, data in enumerate(training_data, 0):
            if isinstance(data, list):
                data = data[0]
            data = data.to(model.device)

            # # zero the parameter gradients before calculating the losses
            [optimizer.zero_grad() for optimizer in optimizers]

            losses = model.compute_loss(data, batch_size)
            if not isinstance(losses, list):
                losses = [losses]

            if regularizers:
                regs = [reg() for reg in regularizers]
                # Add the regularization terms to the corresponding losses
                losses = [sum(x) for x in zip(losses, regs)]

            vars = [get_vars(optimizer) for optimizer in optimizers]
            grads = [get_grads(*pairs) for pairs in zip(losses, vars)]
            [step_optimizers(*triplets, grad_norm_clip_value=gclip) for triplets in zip(grads, vars, optimizers)]

            if schedulers:
                [scheduler.step() for scheduler in schedulers]

            if i % monitor_interval == 0:
                summaries = model.get_loss_state(4)
                running_loss += [list(summaries.values())]

                s = '[{}, {}] {}'.format(epoch + 1, i + 1, summaries)
                if not on_cluster():
                    trec.set_description_str(s)

                for summary, value in summaries.items():
                    writer.add_scalar(tag=summary, scalar_value=value, global_step=global_step)

        # save the mean training loss for the epoch
        train_loss = np.mean(np.array(running_loss), axis=0)
        train_info = dict(zip(loss_nms, train_loss))
        train_save = train_save.append(train_info, ignore_index=True)

        # Save the latest model, overwriting the last save
        mdl_dir = top_dir + '/data/saved_models/'
        path = mdl_dir + 'model_{}'.format(model.exp_name)
        if not os.path.exists(mdl_dir):
            os.makedirs(mdl_dir, exist_ok=True)
        model.save(path)

        val_loss = []
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_data, 0):
                if isinstance(data, list):
                    data = data[0]
                data = data.to(model.device)
                # This updates the internal loss states which we want to access, the actual outputs are weighted sums
                model.compute_loss(data, batch_size)
                # Get the current loss state to 4 sf
                val_loss += [list(model.get_loss_state(4).values())]
        model.train()
        # Calculate the averages and make a dictionary of the losses.
        # TODO in cross-val we are interested in the full list - pandas dataframe
        val_loss = np.mean(np.array(val_loss), axis=0)
        val_info = dict(zip(loss_nms, val_loss))
        s = 'Validation Losses: ' + str(val_info)
        if not on_cluster():
            tval.set_description_str(s)
        val_save = val_save.append(val_info, ignore_index=True)

        # If the best validation error doesn't exist create it - useful for reloading models
        loss_track = np.mean(list(val_info.values()))
        try:
            best_val
        except NameError:
            best_val = loss_track
        # Save the best model
        if loss_track <= best_val:
            best_val = loss_track
            best_model_path = top_dir + '/data/saved_models/model_{}_best'.format(model.exp_name)
            model.save(best_model_path)

        # Step schedulers that use the validation data for stepping
        if schedulers_epoch_end:
            [scheduler.step(np.mean(val_loss)) for scheduler in schedulers_epoch_end]

        if shuffle_epoch_end:
            # Shuffle the targets and input
            trainset.shuffle()
            valset.shuffle()
            # This is just to ensure that we don't have more data than will fit in a given batch size
            trainset.data = trainset.data[:ntrain - (ntrain % batch_size), :]
            training_data = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                        num_workers=n_work)
            val_data = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=n_work)

        # Stop the timer
        timer.stop()

    if plot_history:
        fig, axs = plt.subplots(1, len(loss_nms), figsize=(20, 5))
        for j, ax in enumerate(fig.axes):
            nm = loss_nms[j]
            ax.plot(val_save[nm], label='validation')
            ax.plot(train_save[nm], '--', label='train')
            ax.set_title(nm)
            ax.legend()
            ax.set_ylabel("loss")
            ax.set_xlabel("epoch")
        fig.savefig(sv_dir + '/training_{}.png'.format(model.exp_name))

    if load_best:
        model.load(best_model_path)


    print('\nFinished Training')
