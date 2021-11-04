import glob
import numpy as np
import matplotlib.pyplot as plt


def main():
    directory = 'images/cluster_losses'
    files = glob.glob(f'{directory}/*.npy')

    losses = []
    for file in files:
        losses += [np.array(np.load(file, allow_pickle=True), dtype='float32')]

    ncut = 100
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    for i in range(2):
        train_loss = losses[i][0, ncut:]
        val_loss = losses[i][1, ncut:]
        ax[i].plot(np.arange(ncut, len(losses[i][0])), train_loss, label='train')
        ax[i].plot(np.arange(ncut, len(losses[i][0])),val_loss, label='valid')
        ax[i].legend()
    fig.savefig(f'{directory}/losses.png')



if __name__ == '__main__':
    main()