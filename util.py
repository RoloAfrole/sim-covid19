import matplotlib.pyplot as plt
import numpy as np

import pickle
from datetime import datetime


def plot_sim(status, basename=None, save=True):

    infl_array = np.asarray(status.h_infl)
    seg_array = np.asarray(status.h_seg)
    x_array = np.arange(infl_array.size)
    plt.plot(x_array, infl_array, label="infl")
    plt.plot(x_array, seg_array, label="segregated")
    plt.legend()

    if basename is None:
        basename = datetime.now().strftime('%Y%m%d%H%M')

    filename = '{}.png'.format(basename)

    if save:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
    else:
        plt.show()


def save_status(status, basename=None):

    if basename is None:
        basename = datetime.now().strftime('%Y%m%d%H%M')

    filename = '{}.pickle'.format(basename)

    with open(filename, 'wb') as f:
        pickle.dump(status, f)


def load_status(basename):

    filename = '{}.pickle'.format(basename)

    with open(filename, 'rb') as f:
        status = pickle.load(f)
    return status
