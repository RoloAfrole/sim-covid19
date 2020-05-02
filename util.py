import matplotlib.pyplot as plt
import numpy as np

import pickle
from datetime import datetime


def plot_sim(status, basename=None, save=True, title=None, ylimit=None):

    # plt.figure()
    infl_array = np.asarray(status.h_infl)
    seg_array = np.asarray(status.h_seg)
    x_array = np.arange(infl_array.size)
    plt.plot(x_array, infl_array, label="Infected", marker='o')
    plt.plot(x_array, seg_array, label="Removed", marker='v')
    plt.legend()

    if title is not None:
        plt.title(title)

    if ylimit is not None:
        plt.ylim(ylimit)

    plt.xlabel('Day')
    plt.ylabel('Population')

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


def save_as_csv(status, basename):
    infl_array = np.asarray(status.h_infl)
    seg_array = np.asarray(status.h_seg)
    filename = '{}.csv'.format(basename)

    stacked = np.stack([infl_array, seg_array])
    np.savetxt(filename,
               stacked.T,
               delimiter=',',
               fmt='%d',
               header='infl,segregated')


def plot_sims(status_list,
              infected=True,
              segregated=True,
              filename=None,
              save=True,
              title=None,
              ylimit=None,
              xlimit=None):

    # plt.figure()

    for status, labels, marks in status_list:
        infl_array = np.asarray(status.h_infl)
        seg_array = np.asarray(status.h_seg)
        x_array = np.arange(infl_array.size)
        if infected:
            plt.plot(x_array, infl_array, label=labels[0], marker=marks[0])
        if segregated:
            plt.plot(x_array, seg_array, label=labels[1], marker=marks[1])

    plt.legend()

    if title is not None:
        plt.title(title)

    if ylimit is not None:
        plt.ylim(ylimit)

    if xlimit is not None:
        plt.xlim(xlimit)

    plt.xlabel('Day')
    plt.ylabel('Population')
    if infected and not segregated:
        plt.ylabel('Infected Population')
    if not infected and segregated:
        plt.ylabel('Removed Population')

    if filename is None:
        filename = datetime.now().strftime('%Y%m%d%H%M')

    filename = '{}.png'.format(filename)

    if save:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
    else:
        plt.show()
