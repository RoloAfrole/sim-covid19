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


def plot_history(history,
                 susceptible=True,
                 infected=True,
                 removed=True,
                 filename=None,
                 save=True,
                 title=None,
                 ylimit=None,
                 xlimit=None):

    # plt.figure()
    day_list, his_dic = history.get_history()

    for city_name, his in his_dic.items():
        SEC_array = np.asarray(his[0])
        INF_array = np.asarray(his[1])
        REM_array = np.asarray(his[2])
        x_array = np.arange(INF_array.size)
        if susceptible:
            plt.plot(x_array, SEC_array, label='{}_S'.format(city_name))
        if infected:
            plt.plot(x_array, INF_array, label='{}_I'.format(city_name))
        if removed:
            plt.plot(x_array, REM_array, label='{}_R'.format(city_name))

    plt.legend()

    if title is not None:
        plt.title(title)

    if ylimit is not None:
        plt.ylim(ylimit)

    if xlimit is not None:
        plt.xlim(xlimit)

    plt.xlabel('Day')
    plt.ylabel('Population')
    if susceptible and not infected and not removed:
        plt.ylabel('Susceptible Population')
    if not susceptible and infected and not removed:
        plt.ylabel('Infected Population')
    if not susceptible and not infected and removed:
        plt.ylabel('Removed Population')

    if filename is None:
        filename = datetime.now().strftime('%Y%m%d%H%M')

    filename = '{}.png'.format(filename)

    if save:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
    else:
        plt.show()


def show_history_list(history_list,
                      susceptible=True,
                      infected=True,
                      removed=True,
                      use_def=True,
                      filename=None,
                      save=True,
                      title=None,
                      ylimit=None,
                      xlimit=None):

    # plt.figure()
    total_array_dic = {}
    for history in history_list:
        day_list, his_dic = history.get_history()
        for city_name, his in his_dic.items():
            SEC_array = his[0]
            INF_array = his[1]
            REM_array = his[2]

            if city_name not in total_array_dic:
                total_array_dic[city_name] = [[], [], []]
                total_array_dic[city_name][0].extend(SEC_array)
                total_array_dic[city_name][1].extend(INF_array)
                total_array_dic[city_name][2].extend(REM_array)
            else:
                total_array_dic[city_name][0].extend(SEC_array[1:-1])
                total_array_dic[city_name][1].extend(INF_array[1:-1])
                total_array_dic[city_name][2].extend(REM_array[1:-1])

    for city_name, arrays in total_array_dic.items():
        if use_def is True:
            SEC_array = np.asarray(calc_deff(arrays[0]))
            INF_array = np.asarray(calc_deff(arrays[1]))
            REM_array = np.asarray(calc_deff(arrays[2]))
        else:
            SEC_array = np.asarray(arrays[0][1:-1])
            INF_array = np.asarray(arrays[1][1:-1])
            REM_array = np.asarray(arrays[2][1:-1])
        x_array = np.arange(INF_array.size)
        if susceptible:
            plt.plot(x_array, SEC_array, label='{}_S'.format(city_name))
        if infected:
            plt.plot(x_array, INF_array, label='{}_I'.format(city_name))
        if removed:
            plt.plot(x_array, REM_array, label='{}_R'.format(city_name))

    plt.legend()

    if title is not None:
        plt.title(title)

    if ylimit is not None:
        plt.ylim(ylimit)

    if xlimit is not None:
        plt.xlim(xlimit)

    plt.xlabel('Day')
    plt.ylabel('Population')
    if susceptible and not infected and not removed:
        plt.ylabel('Susceptible Population')
    if not susceptible and infected and not removed:
        plt.ylabel('Infected Population')
    if not susceptible and not infected and removed:
        plt.ylabel('Removed Population')

    if filename is None:
        filename = datetime.now().strftime('%Y%m%d%H%M')

    filename = '{}.png'.format(filename)

    if save:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
    else:
        plt.show()


def calc_deff(array):
    new_array = []
    for i in range(len(array)-1):
        new_array.append(array[i+1] - array[i])

    return new_array
