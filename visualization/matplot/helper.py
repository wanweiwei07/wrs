import numpy as np
import matplotlib.pyplot as plt


def plot_list(list, label=None, title=None):
    """
    :param list:
    :param label:
    :param title:
    :return:
    author: ruishuang
    date:
    """
    plt.plot(range(len(list)), list, label=label)
    if title is not None:
        plt.title(title)


def plot_motion_by_joints(conf_list, scatter=False, title="auto"):
    """
    :param conf_list:
    :param scatter:
    :param title:
    :return:
    author: ruishuang
    date:
    """
    x = range(len(conf_list))
    conf_list = np.array(conf_list)
    if scatter:
        plt.scatter(x, [p for p in conf_list[:, 0]])
        plt.scatter(x, [p for p in conf_list[:, 1]])
        plt.scatter(x, [p for p in conf_list[:, 2]])
        plt.scatter(x, [p for p in conf_list[:, 3]])
        plt.scatter(x, [p for p in conf_list[:, 4]])
        plt.scatter(x, [p for p in conf_list[:, 5]])
    else:
        plt.plot(x, [p for p in conf_list[:, 0]])
        plt.plot(x, [p for p in conf_list[:, 1]])
        plt.plot(x, [p for p in conf_list[:, 2]])
        plt.plot(x, [p for p in conf_list[:, 3]])
        plt.plot(x, [p for p in conf_list[:, 4]])
        plt.plot(x, [p for p in conf_list[:, 5]])
    plt.title(title)
    plt.legend(range(6))

def show():
    plt.show()