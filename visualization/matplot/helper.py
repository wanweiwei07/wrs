import numpy as np
import matplotlib.pyplot as plt

PIXEL_TO_INCH = 0.0104166667


def list_to_plt_xy(data_list):
    return range(len(data_list)), data_list


def twodlist_to_plt_xys(data_2dlist):
    data_array = np.array(data_2dlist)
    return range(data_array.shape[0]), data_array
