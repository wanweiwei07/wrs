import numpy as np
import matplotlib.pyplot as plt

jnt_max = 5
jnt_min = 0
jnt_threshhold_max = 4
jnt_threshhold_min = 1


def get_weight(jnt_value):
    return_weight = np.ones(1)
    # min damping interval
    if jnt_value < jnt_threshhold_min:
        if jnt_value < jnt_min:
            return 1e-6
        normalized_value = (jnt_value - jnt_min) / (jnt_threshhold_min - jnt_min)
        return -2 * np.power(normalized_value, 3) + 3 * np.power(normalized_value, 2)
    if jnt_value > jnt_threshhold_max:
        if jnt_value > jnt_max:
            return 1e-6
        normalized_value = (jnt_max - jnt_value) / (jnt_max - jnt_threshhold_max)
        return -2 * np.power(normalized_value, 3) + 3 * np.power(normalized_value, 2)
    return 1


jnt_max = np.array([5, 5, 5])
jnt_min = np.array([0, 0, 0])
jnt_threshhold_max = np.array([4, 4, 4])
jnt_threshhold_min = np.array([1, 1, 1])


def _wln_weightmat(jntvalues):
    """
    get the wln weightmat
    :param jntvalues:
    :return:
    author: weiwei
    date: 20201126
    """
    wtmat = np.ones(len(jntvalues))
    # min damping interval
    jnts_below_min = jnt_threshhold_min - jntvalues
    selection = jnts_below_min > 0
    normalized_diff_at_selected = ((jntvalues - jnt_min) / (jnt_threshhold_min - jnt_min))[selection]
    wtmat[selection] = -2 * np.power(normalized_diff_at_selected, 3) + 3 * np.power(normalized_diff_at_selected, 2)
    # max damping interval
    jnts_above_max = jntvalues - jnt_threshhold_max
    selection = jnts_above_max > 0
    normalized_diff_at_selected = ((jnt_max - jntvalues) / (jnt_max - jnt_threshhold_max))[selection]
    wtmat[selection] = -2 * np.power(normalized_diff_at_selected, 3) + 3 * np.power(normalized_diff_at_selected, 2)
    wtmat[jntvalues >= jnt_max] = 1e-6
    wtmat[jntvalues <= jnt_min] = 1e-6
    return wtmat


onej = np.linspace(-2, 7, 1000).tolist()
x = []
for element in onej:
    x.append(np.array([element, element, element]))
y = []
for each_value in x:
    y.append(_wln_weightmat(each_value))
print(x,y)
plt.plot(x, y)
plt.show()
