import math
import numpy as np


class Trajectory(object):

    def __init__(self, method="cubic"):
        if method == "cubic":
            self.fit = self._cubic_coeffs
            self.predict = self._predict_cubic
        elif method == "quintic":
            self.fit = self._quintic_coeffs
            self.predict = self._predict_quintic
        self.cubicmat = np.array([[2, 1, -2, 1],
                                  [-3, -2, 3, -1],
                                  [0, 1, 0, 0],
                                  [1, 0, 0, 0]])
        self.quinticmat = np.array([[0, 0, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1, 1],
                                    [0, 0, 0, 0, 1, 0],
                                    [5, 4, 3, 2, 1, 0],
                                    [0, 0, 0, 2, 0, 0],
                                    [20, 12, 6, 2, 0, 0]])
        self.coeffs_array = None

    def _cubic_coeffs(self, conf0, spd0, conf1, spd1):
        self.coeffs_array = np.dot(self.cubicmat, np.vstack((conf0, spd0, conf1, spd1)))

    def _quintic_coeffs(self, conf0, spd0, conf1, spd1, acc0=None, acc1=None):
        if acc0 is None:
            acc0 = np.zeros_like(spd0)
        if acc1 is None:
            acc1 = np.zeros_like(spd1)
        self.coeffs_array = np.linalg.solve(self.quinticmat, np.vstack((conf0, conf1, spd0, spd1, acc0, acc1)))

    def _predict_cubic(self, step):
        """
        step = currenttime/timeinterval
        :return:
        author: weiwei
        date: 20200327
        """
        step_array = np.vstack([step ** 3, step ** 2, step, np.ones_like(step)])
        spd_step_array = np.vstack([3 * step ** 2, 2 * step, step, np.zeros_like(step)])
        if isinstance(step, np.ndarray):
            return np.dot(self.coeffs_array.T, step_array).T, np.dot(self.coeffs_array.T, spd_step_array).T
        else:
            return np.dot(self.coeffs_array.T, step_array).T[0][0], np.dot(self.coeffs_array.T, spd_step_array).T[0][0]

    def _predict_quintic(self, step):
        """
        step = currenttime/timeinterval
        :return:
        author: weiwei
        date: 20200327
        """
        conf_step_array = np.vstack([step ** 5, step ** 4, step ** 3, step ** 2, step, np.ones_like(step)])
        spd_step_array = np.vstack(
            [5 * step ** 4, 4 * step ** 3, 3 * step ** 2, 2 * step, np.ones_like(step), np.zeros_like(step)])
        if isinstance(step, np.ndarray):
            return np.dot(self.coeffs_array.T, conf_step_array).T, np.dot(self.coeffs_array.T, spd_step_array).T
        else:
            return np.dot(self.coeffs_array.T, conf_step_array).T[0][0], \
                   np.dot(self.coeffs_array.T, spd_step_array).T[0][0]

    def set_interpolation_method(self, method):
        """
        change interpolation method
        :param name: 'cubic' or 'quintic'
        :return:
        author: weiwei
        date: 20210331
        """
        if method == "cubic":
            self.fit = self._cubic_coeffs
            self.predict = self._predict_cubic
        elif method == "quintic":
            self.fit = self._quintic_coeffs
            self.predict = self._predict_quintic
        else:
            pass

    def piecewise_interpolation(self, path, control_frequency=.005, interval_time=1.0):
        """
        :param path: a 1d array of configurations
        :param control_frequency: the program will sample interval_time/control_frequency confs
        :param interval_time: time to move between adjacent joints
        :return:
        author: weiwei
        date: 20200328
        """
        path = np.array(path)
        passing_conf_list = []
        passing_spd_list = []
        for id, jntconf in enumerate(path[:-1]):
            passing_conf_list.append(jntconf)
            if id == 0:
                passing_spd_list.append(np.zeros_like(jntconf))
            else:
                pre_conf = path[id - 1]
                nxt_conf = path[id + 1]
                pre_avg_spd = (jntconf - pre_conf) / interval_time
                nxt_avg_spd = (nxt_conf - jntconf) / interval_time
                # set to 0 if signs are different -> reduces overshoot
                zero_id = np.where((np.sign(pre_avg_spd) + np.sign(nxt_avg_spd)) == 0)
                pass_spd = (pre_avg_spd + nxt_avg_spd) / 2.0
                pass_spd[zero_id] = 0.0
                passing_spd_list.append(pass_spd)
        passing_conf_list.append(path[-1])
        passing_spd_list.append(np.zeros_like(path[-1]))
        interpolated_confs = []
        interpolated_spds = []
        for id, passing_conf in enumerate(passing_conf_list):
            if id == 0:
                continue
            pre_passing_conf = passing_conf_list[id - 1]
            pre_passing_spd = passing_spd_list[id - 1]
            passing_spd = passing_spd_list[id]
            self.fit(pre_passing_conf, pre_passing_spd, passing_conf, passing_spd)
            samples = np.linspace(0, 1, math.floor(interval_time / control_frequency))
            local_interpolated_confs, local_interplated_spds = self.predict(samples)
            interpolated_confs += local_interpolated_confs.tolist()
            interpolated_spds += local_interplated_spds.tolist()
        return interpolated_confs, interpolated_spds


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    y = [[0], [3], [0], [9], [0]]
    control_frequency = .02
    interval_time = 1.0
    traj = Trajectory(method="quintic")
    interpolated_confs, interpolated_spds = traj.piecewise_interpolation(y, control_frequency=control_frequency,
                                                                         interval_time=interval_time)
    # print(interpolated_spds)
    # interpolated_spds=np.array(interpolated_spds)
    print(interpolated_confs)
    x = np.linspace(0, len(y) - 1, (len(y) - 1) * math.floor(interval_time / control_frequency))
    plt.plot(x, interpolated_confs)
    plt.quiver(x, interpolated_confs, x, interpolated_spds, width=.001)
    plt.plot(y)
    plt.show()
