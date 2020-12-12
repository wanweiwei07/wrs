import math
import numpy as np


class Trajectory(object):

    def __init__(self, type="cubic"):
        if type is "cubic":
            self.fit = self._cubic_coeffs
            self.predict = self._predict_cubic
        elif type is "quintic":
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
        if isinstance(step, np.ndarray):
            return np.dot(self.coeffs_array.T, step_array).T
        else:
            return np.dot(self.coeffs_array.T, step_array).T[0][0]

    def _predict_quintic(self, step):
        """
        step = currenttime/timeinterval
        :return:
        author: weiwei
        date: 20200327
        """
        step_array = np.vstack([step ** 5, step ** 4, step ** 3, step ** 2, step, np.ones_like(step)])
        if isinstance(step, np.ndarray):
            return np.dot(self.coeffs_array.T, step_array).T
        else:
            return np.dot(self.coeffs_array.T, step_array).T[0][0]

    def piecewise_interpolation(self, path, sampling=200, intervaltime=1.0):
        """
        :param path: a 1d array of configurations
        :param sampling: how many confs to generate in the intervaltime
        :param intervaltime: time to move between adjacent joints, intervaltime = expandis/speed, speed = degree/second
                             by default, the value is 1.0 and the speed is expandis/second
        :return:
        author: weiwei
        date: 20200328
        """
        path = np.array(path)
        passingconflist = []
        passingspdlist = []
        for id, jntconf in enumerate(path[:-1]):
            passingconflist.append(jntconf)
            if id == 0:
                passingspdlist.append(np.zeros_like(jntconf))
            else:
                preconf = path[id - 1]
                nxtconf = path[id + 1]
                preavgspd = (jntconf - preconf) / intervaltime
                nxtavgspd = (nxtconf - jntconf) / intervaltime
                # set to 0 if signs are different
                zeroid = np.where((np.sign(preavgspd) + np.sign(nxtavgspd)) == 0)
                passspd = (preavgspd + nxtavgspd) / 2.0
                passspd[zeroid] = 0.0
                passingspdlist.append(passspd)
        passingconflist.append(path[-1])
        passingspdlist.append(np.zeros_like(path[-1]))
        interpolatedconfs = []
        for id, passingconf in enumerate(passingconflist):
            if id == 0:
                continue
            prepassingconf = passingconflist[id - 1]
            prepassingspd = passingspdlist[id - 1]
            passingspd = passingspdlist[id]
            self.fit(prepassingconf, prepassingspd, passingconf, passingspd)
            samples = np.linspace(0, 1, sampling)
            localinterpolatedconfs = self.predict(samples)
            interpolatedconfs += localinterpolatedconfs.tolist()
        return interpolatedconfs


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    y = [[0], [3], [5], [9]]
    traj = Trajectory(type="quintic")
    interpolatedconfs = traj.piecewise_interpolation(y)
    print(interpolatedconfs)
    x = np.linspace(0, 9, 600)
    plt.plot(x, interpolatedconfs)
    plt.show()
