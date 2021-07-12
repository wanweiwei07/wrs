import scipy.interpolate as sinter
import numpy as np
import math


class PiecewisePoly(object):

    def __init__(self, method="linear"):
        self.change_method(method=method)

    def change_method(self, method="cubic"):
        self.method = method
        if method == "linear":
            self._solve = self._linear_solve
            self._interpolate = self._linear_interpolate
        elif method == "cubic":
            self._solve = self._cubic_solve
            self._interpolate = self._cubic_interpolate
        elif method == "quintic":
            self._solve = self._quintic_solve
            self._interpolate = self._quintic_interpolate

    def _linear_solve(self):
        x = range(self._path_array.shape[0])
        y = self._path_array
        return sinter.interp1d(x, y, kind="linear", axis=0)

    def _linear_interpolate(self, A, samples_list):
        """
        :param A: a linear call back function since we are using scipy
        :param samples_list: a list of 1xn_jnts nparray, with each element holding the samples need to be interpolated
        :return: interpolated_pos, interpolated_spd, interpolated_acc, interpolated_x (x axis ticks)
        author: weiwei
        date: 20210712
        """
        n_sections = self._n_pnts - 1
        interpolated_x = []
        for i in range(n_sections):
            if i == n_sections - 1:  # last
                interpolated_x_sec = (samples_list[i] + i).tolist()
            else:
                interpolated_x_sec = (samples_list[i] + i).tolist()[:-1]
            interpolated_x += interpolated_x_sec
        interpolated_y = A(np.array(interpolated_x)).tolist()
        interpolated_y_dot = np.zeros_like(interpolated_y).tolist()
        interpolated_y_dotdot = np.zeros_like(interpolated_y).tolist()
        return interpolated_y, interpolated_y_dot, interpolated_y_dotdot, interpolated_x

    def _cubic_solve(self):
        N = self._n_pnts - 1
        X = np.zeros((4 * N, 4 * N))
        Y = np.zeros((4 * N, self._n_dim))
        ridx = 0
        for i in range(N):  # left end (N in total)
            X[ridx, i * 4:i * 4 + 4] = np.array([i ** 3, i ** 2, i, 1])
            Y[ridx, :] = self._path_array[i][:]
            ridx += 1
        for i in range(N):  # right end (N in total)
            X[ridx, i * 4:i * 4 + 4] = np.array([(i + 1) ** 3, (i + 1) ** 2, i + 1, 1])
            Y[ridx, :] = self._path_array[i + 1][:]
            ridx += 1
        for i in range(N - 1):  # equal speed at key points
            X[ridx, i * 4:i * 4 + 4] = np.array([3 * (i + 1) ** 2, 2 * (i + 1), 1, 0])
            X[ridx, (i + 1) * 4:(i + 1) * 4 + 4] = np.array([-3 * (i + 1) ** 2, -2 * (i + 1), -1, 0])
            Y[ridx, :] = np.zeros(self._n_dim)
            ridx += 1
        for i in range(N - 1):  # equal acceleration at key points
            X[ridx, i * 4:i * 4 + 4] = np.array([3 * (i + 1), 1, 0, 0])
            X[ridx, (i + 1) * 4:(i + 1) * 4 + 4] = np.array([-3 * (i + 1), -1, 0, 0])
            Y[ridx, :] = np.zeros(self._n_dim)
            ridx += 1
        X[-2, :4] = np.array([3 * 0 ** 2, 2 * 0, 1, 0])
        Y[-2, :] = np.zeros(self._n_dim)  # zero init speed
        X[-1, -4:] = np.array([3 * (self._n_pnts - 1) ** 2, 2 * (self._n_pnts - 1), 1, 0])
        Y[-1, :] = np.zeros(self._n_dim)  # zero end speed
        A = np.linalg.solve(X, Y)
        return A

    def _cubic_interpolate(self, A, samples_list):
        """
        :param A: cubic coefficient matrix
        :param samples_list: a list of 1xn_jnts nparray, with each element holding the samples need to be interpolated
        :return: interpolated_pos, interpolated_spd, interpolated_acc, interpolated_x (x axis ticks)
        author: weiwei
        date: 20210712
        """
        n_sections = self._n_pnts - 1
        interpolated_y = []
        interpolated_y_dot = []
        interpolated_y_dotdot = []
        interpolated_x = []
        for i in range(n_sections):
            if i == n_sections - 1:  # last
                interpolated_x_sec = np.array((samples_list[i] + i).tolist())
            else:
                interpolated_x_sec = np.array((samples_list[i] + i).tolist()[:-1])
            interpolated_x += interpolated_x_sec
            # y
            tmp_x_items = np.ones((interpolated_x_sec.shape[0], 4))
            tmp_x_items[:, 0] = interpolated_x_sec ** 3
            tmp_x_items[:, 1] = interpolated_x_sec ** 2
            tmp_x_items[:, 2] = interpolated_x_sec
            interpolated_y_i = tmp_x_items.dot(A[4 * i:4 * i + 4, :])
            interpolated_y += interpolated_y_i.tolist()
            # y dot
            tmp_x_items_dot = np.zeros((interpolated_x_sec.shape[0], 4))
            tmp_x_items_dot[:, 0] = 3 * interpolated_x_sec ** 2
            tmp_x_items_dot[:, 1] = 2 * interpolated_x_sec
            tmp_x_items_dot[:, 2] = 1
            interpolated_y_dot_i = tmp_x_items_dot.dot(A[4 * i:4 * i + 4, :])
            interpolated_y_dot += interpolated_y_dot_i.tolist()
            # y dot dot
            tmp_x_items_dotdot = np.zeros((interpolated_x_sec.shape[0], 4))
            tmp_x_items_dotdot[:, 0] = 6 * interpolated_x_sec
            tmp_x_items_dotdot[:, 1] = 2
            interpolated_y_dotdot_i = tmp_x_items_dotdot.dot(A[4 * i:4 * i + 4, :])
            interpolated_y_dotdot += interpolated_y_dotdot_i.tolist()
        return interpolated_y, interpolated_y_dot, interpolated_y_dotdot, interpolated_x

    def _quintic_solve(self):
        N = self._n_pnts - 1
        X = np.zeros((5 * N, 5 * N))
        Y = np.zeros((5 * N, self._n_dim))
        ridx = 0
        for i in range(N):  # left end (N in total)
            X[ridx, i * 5:i * 5 + 5] = np.array([i ** 4, i ** 3, i ** 2, i, 1])
            Y[ridx, :] = self._path_array[i][:]
            ridx += 1
        for i in range(N):  # right end (N in total)
            X[ridx, i * 5:i * 5 + 5] = np.array([(i + 1) ** 4, (i + 1) ** 3, (i + 1) ** 2, i + 1, 1])
            Y[ridx, :] = self._path_array[i + 1][:]
            ridx += 1
        for i in range(N - 1):  # speed 0
            X[ridx, i * 5:i * 5 + 5] = np.array([4 * (i + 1) ** 3, 3 * (i + 1) ** 2, 2 * (i + 1), 1, 0])
            X[ridx, (i + 1) * 5:(i + 1) * 5 + 5] = np.array([-4 * (i + 1) ** 3, -3 * (i + 1) ** 2, -2 * (i + 1), -1, 0])
            Y[ridx, :] = np.zeros(self._n_dim)
            ridx += 1
        for i in range(N - 1):  # acc 0
            X[ridx, i * 5:i * 5 + 5] = np.array([6 * (i + 1) ** 2, 3 * (i + 1), 1, 0, 0])
            X[ridx, (i + 1) * 5:(i + 1) * 5 + 5] = np.array([-6 * (i + 1) ** 2, -3 * (i + 1), -1, 0, 0])
            Y[ridx, :] = np.zeros(self._n_dim)
            ridx += 1
        for i in range(N - 1):  # jerk 0
            X[ridx, i * 5:i * 5 + 5] = np.array([4 * (i + 1), 1, 0, 0, 0])
            X[ridx, (i + 1) * 5:(i + 1) * 5 + 5] = np.array([-4 * (i + 1), -1, 0, 0, 0])
            Y[ridx, :] = np.zeros(self._n_dim)
            ridx += 1
        X[-3, :5] = np.array([4 * 0 ** 3, 3 * 0 ** 2, 2 * 0, 1, 0])
        Y[-3, :] = np.zeros(self._n_dim)  # zero init speed
        X[-2, -5:] = np.array([4 * (self._n_pnts - 1) ** 3, 3 * (self._n_pnts - 1) ** 2, 2 * (self._n_pnts - 1), 1, 0])
        Y[-2, :] = np.zeros(self._n_dim)  # zero end speed
        X[-1, -5:] = np.array([6 * (self._n_pnts - 1) ** 2, 3 * (self._n_pnts - 1), 1, 0, 0])
        Y[-1, :] = np.zeros(self._n_dim)  # zero end acc
        A = np.linalg.solve(X, Y)
        return A

    def _quintic_interpolate(self, A, samples_list):
        """
        :param A: quintic coefficient matrix
        :param samples_list: a list of 1xn_jnts nparray, with each element holding the samples need to be interpolated
        :return: interpolated_pos, interpolated_spd, interpolated_acc, interpolated_x (x axis ticks)
        author: weiwei
        date: 20210712
        """
        n_sections = self._n_pnts - 1
        interpolated_y = []
        interpolated_y_dot = []
        interpolated_y_dotdot = []
        interpolated_x = []
        for i in range(n_sections):
            if i == n_sections - 1:  # last
                interpolated_x_sec = np.array((samples_list[i] + i).tolist())
            else:
                interpolated_x_sec = np.array((samples_list[i] + i).tolist()[:-1])
            interpolated_x += interpolated_x_sec
            # y
            tmp_x_items = np.ones((interpolated_x_sec.shape[0], 5))
            tmp_x_items[:, 0] = interpolated_x_sec ** 4
            tmp_x_items[:, 1] = interpolated_x_sec ** 3
            tmp_x_items[:, 2] = interpolated_x_sec ** 2
            tmp_x_items[:, 3] = interpolated_x_sec
            interpolated_y_i = tmp_x_items.dot(A[5 * i:5 * i + 5, :])
            interpolated_y += interpolated_y_i.tolist()
            # y dot
            tmp_x_items_dot = np.zeros((interpolated_x_sec.shape[0], 5))
            tmp_x_items_dot[:, 0] = 4 * interpolated_x_sec ** 3
            tmp_x_items_dot[:, 1] = 3 * interpolated_x_sec ** 2
            tmp_x_items_dot[:, 2] = 2 * interpolated_x_sec
            tmp_x_items_dot[:, 3] = 1
            interpolated_y_dot_i = tmp_x_items_dot.dot(A[5 * i:5 * i + 5, :])
            interpolated_y_dot += interpolated_y_dot_i.tolist()
            # y dot dot
            tmp_x_items_dotdot = np.zeros((interpolated_x_sec.shape[0], 5))
            tmp_x_items_dotdot[:, 0] = 12 * interpolated_x_sec ** 2
            tmp_x_items_dotdot[:, 1] = 6 * interpolated_x_sec
            tmp_x_items_dotdot[:, 2] = 2
            interpolated_y_dotdot_i = tmp_x_items_dotdot.dot(A[5 * i:5 * i + 5, :])
            interpolated_y_dotdot += interpolated_y_dotdot_i.tolist()
        return interpolated_y, interpolated_y_dot, interpolated_y_dotdot, interpolated_x

    def interpolate(self, path, control_frequency=.005, time_interval=1.0):
        """
        :param path:
        :param control_frequency:
        :param time_interval: motion time between two adjacent poses in the path, metrics = second
        :return:
        author: weiwei
        date: 20210712
        """
        self._path_array = np.array(path)
        self._n_pnts, self._n_dim = self._path_array.shape
        samples = np.linspace(0,
                              time_interval,
                              math.floor(time_interval / control_frequency),
                              endpoint=True) / time_interval
        samples_list = [samples] * (self._n_pnts - 1)
        A = self._solve()
        return self._interpolate(A, samples_list)

    def interpolate_by_max_jntspeed(self, path, control_frequency=.005, max_jntspeed=1.0472):
        """
        TODO: prismatic motor speed is not considered
        :param path:
        :param control_frequency:
        :param max_jntspeed: max jnt speed between two adjacent poses in the path, metrics = rad/second, 1.0472=60^o/s
        :return:
        author: weiwei
        date: 20210712
        """
        self._path_array = np.array(path)
        self._n_pnts, self._n_dim = self._path_array.shape
        samples_list = []
        for i in range(self._n_pnts - 1):
            tmp_time_interval = max(abs(path[i + 1] - path[i])) / max_jntspeed
            samples = np.linspace(0,
                                  tmp_time_interval,
                                  math.floor(tmp_time_interval / control_frequency),
                                  endpoint=True) / tmp_time_interval
            samples_list.append(samples)
        A = self._solve()
        return self._interpolate(A, samples_list)
