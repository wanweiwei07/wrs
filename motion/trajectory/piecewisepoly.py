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
            self._interpolate = self._interpolate
        if method == "quadratic":
            self._solve = self._quadratic_solve
            self._interpolate = self._interpolate
        elif method == "cubic":
            self._solve = self._cubic_solve
            self._interpolate = self._cubic_interpolate
        elif method == "quintic":
            self._solve = self._quintic_solve
            self._interpolate = self._quintic_interpolate

    def _linear_solve(self):
        x = self._x
        y = self._path_array
        # return sinter.interp1d(x, y, kind="linear", axis=0)
        return sinter.make_interp_spline(x, y, k=1, axis=0)

    def _quadratic_solve(self):
        x = self._x
        y = self._path_array
        # return sinter.interp1d(x, y, kind="linear", axis=0)
        return sinter.make_interp_spline(x, y, k=2, axis=0, bc_type=['clamped', None])

    def _interpolate(self, A, samples_list):
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
                interpolated_x_sec = (samples_list[i] +  self._x[i]).tolist()
            else:
                interpolated_x_sec = (samples_list[i] +  self._x[i]).tolist()[:-1]
            interpolated_x += interpolated_x_sec
        interpolated_y = A(np.array(interpolated_x)).tolist()
        # interpolated_y_dot = np.zeros_like(interpolated_y).tolist()
        # interpolated_y_dotdot = np.zeros_like(interpolated_y).tolist()
        interpolated_y_dot = A(np.array(interpolated_x), 1).tolist()
        interpolated_y_dotdot = A(np.array(interpolated_x), 2).tolist()
        original_x = self._x
        return interpolated_y, interpolated_y_dot, interpolated_y_dotdot, interpolated_x, original_x

    def _trapezoid_solve(self):
        pass

    def _trapezoid_interpolate(self):
        pass

    def _cubic_solve(self):
        x = np.array(self._x)
        y = self._path_array
        return sinter.make_interp_spline(x, y, k=3, axis=0, bc_type='clamped')

    def _cubic_interpolate(self, A, samples_list):
        """
        :param A: a cubic call back function since we are using scipy
        :param samples_list: a list of 1xn_jnts nparray, with each element holding the samples need to be interpolated
        :return: interpolated_pos, interpolated_spd, interpolated_acc, interpolated_x (x axis ticks)
        author: weiwei
        date: 20210803
        """
        n_sections = self._n_pnts - 1
        interpolated_x = []
        for i in range(n_sections):
            if i == n_sections - 1:  # last
                interpolated_x_sec = (samples_list[i] + self._x[i]).tolist()
            else:
                interpolated_x_sec = (samples_list[i] + self._x[i]).tolist()[:-1]
            interpolated_x += interpolated_x_sec
        interpolated_y = A(np.array(interpolated_x)).tolist()
        interpolated_y_dot = A(np.array(interpolated_x), 1).tolist()
        interpolated_y_dotdot = A(np.array(interpolated_x), 2).tolist()
        original_x = self._x
        print(original_x)
        return interpolated_y, interpolated_y_dot, interpolated_y_dotdot, interpolated_x, original_x

    # def _quintic_solve(self):
    #     x = range(self._path_array.shape[0])
    #     y = self._path_array
    #     return sinter.make_interp_spline(x, y, k=5, axis=0, bc_type=([(1, 0.0)], [(1, 0.0)]))
    #
    # def _quintic_interpolate(self, A, samples_list):
    #     """
    #     :param A: a cubic call back function since we are using scipy
    #     :param samples_list: a list of 1xn_jnts nparray, with each element holding the samples need to be interpolated
    #     :return: interpolated_pos, interpolated_spd, interpolated_acc, interpolated_x (x axis ticks)
    #     author: weiwei
    #     date: 20210803
    #     """
    #     n_sections = self._n_pnts - 1
    #     interpolated_x = []
    #     for i in range(n_sections):
    #         if i == n_sections - 1:  # last
    #             interpolated_x_sec = (samples_list[i] + i).tolist()
    #         else:
    #             interpolated_x_sec = (samples_list[i] + i).tolist()[:-1]
    #         interpolated_x += interpolated_x_sec
    #     interpolated_y = A(np.array(interpolated_x)).tolist()
    #     interpolated_y_dot = A(np.array(interpolated_x), 1).tolist()
    #     interpolated_y_dotdot = A(np.array(interpolated_x), 2).tolist()
    #     return interpolated_y, interpolated_y_dot, interpolated_y_dotdot, interpolated_x

    # def _cubic_solve(self):
    #     N = self._n_pnts - 1
    #     X = np.zeros((4 * N, 4 * N))
    #     Y = np.zeros((4 * N, self._n_dim))
    #     ridx = 0
    #     for i in range(N):  # left end (N in total)
    #         X[ridx, i * 4:i * 4 + 4] = np.array([i ** 3, i ** 2, i, 1])
    #         Y[ridx, :] = self._path_array[i][:]
    #         ridx += 1
    #     for i in range(N):  # right end (N in total)
    #         X[ridx, i * 4:i * 4 + 4] = np.array([(i + 1) ** 3, (i + 1) ** 2, i + 1, 1])
    #         Y[ridx, :] = self._path_array[i + 1][:]
    #         ridx += 1
    #     for i in range(N - 1):  # equal speed at key points
    #         X[ridx, i * 4:i * 4 + 4] = np.array([3 * (i + 1) ** 2, 2 * (i + 1), 1, 0])
    #         X[ridx, (i + 1) * 4:(i + 1) * 4 + 4] = np.array([-3 * (i + 1) ** 2, -2 * (i + 1), -1, 0])
    #         Y[ridx, :] = np.zeros(self._n_dim)
    #         ridx += 1
    #     for i in range(N - 1):  # equal acceleration at key points
    #         X[ridx, i * 4:i * 4 + 4] = np.array([3 * (i + 1), 1, 0, 0])
    #         X[ridx, (i + 1) * 4:(i + 1) * 4 + 4] = np.array([-3 * (i + 1), -1, 0, 0])
    #         Y[ridx, :] = np.zeros(self._n_dim)
    #         ridx += 1
    #     X[-2, :4] = np.array([3 * 0 ** 2, 2 * 0, 1, 0])
    #     Y[-2, :] = np.zeros(self._n_dim)  # zero init speed
    #     X[-1, -4:] = np.array([3 * (self._n_pnts - 1) ** 2, 2 * (self._n_pnts - 1), 1, 0])
    #     Y[-1, :] = np.zeros(self._n_dim)  # zero end speed
    #     A = np.linalg.solve(X, Y)
    #     return A
    #
    # def _cubic_interpolate(self, A, samples_list):
    #     """
    #     :param A: cubic coefficient matrix
    #     :param samples_list: a list of 1xn_jnts nparray, with each element holding the samples need to be interpolated
    #     :return: interpolated_pos, interpolated_spd, interpolated_acc, interpolated_x (x axis ticks)
    #     author: weiwei
    #     date: 20210712
    #     """
    #     n_sections = self._n_pnts - 1
    #     interpolated_y = []
    #     interpolated_y_dot = []
    #     interpolated_y_dotdot = []
    #     interpolated_x = []
    #     for i in range(n_sections):
    #         if i == n_sections - 1:  # last
    #             interpolated_x_sec = (samples_list[i] + i).tolist()
    #         else:
    #             interpolated_x_sec = (samples_list[i] + i).tolist()[:-1]
    #         interpolated_x += interpolated_x_sec
    #         interpolated_x_sec = np.asarray(interpolated_x_sec)
    #         # y
    #         tmp_x_items = np.ones((interpolated_x_sec.shape[0], 4))
    #         tmp_x_items[:, 0] = interpolated_x_sec ** 3
    #         tmp_x_items[:, 1] = interpolated_x_sec ** 2
    #         tmp_x_items[:, 2] = interpolated_x_sec
    #         interpolated_y_i = tmp_x_items.dot(A[4 * i:4 * i + 4, :])
    #         interpolated_y += interpolated_y_i.tolist()
    #         # y dot
    #         tmp_x_items_dot = np.zeros((interpolated_x_sec.shape[0], 4))
    #         tmp_x_items_dot[:, 0] = 3 * interpolated_x_sec ** 2
    #         tmp_x_items_dot[:, 1] = 2 * interpolated_x_sec
    #         tmp_x_items_dot[:, 2] = 1
    #         interpolated_y_dot_i = tmp_x_items_dot.dot(A[4 * i:4 * i + 4, :])
    #         interpolated_y_dot += interpolated_y_dot_i.tolist()
    #         # y dot dot
    #         tmp_x_items_dotdot = np.zeros((interpolated_x_sec.shape[0], 4))
    #         tmp_x_items_dotdot[:, 0] = 6 * interpolated_x_sec
    #         tmp_x_items_dotdot[:, 1] = 2
    #         interpolated_y_dotdot_i = tmp_x_items_dotdot.dot(A[4 * i:4 * i + 4, :])
    #         interpolated_y_dotdot += interpolated_y_dotdot_i.tolist()
    #     return interpolated_y, interpolated_y_dot, interpolated_y_dotdot, interpolated_x

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
                interpolated_x_sec = (samples_list[i] + i).tolist()
            else:
                interpolated_x_sec = (samples_list[i] + i).tolist()[:-1]
            interpolated_x += interpolated_x_sec
            interpolated_x_sec = np.asarray(interpolated_x_sec)
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

    def interpolate_by_time_interval(self, path, control_frequency=.005, time_interval=1.0):
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

    def interpolate_by_max_spdacc(self, path, control_frequency=.005, max_jnts_spd=None, max_jnts_acc=None):
        """
        TODO: prismatic motor speed is not considered
        :param path:
        :param control_frequency:
        :param max_jnts_spd: max jnt speed between two adjacent poses in the path, math.pi if None
        :param max_jnts_acc: max jnt speed between two adjacent poses in the path, math.pi if None
        :return:
        author: weiwei
        date: 20210712
        """
        self._path_array = np.array(path)
        self._n_pnts, self._n_dim = self._path_array.shape
        if max_jnts_spd is None:
            max_jnts_spd = [math.pi*2/3] * path[0].shape[0]
        # if max_jnts_acc is None:
        #     max_jnts_acc = [math.pi]*path[1].shape[0]
        # times_zero_to_maxspd = max_jnts_spd/max_jnts_acc
        # times_maxspd_to_zero = times_zero_to_maxspd
        # dists_zero_to_maxspd = .5*max_jnts_acc*times_zero_to_maxspd**2
        # dists_maxspd_to_zero = .5*max_jnts_acc*times_maxspd_to_zero**2
        samples_list = []
        self._x = [0]
        tmp_total_time = 0
        for i in range(self._n_pnts - 1):
            tmp_time_interval = max(abs(path[i + 1] - path[i]) / np.asarray(max_jnts_spd))
            self._x.append(tmp_time_interval+tmp_total_time)
            tmp_total_time = tmp_total_time+tmp_time_interval
            n_samples = math.floor(tmp_time_interval / control_frequency)
            print(n_samples)
            if n_samples <= 1:
                n_samples = 2
            samples = np.linspace(0,
                                  tmp_time_interval,
                                  n_samples,
                                  endpoint=True)
            samples_list.append(samples)
        A = self._solve()
        return self._interpolate(A, samples_list)


# TODO
"""
function trapezoidInterpolate(distance,v0,v3,vmax,a,t) {
// assumes t0=0
t1 = (vmax-v0) / a; // time from v0 to vmax (time to reach full speed)
t4 = (max-v3) / a; // time from vmax to v3 (time to brake)
 d1 = v0*t1 + 0.5*a*t1*t1; // distance t0-t1
d2 = v3*t4 + 0.5*a*t4*t4; // distance t2-t3

if( d1+d2 < distance ) {
// plateau at vmax in the middle
tplateau = ( distance – d1 – d2 ) / vmax;
t2 = t1 + tplateau;
t3 = t2 + t4;
} else {
// start breaking before reaching vmax
// http://wikipedia.org/wiki/Classical_mechanics#1-Dimensional_Kinematics
t1 = ( sqrt( 2.0*a*brake_distance + v0*v0 ) – v0 ) / a;
t2 = t1;
t3 = t2 + ( sqrt( 2.0*a*(distance-brake_distance) + v3*v3 ) – v3 ) / a;
}

if(t<t1) {
return v0*t + 0.5*a*t*t;
}
if(t<t2) {
up = v0*t1 + 0.5*a*t1*t1;
plateau = vmax*(t-t1);
return up+plateau;
}
if(t<t3) {
up = v0*t1 + 0.5*a*t1*t1;
plateau = vmax*(t2-t1);
t4=t-t2;
v2 = accel * t1;
down = v2*t4 + 0.5*a*t4*t4;
return up+plateau+down;
}
return distance;
}
"""
