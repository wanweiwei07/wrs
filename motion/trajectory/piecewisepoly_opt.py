import scipy.interpolate as sinter
import numpy as np
import math
import time
from scipy.optimize import minimize


# interpolation ignored during optimization
class PiecewisePolyOpt(object):

    def __init__(self, method="linear"):
        self._toggle_debug_fine = False
        self._x = None
        self._path_array = None
        self._n_pnts = None
        self._n_jnts = None
        self._control_frequency = None
        self._start_vels = 0
        self._start_accs = 0
        self._end_vels = 0
        self._goal_acc = 0
        self._max_vels = None
        self._max_accs = None
        self.change_method(method=method)

    def _optimization_goal(self, time_intervals):
        return np.sum(time_intervals)

    def _constraint_spdacc(self, time_intervals):
        self._x = [0]
        tmp_total_time = 0
        samples_list = []
        for i in range(len(time_intervals)):
            tmp_time_interval = time_intervals[i]
            n_samples = math.ceil(tmp_time_interval / .008)
            if n_samples <= 1:
                n_samples = 2
            samples = np.linspace(0,
                                  tmp_time_interval,
                                  n_samples,
                                  endpoint=True)
            samples_list.append(samples + self._x[-1])
            self._x.append(tmp_time_interval + tmp_total_time)
            tmp_total_time += tmp_time_interval
        interpolated_x = []
        for i in range(len(samples_list)):
            if i == len(samples_list) - 1:  # last
                interpolated_x_sec = (samples_list[i]).tolist()
            else:
                interpolated_x_sec = (samples_list[i]).tolist()[:-1]
            interpolated_x += interpolated_x_sec
        A = self._solve()
        interpolated_y = A(np.array(interpolated_x)).tolist()
        interpolated_y_d1 = A(np.array(interpolated_x), 1).tolist()
        interpolated_y_d2 = A(np.array(interpolated_x), 2).tolist()
        interpolated_y_d3 = A(np.array(interpolated_x), 3).tolist()
        original_x = self._x
        if self._toggle_debug_fine:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(4, figsize=(35, 47.5))
            fig.tight_layout(pad=.7)
            axs[0].plot(interpolated_x, interpolated_y, 'o')
            for xc in original_x:
                axs[0].axvline(x=xc)
            # axs[0].plot(np.arange(len(jnt_values_list)), jnt_values_list, '--o')
            axs[1].plot(interpolated_x, interpolated_y_d1)
            for xc in original_x:
                axs[1].axvline(x=xc)
            axs[2].plot(interpolated_x, interpolated_y_d2)
            for xc in original_x:
                axs[2].axvline(x=xc)
            axs[3].plot(interpolated_x, interpolated_y_d3)
            for xc in original_x:
                axs[3].axvline(x=xc)
            for xc in original_x:
                axs[2].axvline(x=xc)
            plt.show()
        acc_diff = np.tile(self._max_accs, (len(interpolated_y_d2), 1)) - np.abs(interpolated_y_d2)
        # jks_diff = np.tile(self._max_accs, (len(interpolated_y_d3), 1)) - np.abs(interpolated_y_d3)
        print(np.sum(acc_diff[acc_diff < 0] ** 2))
        return np.sum(acc_diff[acc_diff < 0] ** 2)
        # The following code only uses give points
        # for i in range(self._n_pnts - 1):
        #     tmp_time_interval = time_intervals[i]
        #     self._x.append(tmp_time_interval + tmp_total_time)
        #     tmp_total_time += tmp_time_interval
        # A = self._solve()
        # interpolated_y_d2 = A(np.array(self._x), 2).tolist()
        # acc_diff = np.tile(self._max_accs, (len(interpolated_y_d2),1)) - np.abs(interpolated_y_d2)
        # print(np.sum(acc_diff[acc_diff<0]**2))
        # return np.sum(acc_diff[acc_diff<0]**2)

    def _solve_opt(self, method='SLSQP'):
        """
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param method:
        :return:
        """
        constraints = []
        constraints.append({'end_type': 'eq', 'fun': self._constraint_spdacc})
        bounds = []
        for i in range(len(self._seed_time_intervals)):
            bounds.append((self._seed_time_intervals[i], None))
        sol = minimize(self._optimization_goal,
                       self._seed_time_intervals,
                       method=method,
                       bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 10e6, "disp": True},
                       tol=.01)
        if sol.success:
            return sol.x, sol.fun
        else:
            return None, None

    def change_method(self, method="cubic"):
        self.method = method
        if method == "linear":
            self._solve = self._linear_solve
        if method == "quadratic":
            self._solve = self._quadratic_solve
        elif method == "cubic":
            self._solve = self._cubic_solve
        elif method == "quintic":
            self._solve = self._quintic_solve

    def _linear_solve(self):
        return sinter.make_interp_spline(self._x, self._path_array, k=1, axis=0)

    def _quadratic_solve(self):
        bc_type = [(1, self._start_vels), None]
        return sinter.make_interp_spline(self._x, self._path_array, k=2, axis=0, bc_type=bc_type)

    def _cubic_solve(self):
        bc_type = [(1, self._start_vels), (1, self._end_vels)]
        return sinter.make_interp_spline(self._x, self._path_array, k=3, axis=0, bc_type=bc_type)

    def _quintic_solve(self):
        bc_type = [[(1, self._start_vels), (2, self._start_accs)],
                   [(1, self._end_vels), (2, self._end_accs)]]
        return sinter.make_interp_spline(self._x, self._path_array, k=5, axis=0, bc_type=bc_type)

    def _trapezoid_solve(self):
        pass

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
                interpolated_x_sec = (samples_list[i]).tolist()
            else:
                interpolated_x_sec = (samples_list[i]).tolist()[:-1]
            interpolated_x += interpolated_x_sec
        interpolated_y = A(np.array(interpolated_x)).tolist()
        interpolated_y_d1 = A(np.array(interpolated_x), 1).tolist()
        interpolated_y_d2 = A(np.array(interpolated_x), 2).tolist()
        interpolated_y_d3 = A(np.array(interpolated_x), 3).tolist()
        original_x = self._x
        return interpolated_y, interpolated_y_d1, interpolated_y_d2, interpolated_y_d3, interpolated_x, original_x

    def _trapezoid_interpolate(self):
        pass

    def _remove_duplicate(self, path):
        new_path = []
        for i, pose in enumerate(path):
            if i < len(path) - 1 and not np.allclose(pose, path[i + 1]):
                new_path.append(pose)
        new_path.append(path[-1])
        return new_path

    def interpolate(self, control_frequency, time_intervals):
        self._x = [0]
        tmp_total_time = 0
        samples_list = []
        samples_back_index_x = []
        for i in range(self._n_pnts - 1):
            tmp_time_interval = time_intervals[i]
            n_samples = math.floor(tmp_time_interval / control_frequency)
            if n_samples <= 1:
                n_samples = 2
            samples = np.linspace(0,
                                  tmp_time_interval,
                                  n_samples,
                                  endpoint=True)
            for j in range(n_samples):
                samples_back_index_x.append(i)
            samples_list.append(samples + self._x[-1])
            self._x.append(tmp_time_interval + tmp_total_time)
            tmp_total_time += tmp_time_interval
        A = self._solve()
        interpolated_confs, interpolated_vels, interpolated_accs, interpolated_jks, interpolated_x, original_x = \
            self._interpolate(A, samples_list)
        return interpolated_confs, interpolated_vels, interpolated_accs, interpolated_jks, interpolated_x, original_x, samples_back_index_x

    def interpolate_by_max_spdacc(self,
                                  path,
                                  control_frequency=.005,
                                  start_vels=None,
                                  end_vels=None,
                                  start_accs=None,
                                  end_accs=None,
                                  max_vels=None,
                                  max_accs=None,
                                  toggle_debug_fine=False,
                                  toggle_debug=True):
        """
        TODO: prismatic motor speed is not considered
        :param path:
        :param control_frequency:
        :param start_vels:
        :param end_vels:
        :param start_accs:
        :param end_accs:
        :param max_jnts_vels: max joint speed between two adjacent poses in the path, math.pi if None
        :param max_jnts_accs: max joint speed between two adjacent poses in the path, math.pi if None
        :return:
        author: weiwei
        date: 20210712, 20211012
        """
        self._toggle_debug_fine = toggle_debug_fine
        path = self._remove_duplicate(path)
        self._path_array = np.array(path)
        self._n_pnts, self._n_jnts = self._path_array.shape
        self._control_frequency = control_frequency
        if start_vels is None:
            start_vels = [0] * path[0].shape[0]
        if start_accs is None:
            start_accs = [0] * path[0].shape[0]
        if end_vels is None:
            end_vels = [0] * path[0].shape[0]
        if end_accs is None:
            end_accs = [0] * path[0].shape[0]
        self._start_vels = np.asarray(start_vels)
        self._start_accs = np.asarray(start_accs)
        self._end_vels = np.asarray(end_vels)
        self._end_accs = np.asarray(end_accs)
        if max_vels is None:
            max_vels = [math.pi * 2 / 3] * path[0].shape[0]
        if max_accs is None:
            max_accs = [math.pi] * path[0].shape[0]
        self._max_vels = np.asarray(max_vels)
        self._max_accs = np.asarray(max_accs)
        # initialize time inervals
        time_intervals = []
        for i in range(self._n_pnts - 1):
            pose_diff = abs(path[i + 1] - path[i])
            tmp_time_interval = np.max(pose_diff / max_vels)
            time_intervals.append(tmp_time_interval)
        time_intervals = np.asarray(time_intervals)
        print("seed total time", np.sum(time_intervals))
        # # time scaling
        # # interpolate
        # interpolated_confs, interpolated_vels, interpolated_accs, interpolated_x, original_x, samples_back_index_x = \
        #     self.interpolate(control_frequency=control_frequency, time_intervals=time_intervals,
        #                      toggle_dbg=toggle_debug_fine)
        # while True:
        #     samples_back_index_x = np.asarray(samples_back_index_x)
        #     interpolated_accs_abs = np.asarray(np.abs(interpolated_accs))
        #     diff_accs = np.tile(max_accs, (len(interpolated_accs_abs), 1)) - interpolated_accs_abs
        #     selection = np.where(np.min(diff_accs, axis=1) < 0)
        #     if len(selection[0]) > 0:
        #         time_intervals += .001
        #         x_sel = np.unique(samples_back_index_x[selection[0] - 1])
        #         time_intervals[x_sel] += .001
        #     else:
        #         break
        #     interpolated_confs, interpolated_vels, interpolated_accs, interpolated_x, original_x, samples_back_index_x = \
        #         self.interpolate(control_frequency=control_frequency, time_intervals=time_intervals,
        #                          toggle_dbg=toggle_debug_fine)

        self._seed_time_intervals = time_intervals
        time_intervals, _ = self._solve_opt()
        # interpolate
        interpolated_confs, interpolated_vels, interpolated_accs, interpolated_jks, interpolated_x, original_x, samples_back_index_x = self.interpolate(
            control_frequency=control_frequency, time_intervals=time_intervals)
        print("final total time", original_x[-1])
        if toggle_debug:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(4, figsize=(35, 47.5))
            fig.tight_layout(pad=.7)
            axs[0].plot(interpolated_x, interpolated_confs, 'o')
            for xc in original_x:
                axs[0].axvline(x=xc)
            # axs[0].plot(np.arange(len(jnt_values_list)), jnt_values_list, '--o')
            axs[1].plot(interpolated_x, interpolated_vels)
            for xc in original_x:
                axs[1].axvline(x=xc)
            axs[2].plot(interpolated_x, interpolated_accs)
            for xc in original_x:
                axs[2].axvline(x=xc)
            axs[3].plot(interpolated_x, interpolated_jks)
            for xc in original_x:
                axs[3].axvline(x=xc)
            plt.show()
        return interpolated_confs
