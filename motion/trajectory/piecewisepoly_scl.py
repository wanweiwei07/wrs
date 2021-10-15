import copy

import scipy.interpolate as sinter
import numpy as np
import math


class PiecewisePolyScl(object):

    def __init__(self, method="linear"):
        self.change_method(method=method)

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
        return sinter.make_interp_spline(self._x, self._path_array, k=2, axis=0, bc_type=['clamped', None])

    def _cubic_solve(self):
        return sinter.make_interp_spline(self._x, self._path_array, k=3, axis=0, bc_type='clamped')

    def _quintic_solve(self):
        bc_type = [[(1, np.zeros(self._n_dim)), (2, np.zeros(self._n_dim))],
                   [(1, np.zeros(self._n_dim)), (2, np.zeros(self._n_dim))]]
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
        interpolated_y_dot = A(np.array(interpolated_x), 1).tolist()
        interpolated_y_dotdot = A(np.array(interpolated_x), 2).tolist()
        original_x = self._x
        return interpolated_y, interpolated_y_dot, interpolated_y_dotdot, interpolated_x, original_x

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
            for j in range(len(samples) - 1):
                samples_back_index_x.append(i)
            samples_list.append(samples + self._x[-1])
            self._x.append(tmp_time_interval + tmp_total_time)
            tmp_total_time += tmp_time_interval
        samples_back_index_x.append(self._n_pnts - 1)
        A = self._solve()
        interpolated_confs, interpolated_spds, interpolated_accs, interpolated_x, original_x = \
            self._interpolate(A, samples_list)
        return interpolated_confs, interpolated_spds, interpolated_accs, interpolated_x, original_x, samples_back_index_x

    def interpolate_by_max_spdacc(self,
                                  path,
                                  control_frequency=.005,
                                  max_spds=None,
                                  max_accs=None,
                                  toggle_debug_fine=False,
                                  toggle_debug=True):
        """
        TODO: prismatic motor speed is not considered
        :param path:
        :param control_frequency:
        :param max_spds: max jnt speed between two adjacent poses in the path, math.pi if None
        :param max_accs: max jnt speed between two adjacent poses in the path, math.pi if None
        :return:
        author: weiwei
        date: 20210712, 20211012
        """
        path = self._remove_duplicate(path)
        self._path_array = np.array(path)
        self._n_pnts, self._n_dim = self._path_array.shape
        if max_spds is None:
            max_spds = [math.pi * 2 / 3] * path[0].shape[0]
        if max_accs is None:
            max_accs = [math.pi] * path[0].shape[0]
        max_spds = np.asarray(max_spds)
        max_accs = np.asarray(max_accs)
        self._max_spds = max_spds
        self._max_accs = max_accs
        # initialize time inervals
        time_intervals = []
        for i in range(self._n_pnts - 1):
            pose_diff = abs(path[i + 1] - path[i])
            tmp_time_interval = np.max(pose_diff / max_spds)
            time_intervals.append(tmp_time_interval)
        time_intervals = np.array(time_intervals)
        # interpolate
        interpolated_confs, interpolated_spds, interpolated_accs, interpolated_x, original_x, samples_back_index_x = \
            self.interpolate(control_frequency=control_frequency, time_intervals=time_intervals)
        print("seed total time", original_x[-1])
        # time scaling for speed
        count = 0
        while True:
            count+=1
            if count > 50:
                toggle_debug_fine=True
                break
            samples_back_index_x = np.asarray(samples_back_index_x)
            interpolated_spds_abs = np.asarray(np.abs(interpolated_spds))
            diff_spds = np.tile(max_spds, (len(interpolated_spds_abs), 1)) - interpolated_spds_abs
            selection_spds = np.where(np.min(diff_spds, axis=1) < 1e-6)
            # print("max spd ", max_spds, "sel_spd ", selection_spds)
            # print("samples_back_index_x ", samples_back_index_x)
            if len(selection_spds[0]) > 0:
                x_sel_spds = np.unique(samples_back_index_x[selection_spds[0]])
                # print("spd ", x_sel_spds)
                time_intervals[x_sel_spds] += .001*((1/time_intervals[x_sel_spds])/np.max(1/time_intervals[x_sel_spds]))
            else:
                break
            if toggle_debug_fine:
                print("toggle_debug_fine")
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(3, figsize=(20, 40))
                fig.tight_layout(pad=.7)
                axs[0].plot(interpolated_x, interpolated_confs, 'o')
                for xc in original_x:
                    axs[0].axvline(x=xc)
                # axs[0].plot(np.arange(len(jnt_values_list)), jnt_values_list, '--o')
                axs[1].plot(interpolated_x, interpolated_spds)
                for xc in original_x:
                    axs[1].axvline(x=xc)
                for ys in self._max_spds:
                    axs[1].axhline(y=ys)
                    axs[1].axhline(y=-ys)
                axs[2].plot(interpolated_x, interpolated_accs)
                for xc in original_x:
                    axs[2].axvline(x=xc)
                for ys in self._max_accs:
                    axs[2].axhline(y=ys)
                    axs[2].axhline(y=-ys)
                for i in x_sel_spds:
                    cx = (original_x[i] + original_x[i + 1]) / 2
                    axs[1].axvline(cx, linewidth=10, color='r', alpha=.1)
                plt.show()
            interpolated_confs, interpolated_spds, interpolated_accs, interpolated_x, original_x, samples_back_index_x = \
                self.interpolate(control_frequency=control_frequency/10, time_intervals=time_intervals)
        # time scaling for acceleration
        while True:
            samples_back_index_x = np.asarray(samples_back_index_x)
            interpolated_accs_abs = np.asarray(np.abs(interpolated_accs))
            diff_accs = np.tile(max_accs, (len(interpolated_accs_abs), 1)) - interpolated_accs_abs
            selection_accs = np.where(np.min(diff_accs, axis=1) < 1e-6)
            # ratio_selected_accs = \
            #     np.min(np.tile(max_accs, (len(interpolated_accs_abs), 1)) / interpolated_accs_abs, axis=1)[
            #         selection_accs]
            if len(selection_accs[0]) > 0:
                x_sel_accs = np.unique(samples_back_index_x[selection_accs[0]])
                # print("acc ", x_sel_accs)
                # for i in x_sel_accs:
                #     indices = np.where(x_sel_raw == i)
                #     print(np.max(1 - ratio_selected_accs[indices]))
                #     print(.001 * np.max(1 - ratio_selected_accs[indices]))
                #     time_intervals[i] += .01 * np.max(1 - ratio_selected_accs[indices])
                time_intervals[x_sel_accs] += .001*((1/time_intervals[x_sel_accs])/np.max(1/time_intervals[x_sel_accs]))
            else:
                break
            if toggle_debug_fine:
                print("toggle_debug_fine")
                # import matplotlib.pyplot as plt
                # fig, axs = plt.subplots(3, figsize=(20, 40))
                # fig.tight_layout(pad=.7)
                # axs[0].plot(interpolated_x, interpolated_confs, 'o')
                # for xc in original_x:
                #     axs[0].axvline(x=xc)
                # # axs[0].plot(np.arange(len(jnt_values_list)), jnt_values_list, '--o')
                # axs[1].plot(interpolated_x, interpolated_spds)
                # for xc in original_x:
                #     axs[1].axvline(x=xc)
                # for ys in self._max_spds:
                #     axs[1].axhline(y=ys)
                #     axs[1].axhline(y=-ys)
                # axs[2].plot(interpolated_x, interpolated_accs)
                # for xc in original_x:
                #     axs[2].axvline(x=xc)
                # for ys in self._max_accs:
                #     axs[2].axhline(y=ys)
                #     axs[2].axhline(y=-ys)
                # for i in x_sel_accs:
                #     cx = (original_x[i] + original_x[i + 1]) / 2
                #     axs[2].axvline(cx, linewidth=10, color='b', alpha=.1)
                # plt.show()
            interpolated_confs, interpolated_spds, interpolated_accs, interpolated_x, original_x, samples_back_index_x = \
                self.interpolate(control_frequency=control_frequency/10, time_intervals=time_intervals)
        print("final total time", original_x[-1])
        if toggle_debug:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(3, figsize=(10, 30))
            fig.tight_layout(pad=.7)
            axs[0].plot(interpolated_x, interpolated_confs, 'o')
            for xc in original_x:
                axs[0].axvline(x=xc)
            # axs[0].plot(np.arange(len(jnt_values_list)), jnt_values_list, '--o')
            axs[1].plot(interpolated_x, interpolated_spds)
            for xc in original_x:
                axs[1].axvline(x=xc)
            for ys in max_spds:
                axs[1].axhline(y=ys)
            axs[2].plot(interpolated_x, interpolated_accs)
            for xc in original_x:
                axs[2].axvline(x=xc)
            plt.show()
        return interpolated_confs, interpolated_spds, interpolated_accs, interpolated_x, original_x


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
