import math


class PiecewisePoly(object):

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
        step_array = np.vstack([step ** 3,
                                step ** 2,
                                step,
                                np.ones_like(step)])
        spd_step_array = np.vstack([3 * step ** 2,
                                    2 * step,
                                    np.ones_like(step),
                                    np.zeros_like(step)])
        acc_step_array = np.vstack([6 * step,
                                    2 * np.ones_like(step),
                                    np.zeros_like(step),
                                    np.zeros_like(step)])
        if isinstance(step, np.ndarray):
            return np.dot(self.coeffs_array.T, step_array).T, \
                np.dot(self.coeffs_array.T, spd_step_array).T, \
                np.dot(self.coeffs_array.T, acc_step_array).T
        else:
            return np.dot(self.coeffs_array.T, step_array).T[0][0], \
                np.dot(self.coeffs_array.T, spd_step_array).T[0][0], \
                np.dot(self.coeffs_array.T, acc_step_array).T[0][0]

    def _predict_quintic(self, step):
        """
        step = currenttime/timeinterval
        :return:
        author: weiwei
        date: 20200327
        """
        conf_step_array = np.vstack([step ** 5,
                                     step ** 4,
                                     step ** 3,
                                     step ** 2,
                                     step,
                                     np.ones_like(step)])
        spd_step_array = np.vstack([5 * step ** 4,
                                    4 * step ** 3,
                                    3 * step ** 2,
                                    2 * step,
                                    np.ones_like(step),
                                    np.zeros_like(step)])
        acc_step_array = np.vstack([20 * step ** 3,
                                    12 * step ** 2,
                                    6 * step,
                                    2 * np.ones_like(step),
                                    np.zeros_like(step),
                                    np.zeros_like(step)])
        if isinstance(step, np.ndarray):
            return np.dot(self.coeffs_array.T, conf_step_array).T, \
                np.dot(self.coeffs_array.T, spd_step_array).T, \
                np.dot(self.coeffs_array.T, acc_step_array).T
        else:
            return np.dot(self.coeffs_array.T, conf_step_array).T[0][0], \
                np.dot(self.coeffs_array.T, spd_step_array).T[0][0], \
                np.dot(self.coeffs_array.T, acc_step_array).T[0][0]

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

    def piecewise_interpolation(self, path, ctrl_freq=.005, time_interval=0.1):

        """
        :param path: a 1d array of configurations
        :param ctrl_freq: the program will sample time_intervals/ctrl_freq confs
        :param time_interval: time to move between adjacent joints
        :return:
        author: weiwei
        date: 20200328
        """
        path = np.array(path)
        n_jnts = len(path[0])
        passing_conf_list = []
        passing_spd_list = []
        for id, mid_jnt_values in enumerate(path[:-1]):
            passing_conf_list.append(mid_jnt_values)
            if id == 0:
                passing_spd_list.append(np.zeros_like(mid_jnt_values))
            else:
                prev_jnt_values = path[id - 1]
                next_jnt_values = path[id + 1]
                prev_avg_spd = (mid_jnt_values - prev_jnt_values) / time_interval
                nxt_avg_spd = (next_jnt_values - mid_jnt_values) / time_interval
                pass_spd = (prev_avg_spd + nxt_avg_spd) / 2.0
                # set to 0 if signs are different -> reduces overshoot
                pass_spd[np.where((np.sign(prev_avg_spd) + np.sign(nxt_avg_spd)) == 0.0)] = 0.0
                passing_spd_list.append(pass_spd)
                print("prev spd ", prev_avg_spd)
                print("next spd ", nxt_avg_spd)
                print("avg_spd ", pass_spd)
        passing_conf_list.append(path[-1])  # last pos
        passing_spd_list.append(np.zeros_like(path[-1]))  # last spd
        interp_confs = []
        interp_spds = []
        interp_accs = []
        for id in range(len(passing_conf_list)):
            if id == 0:
                continue
            self.fit(passing_conf_list[id - 1], passing_spd_list[id - 1], passing_conf_list[id],
                     passing_spd_list[id])
            num = math.floor(time_interval / ctrl_freq)
            if num <= 1:
                num = 2
            samples = np.linspace(0,
                                  time_interval,
                                  num,
                                  endpoint=True) / time_interval
            loc_interp_confs, loc_interp_spds, loc_interp_accs = self.predict(samples)
            if id == len(passing_conf_list) - 1:
                interp_confs += loc_interp_confs.tolist()
                interp_spds += loc_interp_spds.tolist()
                interp_accs += loc_interp_accs.tolist()
            else:
                interp_confs += loc_interp_confs.tolist()[:-1]
                interp_spds += loc_interp_spds.tolist()[:-1]
                interp_accs += loc_interp_accs.tolist()[:-1]
        print("interp_spds, quintic ",  interp_spds)
        return interp_confs, interp_spds, interp_accs


if __name__ == '__main__':
    import numpy as np
    from wrs import basis as rm, motion as mip, modeling as mgm
    import matplotlib.pyplot as plt
    import wrs.visualization.panda.world as wd
    import wrs.robot_sim.robots.nova2_wg.nova2wg3 as dnw

    base = wd.World(cam_pos=[3, -1, 1], lookat_pos=[0, 0, 0.5])
    mgm.gen_frame().attach_to(base)
    robot = dnw.Nova2WG3()
    interplated_planner = mip.InterplatedMotion(robot)
    mot_data = interplated_planner.gen_circular_motion(circle_center_pos=np.array([.4, 0, .4]),
                                                       circle_normal_ax=np.array([1, 0, 0]),
                                                       start_tcp_rotmat=rm.rotmat_from_axangle(np.array([0, 1, 0]),
                                                                                               math.pi / 2),
                                                       end_tcp_rotmat=rm.rotmat_from_axangle(np.array([0, 1, 0]),
                                                                                             math.pi / 2),
                                                       radius=0.05)
    x = np.arange(len(mot_data.jv_list))
    plt.figure(figsize=(10, 5))
    plt.plot(x, mot_data.jv_list, '-o')
    # plt.xticks(x)
    # plt.show()
    # for model in mot_data.mesh_list:
    #     model.attach_to(base)
    traj_planner = PiecewisePoly(method="quintic")
    interp_confs, interp_spds, interp_accs = traj_planner.piecewise_interpolation(mot_data.jv_list,
                                                                                                    ctrl_freq=.1)
    # for conf in interp_confs:
    #     robot.goto_given_conf(conf)
    #     robot.gen_meshmodel().attach_to(base)
    # base.run()
    x = np.arange(len(interp_spds))
    plt.figure(figsize=(10, 5))
    plt.plot(x, interp_spds, '-')
    # plt.xticks(x)
    plt.show()

    base.run()
