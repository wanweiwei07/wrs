import math
import numpy as np


class TrajTrap(object):

    def __init__(self):
        self.fit = self._fit_max_acc
        self.predict = self._predict_max_acc

    def _fit_max_acc(self, conf0, spd0, conf1, spd1):
        print(conf0, conf1)
        # assume max speed and check if it is needed in the given time interval
        self.avg_spd = self._max_spd * (np.sign((conf0 + conf1)) + ((conf0 + conf1) == 0))
        self.acc_begin = np.sign(self.avg_spd - spd0) * self._max_acc
        self.acc_end = np.sign(spd1 - self.avg_spd) * self._max_acc
        self.t_begin = abs(self.avg_spd - spd0) / self._max_acc
        self.t_end = abs(self.avg_spd - spd1) / self._max_acc
        begin_movement = spd0 * self.t_begin + (self.acc_begin * self.t_begin ** 2) / 2
        end_movement = self.avg_spd * self.t_end + (self.acc_end * self.t_end ** 2) / 2
        self.t_middle = (conf1 - conf0 - begin_movement - end_movement) / self.avg_spd
        # print(self.t_middle, self.avg_spd, self._interval_time, self.t_begin, self.t_end)
        # # for those do not need the max speed, only consider acceleration
        # slctn = self.t_begin + self.t_end >= self._interval_time
        # self.avg_spd[slctn] = ((-self.acc_begin * self.acc_end * self._interval_time -
        #                         spd0 * self.acc_end + spd1 * self.acc_begin) /
        #                        (self.acc_begin - self.acc_end))[slctn]
        # self.t_begin[slctn] = (abs(self.avg_spd - spd0) / self._max_acc)[slctn]
        # self.t_end[slctn] = (abs(self.avg_spd - spd1) / self._max_acc)[slctn]
        # self.t_middle[slctn] = 0
        # print(self._interval_time, self.t_begin, self.t_end)
        print(self.t_middle, (self._interval_time - self.t_begin - self.t_end), self.t_begin, self.t_end)
        if np.any(np.logical_and(self.t_middle > self._interval_time - self.t_begin - self.t_end, self.t_middle > 0)):
            # for those need that max speed, check if the max speed is fast enough to finish the given motion
            raise ValueError("The required time interval is too short!")
        # also check if a lower max speed works
        # print(abs(self.t_middle - (self._interval_time - self.t_begin - self.t_end)), self.t_middle)
        init_avg_spd = self.avg_spd
        cnter = np.ones_like(self.avg_spd)
        cnter_last = np.zeros_like(self.avg_spd, dtype=bool)
        while True:
            slctn = np.logical_or(abs(self.t_middle - (self._interval_time - self.t_begin - self.t_end)) > .001,
                                  self.t_middle < 1e-6)
            if np.any(np.logical_and(abs(self._interval_time - self.t_begin - self.t_end)<1e-6, self.t_middle>0)):
                raise ValueError("The required time interval is too short!")
            # print(slctn)
            if np.any(slctn):
                # print(self.t_middle)
                sign = -np.ones_like(self.avg_spd)
                loc_slctn = np.logical_and(self.t_middle > 1e-6, self._interval_time - self.t_begin - self.t_end > 0)
                loc_slctn = np.logical_and(loc_slctn, self.t_middle > self._interval_time - self.t_begin - self.t_end)
                print(loc_slctn)
                if np.any(loc_slctn):
                    sign[loc_slctn] = 1
                    cnter[np.logical_and(loc_slctn, not cnter_last[loc_slctn])] += 1
                    cnter_last[loc_slctn] = True
                not_loc_slctn = np.logical_not(loc_slctn)
                cnter[np.logical_and(not_loc_slctn, cnter_last[not_loc_slctn])] += 1
                cnter_last[not_loc_slctn] = False
                self.avg_spd[slctn] += (sign * init_avg_spd / np.exp2(cnter))[slctn]
                # if any(abs(self.t_middle - (self._interval_time - self.t_begin - self.t_end)) < .001):
                #     print(self.t_middle, self.t_begin, self.t_end, self.avg_spd)
                self.acc_begin[slctn] = (np.sign(self.avg_spd - spd0) * self._max_acc)[slctn]
                self.acc_end[slctn] = (np.sign(spd1 - self.avg_spd) * self._max_acc)[slctn]
                self.t_begin[slctn] = (abs(self.avg_spd - spd0) / self._max_acc)[slctn]
                self.t_end[slctn] = (abs(self.avg_spd - spd1) / self._max_acc)[slctn]
                begin_movement = spd0 * self.t_begin + (self.acc_begin * self.t_begin ** 2) / 2
                end_movement = self.avg_spd * self.t_end + (self.acc_end * self.t_end ** 2) / 2
                self.t_middle = (conf1 - conf0 - begin_movement - end_movement) / self.avg_spd
                print(self.t_middle, self.t_begin, self.t_end, self.avg_spd)
                print(self.t_middle, (self._interval_time - self.t_begin - self.t_end))
                # print("xxxx")
                # print(self.acc_begin)
                # print(self.acc_end)
                # print(self.t_begin)
                # print(self.t_end)
                # print(self.avg_spd)
                # print(self._interval_time)
                # print(slctn)
            else:
                break
        self.conf0 = conf0
        self.conf1 = conf1
        self.spd0 = spd0
        self.spd1 = spd1

    def _predict_max_acc(self, step):
        local_interpolated_confs = np.zeros_like(step)
        local_interplated_spds = np.zeros_like(step)
        local_accs = np.zeros_like(step)
        slctn = step <= self.t_begin
        local_interpolated_confs[slctn] = (self.conf0 + self.spd0 * step + (self.acc_begin * step ** 2) / 2)[slctn]
        local_interplated_spds[slctn] = (self.spd0 + self.acc_begin * step)[slctn]
        local_accs[slctn] = self.acc_begin
        slctn = np.logical_and(self.t_begin < step, step <= self.t_begin + self.t_middle)
        t_left = step - self.t_begin
        local_interpolated_confs[slctn] = (self.conf0 + self.spd0 * self.t_begin + (
                self.acc_begin * self.t_begin ** 2) / 2 + self.avg_spd * t_left)[slctn]
        local_interplated_spds[slctn] = (self.avg_spd + local_interplated_spds)[slctn]
        local_accs[slctn] = 0
        slctn = self.t_begin + self.t_middle < step
        t_left = step - self.t_begin - self.t_middle
        local_interpolated_confs[slctn] = (self.conf0 + self.spd0 * self.t_begin + (
                self.acc_begin * self.t_begin ** 2) / 2 + self.avg_spd * self.t_middle + self.avg_spd * t_left + (
                                                   self.acc_end * t_left ** 2) / 2)[slctn]
        local_interplated_spds[slctn] = (self.avg_spd + self.acc_end * t_left)[slctn]
        local_accs[slctn] = self.acc_end
        return local_interpolated_confs, local_interplated_spds, local_accs

    def piecewise_interpolation(self, path, control_frequency=.005, interval_time=2.0, max_acc=math.pi / 6,
                                max_spd=math.pi * 2):
        """
        :param path: a 1d array of configurations
        :param control_frequency: the program will sample time_intervals/control_frequency confs
        :param max_acc, max_spds
        :return:
        author: weiwei
        date: 20200328
        """
        self._max_acc = max_acc
        self._max_spd = max_spd
        self._interval_time = interval_time
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
                pre_avg_spd = (jntconf - pre_conf) / self._interval_time
                nxt_avg_spd = (nxt_conf - jntconf) / self._interval_time
                # set to 0 if signs are different -> reduces overshoot
                zero_id = np.where((np.sign(pre_avg_spd) + np.sign(nxt_avg_spd)) == 0)
                pass_spd = (pre_avg_spd + nxt_avg_spd) / 2.0
                pass_spd[zero_id] = 0.0
                passing_spd_list.append(pass_spd)
        passing_conf_list.append(path[-1])
        passing_spd_list.append(np.zeros_like(path[-1]))
        interpolated_confs = []
        interpolated_spds = []
        interpolated_accs = []
        for id, passing_conf in enumerate(passing_conf_list):
            if id == 0:
                continue
            pre_passing_conf = passing_conf_list[id - 1]
            pre_passing_spd = passing_spd_list[id - 1]
            passing_spd = passing_spd_list[id]
            self.fit(pre_passing_conf, pre_passing_spd, passing_conf, passing_spd)
            samples = np.linspace(0, self._interval_time, math.floor(interval_time / control_frequency))
            local_interpolated_confs, local_interplated_spds, local_accs = self.predict(samples)
            interpolated_confs += local_interpolated_confs.tolist()
            interpolated_spds += local_interplated_spds.tolist()
            interpolated_accs += local_accs.tolist()
        return interpolated_confs, interpolated_spds, interpolated_accs


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # y = [[0],[3]]
    # control_frequency = .005１
    # time_intervals = 15.0
    # y = [[0],[math.pi*3]]
    y = [[math.pi / 6], [math.pi/2]]
    control_frequency = .005
    interval_time = 3
    traj = TrajTrap()
    interpolated_confs, interpolated_spds, local_accs = traj.piecewise_interpolation(y,
                                                                                     control_frequency=control_frequency,
                                                                                     interval_time=interval_time)
    # print(interpolated_spds)
    # interpolated_spds=np.array(interpolated_spds)
    # print(interpolated_confs)
    x = np.linspace(0, interval_time * (len(y) - 1), (len(y) - 1) * math.floor(interval_time / control_frequency))
    fig, axs = plt.subplots(3, figsize=(3.5,4.75))
    fig.tight_layout(pad=.7)
    axs[0].plot(x, interpolated_confs)
    axs[0].plot(range(0, interval_time * (len(y)), interval_time), y, '--o',color='tab:blue')
    axs[1].plot(x, interpolated_spds)
    axs[2].plot(x, local_accs)
    # plt.quiver(x, interpolated_confs, x, interpolated_spds, width=.001)
    # plt.plot(y)
    plt.show()
