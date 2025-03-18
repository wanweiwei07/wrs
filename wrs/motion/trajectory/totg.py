# time optimal trrajectory generation
# jerk is not considered

import wrs.basis.robot_math as rm


def split_path_1d(path_1d):
    """
    find zero crossing way points
    :param path_1d:
    :return:
    """
    zero_crossings = []
    distances = rm.np.diff(path_1d)
    for i in range(len(distances) - 1):
        if rm.np.abs(distances[i]) < 1e-12:
            for j in range(i + 1, len(distances)):
                if rm.np.abs(distances[j]) > 1e-12 and rm.np.sign(distances[j]) != rm.np.sign(distances[i - 1]):
                    zero_crossings.append(j + 1)
                    break
                if rm.np.abs(distances[j]) > 1e-12 and rm.np.sign(distances[j]) == rm.np.sign(distances[i - 1]):
                    break
            if j == len(distances) - 1:
                break
            else:
                print(i)
                i = j + 1
        elif rm.np.sign(distances[i]) != rm.np.sign(distances[i + 1]) and rm.np.abs(distances[i + 1]) > 1e-12:
            zero_crossings.append(i + 2)
    if len(zero_crossings) == 0:
        return [path_1d]
    else:
        segments = rm.np.split(path_1d, zero_crossings)
        return segments


def forward_1d(path_1d, max_vel, max_acc):
    n_waypoints = len(path_1d)
    velocities = rm.np.zeros(n_waypoints)
    distances = rm.np.abs(rm.np.diff(path_1d))
    for i in range(1, n_waypoints):
        velocities[i] = rm.np.minimum(max_vel, rm.np.sqrt(velocities[i - 1] ** 2 + 2 * max_acc * distances[i - 1]))
    return velocities


def backward_1d(path_1d, velocities, max_acc):
    n_waypoints = len(path_1d)
    distances = rm.np.abs(rm.np.diff(path_1d))
    velocities[-1] = 0
    for i in range(n_waypoints - 2, -1, -1):
        velocities[i] = rm.np.minimum(velocities[i], rm.np.sqrt(velocities[i + 1] ** 2 + 2 * max_acc * distances[i]))
    return velocities


def proc_segs(path_1d_segs, max_vel, max_acc):
    velocities = []
    for id, path_1d in enumerate(path_1d_segs):
        if id > 0:
            path_1d = rm.np.insert(path_1d, 0, path_1d_segs[id - 1][-1])
        v_fwd = forward_1d(path_1d, max_vel, max_acc)
        v_bwd = backward_1d(path_1d, v_fwd, max_acc)
        if rm.np.diff(path_1d)[0] < 0:
            v_bwd = v_bwd * -1
        velocities.append(v_bwd)
    return velocities


def time_optimal_trajectory_generation(path, max_vels=None, max_accs=None, ctrl_freq=.005):
    path = rm.np.asarray(path)
    n_waypoints, n_jnts = path.shape
    if max_vels is None:
        max_vels = rm.np.asarray([rm.pi * 2 / 3] * n_jnts)
    if max_accs is None:
        max_accs = rm.np.asarray([rm.pi] * n_jnts)
    velocities = rm.np.zeros((n_waypoints, n_jnts))
    for id_jnt in range(n_jnts):
        path_1d = path[:, id_jnt]
        path_1d_segs = split_path_1d(path_1d)
        vel_1d_segs = proc_segs(path_1d_segs, max_vels[id_jnt], max_accs[id_jnt])
        vel_1d_merged = rm.np.concatenate([v[:-1] for v in vel_1d_segs[:-1]] + [vel_1d_segs[-1]])
        velocities[:, id_jnt] = vel_1d_merged
    distances = rm.np.abs(rm.np.diff(path, axis=0))
    avg_velocities = rm.np.abs((velocities[:-1] + velocities[1:]) / 2)
    avg_velocities = rm.np.where(avg_velocities == 0, 10e6, avg_velocities)  # use a large value to ignore 0 speeds
    time_intervals = rm.np.max(distances / avg_velocities, axis=1)
    time = rm.np.zeros(len(path))
    time[1:] = rm.np.cumsum(time_intervals)
    n_interp_conf = int(time[-1] / ctrl_freq) + 1
    interp_time = rm.np.linspace(0, time[-1], n_interp_conf)
    # interp_confs= rm.np.zeros((len(interp_time), path.shape[1]))
    # interp_spds = rm.np.zeros((len(interp_time), path.shape[1]))
    # interp_accs = rm.np.zeros((len(interp_time), path.shape[1]))
    # for j in range(path.shape[1]):
    #     cs_confs = CubicSpline(time, path[:, j], bc_type=((1, 0), (1, 0)))
    #     interp_confs[:, j] = cs_confs(interp_time)
    #     cs_velocities = cs_confs.derivative()
    #     interp_spds[:, j] = cs_velocities(interp_time)
    #     cs_accs = cs_velocities.derivative()
    #     interp_accs[:, j] = cs_accs(interp_time)
    interp_confs = rm.np.zeros((len(interp_time), path.shape[1]))
    interp_spds = rm.np.zeros((len(interp_time), path.shape[1]))
    for j in range(path.shape[1]):
        interp_confs[:, j] = rm.np.interp(interp_time, time, path[:, j])
        interp_spds[:, j] = rm.np.interp(interp_time, time, velocities[:, j])
    tmp_spds = rm.np.append(interp_spds, rm.np.zeros((1, n_jnts)), axis=0)
    interp_accs = rm.np.diff(tmp_spds, axis=0) / ctrl_freq
    return interp_time, interp_confs, interp_spds, interp_accs


# ===================================================================================================
# jerk is considered
# ===================================================================================================

def forward_1d_jerk(path_1d, max_vel, max_acc, max_jerk):
    # hyper parameters for numerical control
    dt_min = 0
    dt_max = 1.0
    n_waypoints = len(path_1d)
    velocities = np.zeros(n_waypoints)
    accelerations = np.zeros(n_waypoints)
    distances = np.abs(np.diff(path_1d))
    # initialize the first waypoint
    velocities[0] = 0.0
    accelerations[0] = 0.0
    for i in range(1, n_waypoints):
        dist = distances[i - 1]
        if velocities[i - 1] < 1e-4:
            dt_est = (6.0 * dist / max_jerk) ** (1 / 3)  # assume does not reach max_acc in a single step
        else:
            dt_est = dist / velocities[i - 1]
        dt = max(dt_min, min(dt_est, dt_max))
        # update acceleration
        a_candidate = accelerations[i - 1] + max_jerk * dt
        if a_candidate > max_acc:
            a_candidate = max_acc
        accelerations[i] = a_candidate
        # update speed
        a_avg = 0.5 * (accelerations[i - 1] + accelerations[i])
        v_candidate = np.sqrt(velocities[i - 1] ** 2 + 2.0 * a_avg * dist)
        velocities[i] = min(v_candidate, max_vel)
    return velocities, accelerations


def backward_1d_jerk(path_1d, velocities, accelerations, max_vel, max_acc, max_jerk):
    # hyper parameters for numerical control
    dt_min = 0
    dt_max = 1.0
    n_waypoints = len(path_1d)
    distances = np.abs(np.diff(path_1d))
    # initialize the last waypoint
    velocities[-1] = 0.0
    accelerations[-1] = 0.0
    for i in range(n_waypoints - 2, -1, -1):
        dist = distances[i]
        if velocities[i + 1] < 1e-4:
            dt_est = (6.0 * dist / max_jerk) ** (1 / 3)  # assume does not reach max_acc in a single step
        else:
            dt_est = dist / velocities[i + 1]
        dt = max(dt_min, min(dt_est, dt_max))
        # update acceleration
        a_candidate = max(0, accelerations[i + 1] - max_jerk * dt)
        if a_candidate < -max_acc:
            a_candidate = -max_acc
        accelerations[i] = a_candidate
        # update speed
        a_avg = 0.5 * (accelerations[i] + accelerations[i + 1])
        v_candidate = np.sqrt(velocities[i + 1] ** 2 + 2.0 * a_avg * dist)
        velocities[i] = min(v_candidate, velocities[i])
    print("vel ", velocities)
    print("acc ", accelerations)
    return velocities, accelerations

# def proc_segs_jerk(path_1d_segs, max_vel, max_acc):
#     velocities = []
#     for id, path_1d in enumerate(path_1d_segs):
#         if id > 0:
#             path_1d = rm.np.insert(path_1d, 0, path_1d_segs[id - 1][-1])
#         v_fwd = forward_1d(path_1d, max_vel, max_acc)
#         v_bwd = backward_1d(path_1d, v_fwd, max_acc)
#         if rm.np.diff(path_1d)[0] < 0:
#             v_bwd = v_bwd * -1
#         velocities.append(v_bwd)
#     return velocities

def proc_segs_jerk(path_1d_segs, max_vel, max_acc, max_jerk):
    velocities_segs = []
    for i, path_1d in enumerate(path_1d_segs):
        if i > 0:
            path_1d = np.insert(path_1d, 0, path_1d_segs[i - 1][-1])
        v_fwd, a_fwd = forward_1d_jerk(path_1d, max_vel, max_acc, max_jerk)
        v_bwd, a_bwd = backward_1d_jerk(path_1d, v_fwd, a_fwd,
                                        max_vel, max_acc, max_jerk)
        if np.diff(path_1d)[0] < 0:
            v_bwd = -v_bwd
        velocities_segs.append(v_bwd)
    print(velocities_segs)
    return velocities_segs


def totg_with_jerk(path, max_vels=None, max_accs=None, max_jerks=None, ctrl_freq=.005):
    path = rm.np.asarray(path)
    n_waypoints, n_jnts = path.shape
    # 1) 默认上限值设定
    if max_vels is None:
        max_vels = rm.np.asarray([rm.pi * 2 / 3] * n_jnts)
    if max_accs is None:
        max_accs = rm.np.asarray([rm.pi] * n_jnts)
    if max_jerks is None:
        max_jerks = np.asarray([rm.pi * 2] * n_jnts)
    # 2) 分段后做 jerk-limited 的速度规划
    velocities = np.zeros((n_waypoints, n_jnts))
    for j in range(n_jnts):
        path_1d = path[:, j]
        path_1d_segs = split_path_1d(path_1d)
        vel_1d_segs = proc_segs_jerk(path_1d_segs, max_vels[j], max_accs[j], max_jerks[j])
        if len(vel_1d_segs) > 1:
            vel_1d_merged = np.concatenate([seg[:-1] for seg in vel_1d_segs[:-1]] + [vel_1d_segs[-1]])
        else:
            vel_1d_merged = vel_1d_segs[0]
        velocities[:, j] = vel_1d_merged
    # 3) 基于平均速度计算相邻路点时间
    distances = np.abs(np.diff(path, axis=0))
    avg_velocities = np.abs((velocities[:-1] + velocities[1:]) / 2)
    avg_velocities = np.where(avg_velocities == 0, 1.0e6, avg_velocities)
    time_intervals = np.max(distances / avg_velocities, axis=1)
    # 4) 累加得到每个路点时间戳
    time = np.zeros(n_waypoints)
    time[1:] = np.cumsum(time_intervals)
    # 5) 生成插值时间轴
    total_time = time[-1]
    n_interp_conf = int(total_time / ctrl_freq) + 1
    interp_time = np.linspace(0, total_time, n_interp_conf)
    # 6) 在时间轴上对位置、速度插值
    interp_confs = np.zeros((n_interp_conf, n_jnts))
    interp_spds = np.zeros((n_interp_conf, n_jnts))
    for j in range(n_jnts):
        interp_confs[:, j] = np.interp(interp_time, time, path[:, j])
        interp_spds[:, j] = np.interp(interp_time, time, velocities[:, j])
    # 7) 用速度差分 -> 加速度
    tmp_spds = np.vstack([interp_spds, np.zeros((1, n_jnts))])
    interp_accs = np.diff(tmp_spds, axis=0) / ctrl_freq
    # 8) 再用加速度差分 -> jerk
    tmp_accs = np.vstack([interp_accs, np.zeros((1, n_jnts))])
    interp_jerks = np.diff(tmp_accs, axis=0) / ctrl_freq
    return interp_time, interp_confs, interp_spds, interp_accs, interp_jerks


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from wrs import wd, rm, mgm, mip
    import wrs.robot_sim.robots.nova2_wg.nova2wg3 as dnw
    from wrs.motion.trajectory.quintic import QuinticSpline

    base = wd.World(cam_pos=rm.np.array([3, -1, 1]), lookat_pos=rm.np.array([0, 0, 0.5]))
    mgm.gen_frame().attach_to(base)
    robot = dnw.Nova2WG3()

    interp_confs = np.array([[-0.363, 0.415, -0.155, -3.209, -1.148, -4.327],
                             [-0.335, 0.5, -0.28, -3.142, -1.351, -4.378],
                             [-0.335, 0.504, -0.265, -3.142, -1.332, -4.378],
                             [-0.335, 0.509, -0.251, -3.142, -1.313, -4.378],
                             [-0.335, 0.514, -0.236, -3.142, -1.294, -4.378],
                             [-0.335, 0.519, -0.223, -3.142, -1.275, -4.378]])
    interp_time, interp_confs1, interp_spds, interp_accs = time_optimal_trajectory_generation(interp_confs,
                                                                                              ctrl_freq=.002)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15))
    ax1.axis('off')
    ax1.set_title("WRS TOTG plots")  # 或者你也可以注释掉这行
    ax1.plot(interp_confs, '-o')
    # position
    ax2.plot(interp_time, interp_confs1, '-o')
    ax2.set_title("Position vs Time")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Position [rad]")
    # velocity
    ax3.plot(interp_time, interp_spds, '-o')
    ax3.set_title("Velocity vs Time")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Velocity [rad/s]")
    # acceleration
    ax4.plot(interp_time, interp_accs, '-o')
    ax4.set_title("Acceleration vs Time")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Acceleration [rad/s^2]")
    plt.tight_layout()  # 自动调整布局避免标题、标签重叠
    plt.savefig(fname="test.jpg")
    plt.show()

    #
    # interp_time, interp_confs1, interp_spds, interp_accs, interp_jerks = totg_with_jerk(interp_confs, ctrl_freq=.002)
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 18))
    # # ax1: 仅用来放置大标题或空白等
    # ax1.axis('off')
    # ax1.set_title("WRS TOTG plots with Jerk")
    # ax1.plot(interp_confs, '-o')
    # # ax2: 位置
    # ax2.plot(interp_time, interp_confs1, '-o')
    # ax2.set_title("Position vs Time")
    # ax2.set_xlabel("Time [s]")
    # ax2.set_ylabel("Position [rad]")
    # # ax3: 速度
    # ax3.plot(interp_time, interp_spds, '-o')
    # ax3.set_title("Velocity vs Time")
    # ax3.set_xlabel("Time [s]")
    # ax3.set_ylabel("Velocity [rad/s]")
    # # ax4: 加速度
    # ax4.plot(interp_time, interp_accs, '-o')
    # ax4.set_title("Acceleration vs Time")
    # ax4.set_xlabel("Time [s]")
    # ax4.set_ylabel("Acceleration [rad/s^2]")
    # # ax5: 加加速度（jerk）
    # ax5.plot(interp_time, interp_jerks, '-o')
    # ax5.set_title("Jerk vs Time")
    # ax5.set_xlabel("Time [s]")
    # ax5.set_ylabel("Jerk [rad/s^3]")
    # # 自动调整子图布局
    # plt.tight_layout()
    # # 保存并显示
    # plt.savefig("test_with_jerk.jpg")
    # plt.show()
    # base.run()

    interplated_planner = mip.InterplatedMotion(robot)
    mot_data = interplated_planner.gen_circular_motion(circle_center_pos=rm.np.array([.6, 0, .4]),
                                                       circle_normal_ax=rm.np.array([1, 0, 0]),
                                                       start_tcp_rotmat=rm.rotmat_from_axangle(rm.np.array([0, 1, 0]),
                                                                                               rm.pi / 2),
                                                       end_tcp_rotmat=rm.rotmat_from_axangle(rm.np.array([0, 1, 0]),
                                                                                             rm.pi / 2),
                                                       radius=0.1)
    x = rm.np.arange(len(mot_data.jv_list))
    plt.figure(figsize=(10, 5))
    plt.plot(x, mot_data.jv_list, '-o')
    n_jnts = len(mot_data.jv_list[0])

    jv_array = rm.np.asarray(mot_data.jv_list)
    # coeffs_list = []
    # for j in range(n_jnts):
    #     coeffs = rm.np.polyfit(x, jv_array[:, j], 5)
    #     coeffs_list.append(coeffs)
    # interp_confs = rm.np.zeros((len(interp_x), n_jnts))
    # for j in range(n_jnts):
    #     poly = rm.np.poly1d(coeffs_list[j])
    #     interp_confs[:, j] = poly(interp_x)

    # traj_planner = pwp.PiecewisePoly(method="quintic")
    # interp_confs, interp_spds, interp_accs = traj_planner.piecewise_interpolation(mot_data.jv_list, ctrl_freq=0.1, time_interval=0.1)

    interp_x = rm.np.linspace(0, len(jv_array) - 1, 100)
    cs = QuinticSpline(range(len(mot_data.jv_list)), mot_data.jv_list)
    interp_confs = cs(interp_x)

    # import motion.trajectory.quintic as quintic
    # qs = quintic.quintic_spline(range(len(mot_data.jv_list)), mot_data.jv_list)
    # interp_confs = qs(interp_x)

    # import motion.trajectory.piecewisepoly as pwp
    # pwp_planner = pwp.PiecewisePoly(method="quintic")
    # interp_confs, interp_spds, interp_accs = pwp_planner.piecewise_interpolation(mot_data.jv_list)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15))
    ax1.plot(range(len(interp_confs)), interp_confs, '-o')
    interp_time, interp_confs1, interp_spds, interp_accs = generate_time_optimal_trajectory(interp_confs,
                                                                                            ctrl_freq=.05)
    ax2.plot(interp_time, interp_confs1, '-o')
    ax3.plot(interp_time, interp_spds, '-o')
    ax4.plot(interp_time, interp_accs, '-o')
    plt.show()

    _, interp_confs2, _, _ = toppra.generate_time_optimal_trajectory(mot_data.jv_list, ctrl_freq=.05)


    # # plt.figure(figsize=(10, 5))
    # # plt.plot(range(len(interp_confs)), interp_confs, '-o')
    # plt.show()

    class Data(object):
        def __init__(self):
            self.counter = 0
            self.path1 = None
            self.path2 = None
            self.on_screen = []


    anime_data = Data()
    anime_data.path1 = interp_confs1
    anime_data.path2 = interp_confs2


    def update(robot, anime_data, task):
        if anime_data.counter >= len(anime_data.path1):
            for model in anime_data.on_screen:
                model.detach()
            anime_data.counter = 0
            anime_data.on_screen = []
        conf = anime_data.path1[anime_data.counter]
        robot.fix_to(pos=rm.np.array([0, -.3, 0]))
        robot.goto_given_conf(conf)
        model = robot.gen_meshmodel()
        model.attach_to(base)
        anime_data.on_screen.append(model)
        if anime_data.counter < len(anime_data.path2):
            conf = anime_data.path2[anime_data.counter]
            robot.fix_to(pos=rm.np.array([0, .3, 0]))
            robot.goto_given_conf(conf)
            model = robot.gen_meshmodel()
            model.attach_to(base)
            anime_data.on_screen.append(model)
        anime_data.counter += 1
        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[robot, anime_data],
                          appendTask=True)

    base.run()
