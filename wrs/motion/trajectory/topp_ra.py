import wrs.basis.robot_math as rm


def split_path_1d(path_1d):
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


def generate_time_optimal_trajectory(path, max_vels=None, max_accs=None, ctrl_freq=.005):
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


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from wrs import wd, rm, mgm, mip, toppra
    import wrs.robot_sim.robots.nova2_wg.nova2wg3 as dnw
    from wrs.motion.trajectory.quintic import QuinticSpline

    base = wd.World(cam_pos=rm.np.array([3, -1, 1]), lookat_pos=rm.np.array([0, 0, 0.5]))
    mgm.gen_frame().attach_to(base)
    robot = dnw.Nova2WG3()
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
