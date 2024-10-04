import numpy as np
import math
from wrs import basis as rm, robot_sim as ym, modeling as gm
import wrs.visualization.panda.world as wd
import wrs.motion.optimization_based.incremental_nik as inik
import matplotlib.pyplot as plt
import wrs.motion.trajectory.piecewisepoly_opt as pwpo

if __name__ == "__main__":
    base = wd.World(cam_pos=[3, -1, 1], lookat_pos=[0, 0, 0.5])
    gm.gen_frame(axis_length=.2).attach_to(base)
    yumi_s = ym.Yumi(enable_cc=True)
    inik_svlr = inik.IncrementalNIK(yumi_s)
    component_name = 'rgt_arm'
    circle_center_pos = np.array([.5, -.4, .4])
    circle_ax = rm.rotmat_from_axangle(np.array([0, 1, 0]), -math.pi / 3).dot(np.array([1, 0, 0]))
    radius = .17
    start_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    end_rotmat = rm.rotmat_from_axangle(np.array([0, 0, 1]), -math.pi / 3)
    jnt_values_list, tcp_list = inik_svlr.gen_circular_motion(component_name,
                                                              circle_center_pos,
                                                              circle_ax,
                                                              start_rotmat,
                                                              end_rotmat,
                                                              granularity=.3,
                                                              radius=radius,
                                                              toggle_tcp_list=True)
    print(jnt_values_list)
    # import motion.trajectory.polynomial_wrsold as trajp
    # control_frequency = .005
    control_frequency = .1
    interval_time = 1
    # traj_gen = trajp.TrajPoly(method="quintic")
    # interpolated_confs, interpolated_spds, interpolated_accs = \
    #     traj_gen.piecewise_interpolation(jnt_values_list, control_frequency=control_frequency, time_intervals=interval_time)
    # traj_gen = pwp.PiecewisePoly(method="quintic")
    # interpolated_confs, interpolated_spds, interpolated_accs, interpolated_x = \
    #     traj_gen.interpolate(jnt_values_list, control_frequency=control_frequency, time_intervals=interval_time, toggle_dbg=True)
    # interpolated_confs, interpolated_spds, interpolated_accs, interpolated_x, original_x = \
    #     traj_gen.interpolate_by_max_spdacc(jnt_values_list, control_frequency=control_frequency, max_spds=None,
    #                                        toggle_dbg=True)
    # interpolated_confs, interpolated_spds, interpolated_accs, interpolated_x, original_x = \
    #     traj_gen.trapezoid_interpolate_by_max_spdacc(jnt_values_list, control_frequency=control_frequency, max_spds=None)

    trajopt_gen = pwpo.PWPOpt(method="quintic")
    interpolated_confs, interpolated_spds, interpolated_accs, interpolated_x, original_x = \
        trajopt_gen.interpolate_by_max_spdacc(jnt_values_list, control_frequency=control_frequency, max_spds=None,
                                           toggle_debug=True)
    for i in range(len(tcp_list) - 1):
        spos = tcp_list[i][0]
        srotmat = tcp_list[i][1]
        epos = tcp_list[i + 1][0]
        erotmat = tcp_list[i + 1][1]
        print(spos, epos)
        gm.gen_dashed_arrow(spos, epos, stick_radius=.01, rgba=[1, 0, 0, 1]).attach_to(base)
        gm.gen_myc_frame(epos, erotmat, alpha=.7).attach_to(base)
    yumi_s.fk(component_name, jnt_values_list[1])
    yumi_s.gen_meshmodel(toggle_tcpcs=False, rgba=[.7, .3, .3, .57]).attach_to(base)
    yumi_s.fk(component_name, jnt_values_list[2])
    yumi_s.gen_meshmodel(toggle_tcpcs=False, rgba=[.3, .7, .3, .57]).attach_to(base)
    yumi_s.fk(component_name, jnt_values_list[3])
    yumi_s.gen_meshmodel(toggle_tcpcs=False, rgba=[.3, .3, .7, .57]).attach_to(base)
    yumi_s.fk(component_name, jnt_values_list[0])
    yumi_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    # base.run()
    x = np.arange(len(jnt_values_list))
    print(x)
    plt.figure(figsize=(3, 5))
    plt.plot(x, jnt_values_list, '-o')
    plt.xticks(x)
    plt.show()
    base.run()
