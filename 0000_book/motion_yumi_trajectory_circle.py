import numpy as np
import math
from wrs import basis as rm, robot_sim as ym, modeling as gm
import wrs.visualization.panda.world as wd
import wrs.motion.optimization_based.incremental_nik as inik
import matplotlib.pyplot as plt

if __name__ == "__main__":
    base = wd.World(cam_pos=[3, -1, 1], lookat_pos=[0, 0, 0.5])
    gm.gen_frame(axis_length=.2).attach_to(base)
    yumi_s = ym.Yumi(enable_cc=True)
    inik_svlr = inik.IncrementalNIK(yumi_s)
    component_name = 'rgt_arm'
    circle_center_pos = np.array([.3, -.4, .4])
    circle_ax = np.array([1, 0, 0])
    radius = .1
    start_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    end_rotmat = start_rotmat
    jnt_values_list, tcp_list = inik_svlr.gen_circular_motion(component_name,
                                                              circle_center_pos,
                                                              circle_ax,
                                                              start_rotmat,
                                                              end_rotmat,
                                                              radius=radius,
                                                              toggle_tcp_list=True)
    for i in range(len(tcp_list)-1):
        spos = tcp_list[i][0]
        srotmat = tcp_list[i][1]
        epos = tcp_list[i+1][0]
        erotmat = tcp_list[i+1][1]
        print(spos, epos)
        gm.gen_dashed_arrow(spos, epos, stick_radius=.01, rgba=[1, 0, 0, 1]).attach_to(base)
        gm.gen_myc_frame(epos, erotmat, alpha=.7).attach_to(base)
    # robot_s.fk(hnd_name, jnt_values_list[1])
    # robot_s.gen_meshmodel(toggle_flange_frame=False, rgba=[.7,.3,.3,.57]).attach_to(base)
    # robot_s.fk(hnd_name, jnt_values_list[2])
    # robot_s.gen_meshmodel(toggle_flange_frame=False, rgba=[.3,.7,.3,.57]).attach_to(base)
    # robot_s.fk(hnd_name, jnt_values_list[3])
    # robot_s.gen_meshmodel(toggle_flange_frame=False, rgba=[.3,.3,.7,.57]).attach_to(base)
    yumi_s.fk(component_name, jnt_values_list[0])
    yumi_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    # base.run()
    x = np.arange(len(jnt_values_list))
    print(x)
    plt.figure(figsize=(10,5))
    plt.plot(x, jnt_values_list, '-o')
    plt.xticks(x)
    plt.show()
    base.run()