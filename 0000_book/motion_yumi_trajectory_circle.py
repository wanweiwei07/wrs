import numpy as np
import math
import basis.robot_math as rm
import visualization.panda.world as wd
import robot_sim.robots.yumi.yumi as ym
import modeling.geometric_model as gm
import motion.optimization_based.incremental_nik as inik
import matplotlib.pyplot as plt

if __name__ == "__main__":
    base = wd.World(cam_pos=[3, -1, 1], lookat_pos=[0, 0, 0.5])
    gm.gen_frame(length=.2).attach_to(base)
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
        gm.gen_dasharrow(spos, epos, thickness=.01, rgba=[1,0,0,1]).attach_to(base)
        gm.gen_mycframe(epos, erotmat, alpha=.7).attach_to(base)
    # robot_s.fk(hnd_name, jnt_values_list[1])
    # robot_s.gen_meshmodel(toggle_tcpcs=False, rgba=[.7,.3,.3,.57]).attach_to(base)
    # robot_s.fk(hnd_name, jnt_values_list[2])
    # robot_s.gen_meshmodel(toggle_tcpcs=False, rgba=[.3,.7,.3,.57]).attach_to(base)
    # robot_s.fk(hnd_name, jnt_values_list[3])
    # robot_s.gen_meshmodel(toggle_tcpcs=False, rgba=[.3,.3,.7,.57]).attach_to(base)
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