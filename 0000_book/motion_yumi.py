import numpy as np
import math
from wrs import basis as rm, robot_sim as ym, modeling as gm
import wrs.visualization.panda.world as wd
import wrs.motion.optimization_based.incremental_nik as inik

if __name__ == "__main__":
    base = wd.World(cam_pos=[3, -1, 1], lookat_pos=[0, 0, 0.5])
    gm.gen_frame(axis_length=.2).attach_to(base)
    yumi_s = ym.Yumi(enable_cc=True)
    inik_svlr = inik.IncrementalNIK(yumi_s)
    component_name = 'rgt_arm'
    # start_pos = np.array([.1, -.5, .3])
    # start_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    # end_pos = np.array([.4, -.35, .45])
    # end_rotmat = start_rotmat
    # jnt_values_list = inik_svlr.gen_linear_motion(hnd_name, start_pos, start_rotmat, end_pos, end_rotmat)
    circle_center_pos = np.array([.3, -.4, .4])
    circle_ax = np.array([1,0,0])
    radius = .1
    start_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    end_rotmat = start_rotmat
    jnt_values_list = inik_svlr.gen_circular_motion(component_name,
                                                    circle_center_pos,
                                                    circle_ax,
                                                    start_rotmat,
                                                    end_rotmat,
                                                    radius=radius)
    for jnt_values in jnt_values_list:
        yumi_s.fk(component_name, jnt_values)
        yumi_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    base.run()