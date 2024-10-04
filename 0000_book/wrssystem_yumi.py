import numpy as np
import math, time
from wrs import basis as rm, robot_sim as ym, modeling as gm
import wrs.visualization.panda.world as wd

if __name__ == "__main__":
    base = wd.World(cam_pos=[3, 1, 1], lookat_pos=[0, 0, 0.5])
    gm.gen_frame(axis_length=.2).attach_to(base)
    yumi_s = ym.Yumi(enable_cc=True)
    # ik test
    manipulator_name = 'rgt_arm'
    tgt_pos = np.array([.1, -.5, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    tgt_pos1 = np.array([.4, -.35, .45])
    tgt_rotmat1 = tgt_rotmat
    # tgt_rotmat1 = rm.rotmat_from_axangle([0, 1, 0], -math.pi / 3).dot(tgt_rotmat)
    tgt_pos2 = np.array([.6, -.2, .6])
    tgt_rotmat2 = tgt_rotmat
    # tgt_rotmat2 = rm.rotmat_from_axangle([0, 1, 0], -math.pi/6).dot(tgt_rotmat1)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    tic = time.time()
    # jnt_values = robot_s.ik(hnd_name, tgt_pos, tgt_rotmat, seed_jnt_values=np.array([.0,.0,.0,.0,.0,.0,.0]))
    jnt_values = yumi_s.ik(manipulator_name, tgt_pos, tgt_rotmat)
    # jnt_values1 = robot_s.ik(hnd_name, tgt_pos1, tgt_rotmat1)
    # jnt_values2 = robot_s.ik(hnd_name, tgt_pos2, tgt_rotmat2)
    jnt_values1 = yumi_s.ik(manipulator_name, tgt_pos1, tgt_rotmat1, seed_jnt_values=jnt_values)
    jnt_values2 = yumi_s.ik(manipulator_name, tgt_pos2, tgt_rotmat2, seed_jnt_values=jnt_values1)
    toc = time.time()
    print(toc - tic)
    yumi_s.fk(manipulator_name, jnt_values)
    # yumi_meshmodel = robot_s.gen_meshmodel(toggle_flange_frame=True, rgba=[.3, .3, .3, .3])
    yumi_meshmodel = yumi_s.gen_meshmodel(toggle_tcpcs=True, rgba=[1, .3, .3, .3])
    yumi_meshmodel.attach_to(base)
    yumi_s.fk(manipulator_name, jnt_values1)
    # yumi_meshmodel = robot_s.gen_meshmodel(toggle_flange_frame=True, rgba=[.3, .3, .3, .3])
    yumi_meshmodel = yumi_s.gen_meshmodel(toggle_tcpcs=True, rgba=[.3, 1, .3, .3])
    yumi_meshmodel.attach_to(base)
    yumi_s.fk(manipulator_name, jnt_values2)
    # yumi_meshmodel = robot_s.gen_meshmodel(toggle_flange_frame=True, rgba=[.3, .3, .3, .3])
    yumi_meshmodel = yumi_s.gen_meshmodel(toggle_tcpcs=True)
    yumi_meshmodel.attach_to(base)
    yumi_s.gen_stickmodel().attach_to(base)
    base.run()