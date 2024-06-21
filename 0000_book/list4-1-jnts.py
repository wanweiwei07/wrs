import numpy as np
import robot_sim._kinematics.jl as rkjl
import robot_sim._kinematics.model_generator as rkmg
import robot_sim._kinematics.jlchain as rkjlc
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as mgm




option = 'c'
if option == 'a':
    jnt = rkjl.Joint()
    jnt.loc_pos = np.array([0, 0, 0])
    jnt.loc_rotmat = np.eye(3)
    jnt.loc_motion_ax = np.array([0, 1, 0])
    jnt.motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    jnt.update_globals(pos=np.zeros(3), rotmat=np.eye(3))
    # visualize
    base = wd.World(cam_pos=np.array([.3, .3, .3]), lookat_pos=np.array([0, 0, 0]))
    rkmg.gen_jnt(jnt, toggle_frame_0=True, toggle_frame_q=True, toggle_actuation=True).attach_to(base)
if option == 'b':
    jnt = rkjl.Joint()
    jnt.loc_pos = np.array([0, 0, 0])
    jnt.loc_rotmat = np.eye(3)
    jnt.loc_motion_ax = np.array([0, 1, 0])
    jnt.motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    jnt.update_globals(pos=np.zeros(3), rotmat=np.eye(3), motion_value=np.pi / 6)
    # visualize
    base = wd.World(cam_pos=np.array([.3, .3, .3]), lookat_pos=np.array([0, 0, 0]))
    rkmg.gen_jnt(jnt, toggle_frame_0=True, toggle_frame_q=True, toggle_actuation=True).attach_to(base)
if option == 'c':
    jnt_0 = rkjl.Joint()
    jnt_0.loc_pos = np.array([0, 0, 0])
    jnt_0.loc_rotmat = np.eye(3)
    jnt_0.loc_motion_ax = np.array([0, 1, 0])
    jnt_0.motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    jnt_0.update_globals(pos=np.zeros(3), rotmat=np.eye(3), motion_value=np.pi / 6)
    jnt_1 = rkjl.Joint()
    jnt_1.loc_pos = np.array([-.15, .15, 0])
    jnt_1.loc_rotmat = np.eye(3)
    jnt_1.loc_motion_ax = np.array([0, 0, 1])
    jnt_1.motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])


    # visualize
    base = wd.World(cam_pos=np.array([.5, .5, .5]), lookat_pos=np.array([0, 0, 0]))
    rkmg.gen_jnt(jnt_0, toggle_frame_0=True, toggle_frame_q=True, toggle_actuation=True).attach_to(base)
    jnt_1.update_globals(pos=jnt_0.gl_pos_q, rotmat=jnt_0.gl_rotmat_q)
    rkmg.gen_jnt(jnt_1, toggle_frame_0=True, toggle_frame_q=True, toggle_actuation=False).attach_to(base)
    mgm.gen_dashed_stick(spos=jnt_0.gl_pos_0, epos=jnt_1.gl_pos_0, rgb=rm.bc.yellow, radius=.0025).attach_to(base)
    jnt_1.update_globals(pos=jnt_0.gl_pos_0, rotmat=jnt_0.gl_rotmat_0)
    rkmg.gen_jnt(jnt_1, toggle_frame_0=True, toggle_frame_q=True, toggle_actuation=False).attach_to(base)
    mgm.gen_stick(spos=jnt_0.gl_pos_0, epos=jnt_1.gl_pos_0, rgb=rm.bc.yellow, radius=.0025).attach_to(base)

base.run()
