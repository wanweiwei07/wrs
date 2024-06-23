import numpy as np
import robot_sim._kinematics.jl as rkjl
import robot_sim._kinematics.model_generator as rkmg
import robot_sim._kinematics.jlchain as rkjlc
import visualization.panda.world as wd

option = 'b'
if option == 'a':
    jlc = rkjlc.JLChain(n_dof=6)
    # root
    jlc.anchor.pos = np.zeros(3)
    jlc.anchor.rotmat = np.eye(3)
    # joint 0
    jlc.jnts[0].loc_pos = np.array([0, 0, .15])
    jlc.jnts[0].loc_rotmat = np.eye(3)
    jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[0].motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    # joint 1
    jlc.jnts[1].loc_pos = np.array([0, 0, .15])
    jlc.jnts[1].loc_rotmat = np.eye(3)
    jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
    jlc.jnts[1].motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    # joint 2
    jlc.jnts[2].loc_pos = np.array([0, 0, .15])
    jlc.jnts[2].loc_rotmat = np.eye(3)
    jlc.jnts[2].loc_motion_ax = np.array([0, 1, 0])
    jlc.jnts[2].motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    # joint 3
    jlc.jnts[3].loc_pos = np.array([0, 0, .15])
    jlc.jnts[3].loc_rotmat = np.eye(3)
    jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[3].motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    # joint 4
    jlc.jnts[4].loc_pos = np.array([0, 0, .15])
    jlc.jnts[4].loc_rotmat = np.eye(3)
    jlc.jnts[4].loc_motion_ax = np.array([0, 1, 0])
    jlc.jnts[4].motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    # joint 5
    jlc.jnts[5].loc_pos = np.array([0, 0, .15])
    jlc.jnts[5].loc_rotmat = np.eye(3)
    jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[5].motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    jlc.finalize()
    # visualize
    base = wd.World(cam_pos=np.array([1.7, 1.7, 1.7]), lookat_pos=np.array([0, 0, .5]))
    jlc.gen_stickmodel(toggle_flange_frame=True, toggle_jnt_frames=True, toggle_actuation=True, alpha=.7).attach_to(base)
if option == 'b':
    jlc = rkjlc.JLChain(n_dof=6)
    # root
    jlc.anchor.pos = np.zeros(3)
    jlc.anchor.rotmat = np.eye(3)
    # joint 0
    jlc.jnts[0].loc_pos = np.array([0, 0, .15])
    jlc.jnts[0].loc_rotmat = np.eye(3)
    jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[0].motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    # joint 1
    jlc.jnts[1].loc_pos = np.array([0, 0, .15])
    jlc.jnts[1].loc_rotmat = np.eye(3)
    jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
    jlc.jnts[1].motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    # joint 2
    jlc.jnts[2].loc_pos = np.array([0, 0, .15])
    jlc.jnts[2].loc_rotmat = np.eye(3)
    jlc.jnts[2].loc_motion_ax = np.array([0, 1, 0])
    jlc.jnts[2].motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    # joint 3
    jlc.jnts[3].loc_pos = np.array([0, 0, .15])
    jlc.jnts[3].loc_rotmat = np.eye(3)
    jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[3].motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    # joint 4
    jlc.jnts[4].loc_pos = np.array([0, 0, .15])
    jlc.jnts[4].loc_rotmat = np.eye(3)
    jlc.jnts[4].loc_motion_ax = np.array([0, 1, 0])
    jlc.jnts[4].motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    # joint 5
    jlc.jnts[5].loc_pos = np.array([0, 0, .15])
    jlc.jnts[5].loc_rotmat = np.eye(3)
    jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[5].motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    jlc.finalize()
    jlc.goto_given_conf(jnt_values=np.array([np.pi/12, -np.pi / 3, np.pi * 2 / 3, np.pi/12, np.pi / 3, 0]))
    # visualize
    base = wd.World(cam_pos=np.array([1.7, 1.7, 1.7]), lookat_pos=np.array([0, 0, .5]))
    jlc.gen_stickmodel(toggle_jnt_frames=True, toggle_actuation=True, alpha=.7).attach_to(base)
if option == 'c':
    jnt = rkjl.Joint()
    jnt.loc_pos = np.array([0, 0, 0])
    jnt.loc_rotmat = np.eye(3)
    jnt.loc_motion_ax = np.array([0, 1, 0])
    jnt.motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    jnt.update_globals(pos=np.zeros(3), rotmat=np.eye(3))
    # visualize
    base = wd.World(cam_pos=np.array([.3, .3, .3]), lookat_pos=np.array([0, 0, 0]))
    rkmg.gen_jnt(jnt, toggle_frame_0=True, toggle_frame_q=True, toggle_actuation=True).attach_to(base)
if option == 'd':
    jnt = rkjl.Joint()
    jnt.loc_pos = np.array([0, 0, 0])
    jnt.loc_rotmat = np.eye(3)
    jnt.loc_motion_ax = np.array([0, 1, 0])
    jnt.motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    jnt.update_globals(pos=np.zeros(3), rotmat=np.eye(3), motion_value=np.pi / 6)
    # visualize
    base = wd.World(cam_pos=np.array([.3, .3, .3]), lookat_pos=np.array([0, 0, 0]))
    rkmg.gen_jnt(jnt, toggle_frame_0=True, toggle_frame_q=True, toggle_actuation=True).attach_to(base)
base.run()
