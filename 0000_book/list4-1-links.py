import numpy as np
import robot_sim._kinematics.jlchain as rkjlc
import visualization.panda.world as wd

jlc = rkjlc.JLChain(n_dof=6)
# root
jlc.anchor.pos = np.zeros(3)
jlc.anchor.rotmat = np.eye(3)
# joint 0
jlc.jnts[0].loc_pos = np.array([0, 0, .1])
jlc.jnts[0].loc_rotmat = np.eye(3)
jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
# joint 1
jlc.jnts[1].loc_pos = np.array([0, 0, .1])
jlc.jnts[1].loc_rotmat = np.eye(3)
jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
# joint 2
jlc.jnts[2].loc_pos = np.array([0, 0, .1])
jlc.jnts[2].loc_rotmat = np.eye(3)
jlc.jnts[2].loc_motion_ax = np.array([0, 1, 0])
# joint 3
jlc.jnts[3].loc_pos = np.array([0, 0, .1])
jlc.jnts[3].loc_rotmat = np.eye(3)
jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
# joint 4
jlc.jnts[4].loc_pos = np.array([0, 0, .1])
jlc.jnts[4].loc_rotmat = np.eye(3)
jlc.jnts[4].loc_motion_ax = np.array([0, 1, 0])
# joint 5
jlc.jnts[5].loc_pos = np.array([0, 0, .1])
jlc.jnts[5].loc_rotmat = np.eye(3)
jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
jlc.finalize()
# visualize
base = wd.World(cam_pos=np.array([1.2, 1.2, 1.2]), lookat_pos=np.array([0, 0, .3]))
jlc.gen_stickmodel(toggle_jnt_frames=True, toggle_actuation=True).attach_to(base)
base.run()
