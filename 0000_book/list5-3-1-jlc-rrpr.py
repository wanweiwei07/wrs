import robot_sim._kinematics.jlchain as rkjlc
import robot_sim._kinematics.jl as rkjl
import visualization.panda.world as wd
import modeling.collision_model as mcm
import numpy as np


if __name__ == '__main__':
    base = wd.World(cam_pos=[1.5, .2, .9], lookat_pos=[0, 0, 0.3])
    mcm.mgm.gen_frame().attach_to(base)

    jlc = rkjlc.JLChain(n_dof=4)
    jlc.anchor.lnk_list[0].cmodel = mcm.gen_box(np.array([.05, .05, .1]))
    jlc.anchor.lnk_list[0].loc_pos = np.array([0, 0, 0.05])
    jlc.jnts[0].loc_pos = np.array([0, 0, .1])
    jlc.jnts[0].loc_motion_ax = np.array([1, 0, 0])
    jlc.jnts[0].lnk.cmodel= mcm.gen_box(np.array([.05, .05, .1]))
    jlc.jnts[0].lnk.loc_pos = np.array([0, 0, 0.05])
    jlc.jnts[1].loc_pos = np.array([0, 0, .1])
    jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
    jlc.jnts[1].lnk.cmodel= mcm.gen_box(np.array([.05, .05, .1]))
    jlc.jnts[1].lnk.loc_pos = np.array([0, .0, 0.05])
    jlc.jnts[2].change_type(type=rkjl.rkc.JntType.PRISMATIC, motion_range=np.array([-.1, .1]))
    jlc.jnts[2].loc_pos = np.array([0, 0, .1])
    jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[3].loc_pos = np.array([0, 0, .1])
    jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[3].lnk.cmodel= mcm.gen_box(np.array([.05, .05, .1]))
    jlc.jnts[3].lnk.loc_pos = np.array([0, 0, 0.05])
    jlc._loc_flange_pos = np.array([0, 0, .1])
    jlc.finalize(ik_solver='d')
    print(jlc.jnts[0].lnk.gl_pos, jlc.jnts[0].lnk.gl_rotmat)
    print(jlc.jnts[1].lnk.gl_pos, jlc.jnts[1].lnk.gl_rotmat)
    print(jlc.jnts[3].lnk.gl_pos, jlc.jnts[3].lnk.gl_rotmat)
    jnt_values = np.array([-np.pi/6, np.pi/3, .05, -np.pi/4])
    goal_pos, goal_rotmat = jlc.fk(jnt_values=jnt_values, update=False)
    mcm.mgm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)

    jnt_values = jlc.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat)
    print(jnt_values)
    jlc.goto_given_conf(jnt_values=jnt_values)

    jlc.gen_stickmodel(toggle_flange_frame=True, toggle_jnt_frames=False).attach_to(base)
    jlc.gen_meshmodel(alpha=.5).attach_to(base)
    base.run()
