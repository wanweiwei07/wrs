import robot_sim._kinematics.jlchain as lib_jlc
import robot_sim._kinematics.jl as lib_jl
import robot_sim._kinematics.model_generator as lib_jlm
import modeling.model_collection as lib_mc
import visualization.panda.world as wd
import modeling.geometric_model as gm
import numpy as np
import basis.constant as cst


def gen_stand():
    """
    generate the base model of a 2R robot
    :param pos:
    :param rotmat:
    :return:
    """
    m_col = lib_mc.ModelCollection()
    hinge = (gm.gen_stick(spos=np.array([.015, 0, 0]),
                          epos=np.array([-.015, 0, 0]),
                          radius=.015,
                          rgba=cst.steel_gray))
    hinge.set_alpha(.5)
    hinge.attach_to(m_col)
    holder = gm.gen_box(xyz_lengths=np.array([.03, .03, .02]),
                        pos=np.array([0, 0, -.01]),
                        rgba=cst.steel_gray)
    holder.set_alpha(.5)
    holder.attach_to(m_col)
    stand = gm.gen_stick(spos=np.array([0, 0, -0.02]),
                         epos=np.array([0, 0, -0.0225]),
                         radius=0.03,
                         rgba=cst.steel_gray)
    stand.set_alpha(.5)
    stand.attach_to(m_col)
    return m_col


if __name__ == '__main__':
    base = wd.World(cam_pos=[1.5, 0, .2], lookat_pos=[0, 0, 0.2])
    gen_stand().attach_to(base)
    gm.gen_frame().attach_to(base)

    jlc = lib_jlc.JLChain(n_dof=4)
    jlc.jnts[1].loc_pos = np.array([0, 0, 0])
    jlc.jnts[1].loc_motion_axis = np.array([1, 0, 0])
    jlc.jnts[2].loc_pos = np.array([0, 0, .1])
    jlc.jnts[2].loc_motion_axis = np.array([0, 1, 0])
    jlc.jnts[3].change_type(type=lib_jl.JointType.PRISMATIC, motion_rng=np.array([-1, 1]))
    jlc.jnts[3].loc_pos = np.array([0, 0, .1])
    jlc.jnts[3].loc_motion_axis = np.array([0, 0, 1])
    jlc.jnts[4].loc_pos = np.array([0, 0, .1])
    jlc.jnts[4].loc_motion_axis = np.array([0, 0, 1])
    jlc.tcp_loc_pos = np.array([0, 0, .1])
    jlc.finalize()
    jnt_values = np.array([np.radians(0), np.radians(0), .03, np.radians(0)])
    tcp_pos_physical, tcp_rotmat_physical = jlc.fk(joint_values=jnt_values, update=True)
    lib_jlm.gen_stick_model(jlc, toggle_tcpcs=True, toggle_jntscs=False).attach_to(base)

    base.run()
