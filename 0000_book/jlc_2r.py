import robot_sim._kinematics.jlchain as lib_jlc
import visualization.panda.world as wd
import robot_sim._kinematics.model_generator as lib_jlm
import modeling.geometric_model as gm
import numpy as np
import basis.constant as cst
import modeling.model_collection as lib_mc


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
    hinge.alpha = .5
    hinge.attach_to(m_col)
    holder = gm.gen_box(xyz_lengths=np.array([.03, .03, .02]),
                        pos=np.array([0, 0, -.01]),
                        rgba=cst.steel_gray)
    holder.alpha = .5
    holder.attach_to(m_col)
    stand = gm.gen_stick(spos=np.array([0, 0, -0.02]),
                         epos=np.array([0, 0, -0.0225]),
                         radius=0.03,
                         rgba=cst.steel_gray)
    stand.alpha = .5
    stand.attach_to(m_col)
    return m_col


if __name__ == '__main__':
    base = wd.World(cam_pos=[1, 0, .1], lookat_pos=[0, 0, 0.1])
    gen_stand().attach_to(base)
    gm.gen_frame().attach_to(base)

    jlc = lib_jlc.JLChain(n_dof=2)
    jlc.jnts[1].loc_pos = np.array([0, 0, 0])
    jlc.jnts[1].loc_motion_ax = np.array([1, 0, 0])
    jlc.jnts[2].loc_pos = np.array([0, 0, .1])
    jlc.jnts[2].loc_motion_ax = np.array([1, 0, 0])
    jlc._loc_flange_pos = np.array([0, 0, .1])
    jlc.finalize()

    tcp_gl_pos, _ = jlc.fk(joint_values=np.radians([10, 20]), update=True)
    linear_ellipsoid_mat, _ = jlc.manipulability_mat()
    gm.gen_ellipsoid(pos=tcp_gl_pos, axes_mat=linear_ellipsoid_mat).attach_to(base)
    lib_jlm.gen_stick_model(jlc, toggle_tcpcs=True, toggle_jntscs=False).attach_to(base)

    base.run()
