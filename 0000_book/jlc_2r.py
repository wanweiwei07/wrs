from wrs import rm, mgm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim._kinematics.model_generator as rkmg
import wrs.modeling.model_collection as mmc


def gen_stand():
    """
    generate the base model of a 2R robot
    :param pos:
    :param rotmat:
    :return:
    """
    m_col = mmc.ModelCollection()
    hinge = (mgm.gen_stick(spos=rm.vec(.015, 0, 0),
                           epos=rm.vec(-.015, 0, 0),
                           radius=.015,
                           rgb=rm.const.steel_gray))
    hinge.alpha = .5
    hinge.attach_to(m_col)
    holder = mgm.gen_box(xyz_lengths=rm.vec(.03, .03, .02),
                         pos=rm.vec(0, 0, -.01),
                         rgb=rm.const.steel_gray)
    holder.alpha = .5
    holder.attach_to(m_col)
    stand = mgm.gen_stick(spos=rm.vec(0, 0, -0.02),
                          epos=rm.vec(0, 0, -0.0225),
                          radius=0.03,
                          rgb=rm.const.steel_gray)
    stand.alpha = .5
    stand.attach_to(m_col)
    return m_col


if __name__ == '__main__':
    from wrs import wd

    base = wd.World(cam_pos=rm.vec(1, 0, .1), lookat_pos=rm.vec(0, 0, 0.1))
    gen_stand().attach_to(base)
    mgm.gen_frame().attach_to(base)

    jlc = rkjlc.JLChain(n_dof=2)
    jlc.jnts[0].loc_pos = rm.vec(0, 0, 0)
    jlc.jnts[0].loc_motion_ax = rm.vec(1, 0, 0)
    jlc.jnts[1].loc_pos = rm.vec(0, 0, .1)
    jlc.jnts[1].loc_motion_ax = rm.vec(1, 0, 0)
    jlc._loc_flange_pos = rm.vec(0, 0, .1)
    jlc.finalize()

    tcp_gl_pos, _ = jlc.fk(jnt_values=rm.np.radians([10, 20]), update=True)
    linear_ellipsoid_mat, _ = jlc.manipulability_mat()
    mgm.gen_ellipsoid(pos=tcp_gl_pos, axes_mat=linear_ellipsoid_mat).attach_to(base)
    rkmg.gen_jlc_stick(jlc, toggle_flange_frame=True, toggle_jnt_frames=False).attach_to(base)

    base.run()
