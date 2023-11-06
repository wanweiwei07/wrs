import numpy as np
import basis.constant as bc
import modeling.geometric_model as gm
import modeling.collision_model as cm
import modeling.model_collection as mc
import basis.robot_math as rm
import robot_sim.kinematics.constant as rkc


def gen_tcp_frame(jlc,
                  tcp_indicator_rgba=bc.magenta,
                  tcp_indicator_axis_radius=rkc.TCP_INDICATOR_STICK_RADIUS,
                  tcp_frame_rgb_mat=bc.myc_mat,
                  tcp_frame_alpha=1,
                  tcp_frame_axis_radius=rkc.FRAME_STICK_RADIUS,
                  tcp_frame_axis_length=rkc.FRAME_STICK_LENGTH_SHORT):
    """
    :param jlc: an instance of JLChain
    :param attach_target: where to draw the frames to
    :param tcp_joint_id: single id or a list of ids
    :param tcp_loc_pos:
    :param tcp_loc_rotmat:
    :param tcp_indicator_rgba: color that used to render the tcp indicator
    :param tcp_indicator_axis_radius: major_radius the tcp indicator
    :param tcp_frame_axis_radius: major_radius the tcp coordinate frame
    :return: ModelCollection
    author: weiwei
    date: 20201125, 20230926
    """
    m_col = mc.ModelCollection(name="tcp_frame")
    tcp_gl_pos, tcp_gl_rotmat = jlc.cvt_tcp_loc_to_gl()
    gm.gen_dashed_stick(spos=jlc.jnts[jlc.tcp_jnt_id].gl_pos_q,
                        epos=tcp_gl_pos,
                        radius=tcp_indicator_axis_radius,
                        rgba=tcp_indicator_rgba,
                        type="round").attach_to(m_col)
    gm.gen_frame(pos=tcp_gl_pos,
                 rotmat=tcp_gl_rotmat,
                 axis_length=tcp_frame_axis_length,
                 axis_radius=tcp_frame_axis_radius,
                 rgb_mat=tcp_frame_rgb_mat,
                 alpha=tcp_frame_alpha).attach_to(m_col)
    return m_col


def gen_link_mesh(link,
                  rgba=None,
                  toggle_frame=False):
    m_col = mc.ModelCollection()
    if link.collision_model == None:
        raise ValueError("Collision model is unavailable.")
    else:
        model = link.collision_model.copy()
        model.set_homomat(rm.homomat_from_posrot(link.gl_pos, link.gl_rotmat))
        if rgba is None:
            model.set_rgba(link.rgba)
        else:
            model.set_rgba(rgba)
        model.attach_to(m_col)
        if toggle_frame:
            gm.gen_frame(pos=link.gl_pos, rotmat=link.gl_rotmat).attach_to(m_col)
        return m_col


def gen_anchor(anchor,
               radius=rkc.ANCHOR_BALL_RADIUS,
               frame_stick_radius=rkc.FRAME_STICK_RADIUS,
               frame_stick_length=rkc.FRAME_STICK_LENGTH_SHORT,
               toggle_frame=True):
    m_col = mc.ModelCollection()
    gm.gen_sphere(pos=anchor.pos,
                  radius=radius,
                  rgba=bc.joint_parent_rgba).attach_to(m_col)
    if toggle_frame:
        gm.gen_dashed_frame(pos=anchor.pos,
                            rotmat=anchor.rotmat,
                            axis_radius=frame_stick_radius,
                            axis_length=frame_stick_length).attach_to(m_col)
    return m_col


def gen_joint(joint,
              radius=rkc.JOINT_RADIUS,
              frame_stick_radius=rkc.FRAME_STICK_RADIUS,
              frame_stick_length=rkc.FRAME_STICK_LENGTH_SHORT,
              toggle_frame_0=True,
              toggle_frame_q=True,
              toggle_link_mesh=False):
    m_col = mc.ModelCollection()
    spos = joint._gl_pos_0 - joint.gl_motion_ax * radius
    epos = joint._gl_pos_0 + joint.gl_motion_ax * radius
    if joint.type == rkc.JointType.REVOLUTE:
        gm.gen_stick(spos=spos,
                     epos=joint._gl_pos_0,
                     radius=radius,
                     rgba=bc.joint_parent_rgba).attach_to(m_col)
        gm.gen_stick(spos=joint._gl_pos_0,
                     epos=epos,
                     radius=radius,
                     rgba=bc.joint_child_rgba).attach_to(m_col)
    elif joint.type == rkc.JointType.PRISMATIC:
        # gm.gen_stick(spos=joint._gl_pos_0 - joint._gl_motion_axis * .01,
        #              epos=joint._gl_pos_0 + joint._gl_motion_axis * .01,
        gm.gen_stick(spos=spos,
                     epos=epos,
                     radius=radius * 1.2,
                     rgba=bc.joint_parent_rgba,
                     type="round",
                     n_sec=6).attach_to(m_col)
        gm.gen_stick(spos=joint._gl_pos_0,
                     epos=joint._gl_pos_0 + joint._gl_motion_ax * joint.motion_val,
                     radius=radius,
                     rgba=bc.joint_child_rgba,
                     type="round",
                     n_sec=6).attach_to(m_col)
    else:
        raise ValueError("Joint type is not available.")
    if toggle_frame_0:
        gm.gen_dashed_frame(pos=joint._gl_pos_0,
                            rotmat=joint._gl_rotmat_0,
                            axis_radius=frame_stick_radius,
                            axis_length=frame_stick_length).attach_to(m_col)
    if toggle_frame_q:
        gm.gen_frame(pos=joint._gl_pos_q,
                     rotmat=joint._gl_rotmat_q,
                     axis_radius=frame_stick_radius,
                     axis_length=frame_stick_length).attach_to(m_col)
    if toggle_link_mesh and joint.link is not None:
        gen_link_mesh(joint.link).attach_to(m_col)
    return m_col


def gen_jlc_stick(jlc,
                  reference_radius=0.01,
                  link_ratio=.72,
                  joint_ratio=1,
                  stick_rgba=bc.link_stick_rgba,
                  toggle_tcp_frame=True,
                  toggle_joint_frame=False,
                  name='jlc_stick_model'):
    m_col = mc.ModelCollection(name=name)
    gen_anchor(jlc.anchor,
               radius=joint_ratio * reference_radius,
               toggle_frame=toggle_joint_frame).attach_to(m_col)
    gm.gen_dashed_stick(spos=jlc.anchor.pos,
                        epos=jlc.jnts[0].gl_pos_0,
                        radius=link_ratio * reference_radius,
                        type="rect",
                        rgba=stick_rgba).attach_to(m_col)
    for i in range(jlc.n_dof - 1):
        gm.gen_stick(spos=jlc.jnts[i].gl_pos_q,
                     epos=jlc.jnts[i + 1].gl_pos_0,
                     radius=link_ratio * reference_radius,
                     type="rect",
                     rgba=stick_rgba).attach_to(m_col)
        gen_joint(jlc.jnts[i],
                  radius=joint_ratio * reference_radius,
                  toggle_frame_0=toggle_joint_frame,
                  toggle_frame_q=toggle_joint_frame).attach_to(m_col)
    gen_joint(jlc.jnts[jlc.n_dof - 1],
              radius=joint_ratio * reference_radius,
              toggle_frame_0=toggle_joint_frame,
              toggle_frame_q=toggle_joint_frame).attach_to(m_col)
    if toggle_tcp_frame:
        gen_tcp_frame(jlc=jlc).attach_to(m_col)
    return m_col


def gen_jlc_mesh(jlc,
                 toggle_tcp_frame=False,
                 toggle_joint_frame=False,
                 name='jlc_mesh_model',
                 rgba=None):
    m_col = mc.ModelCollection(name=name)
    for i in range(jlc.n_dof):
        if jlc.jnts[i].link is not None:
            gen_link_mesh(jlc.jnts[i].link, rgba=rgba).attach_to(m_col)
    if toggle_tcp_frame:
        gen_tcp_frame(jlc=jlc,
                      tcp_frame_axis_radius=rkc.FRAME_STICK_RADIUS,
                      tcp_frame_axis_length=rkc.FRAME_STICK_LENGTH_LONG).attach_to(m_col)
    if toggle_joint_frame:
        # anchor
        gen_anchor(jlc.anchor,
                   frame_stick_radius=rkc.FRAME_STICK_RADIUS,
                   frame_stick_length=rkc.FRAME_STICK_LENGTH_LONG,
                   toggle_frame=toggle_joint_frame).attach_to(m_col)
        # 0 frame
        gm.gen_dashed_frame(pos=jlc.jnts[i]._gl_pos_0,
                            rotmat=jlc.jnts[i]._gl_rotmat_0,
                            axis_radius=rkc.FRAME_STICK_RADIUS,
                            axis_length=rkc.FRAME_STICK_LENGTH_LONG).attach_to(m_col)
        # q frame
        gm.gen_frame(pos=jlc.jnts[i]._gl_pos_q,
                     rotmat=jlc.jnts[i]._gl_rotmat_q,
                     axis_radius=rkc.FRAME_STICK_RADIUS,
                     axis_length=rkc.FRAME_STICK_LENGTH_LONG).attach_to(m_col)
    return m_col
