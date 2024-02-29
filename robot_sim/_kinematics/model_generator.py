import numpy as np
import basis.constant as bc
import modeling.geometric_model as mgm
import modeling.collision_model as mcm
import modeling.model_collection as mmc
import basis.robot_math as rm
import robot_sim._kinematics.constant as rkc


def gen_tcp_frame(spos,
                  toggle_gl_pos,
                  toggle_gl_rotmat,
                  tcp_indicator_rgba=bc.magenta,
                  tcp_indicator_ax_radius=rkc.TCP_INDICATOR_STICK_RADIUS,
                  tcp_frame_rgb_mat=bc.myc_mat,
                  tcp_frame_alpha=1,
                  tcp_frame_ax_radius=rkc.FRAME_STICK_RADIUS,
                  tcp_frame_ax_length=rkc.FRAME_STICK_LENGTH_SHORT):
    """
    :param spos:
    :param toggle_gl_pos:
    :param toggle_gl_rotmat:
    :param attach_target: where to draw the frames to
    :param tcp_joint_id: single id or a list of ids
    :param loc_tcp_pos:
    :param loc_tcp_rotmat:
    :param tcp_indicator_rgba: color that used to render the tcp indicator
    :param tcp_indicator_ax_radius: major_radius the tcp indicator
    :param tcp_frame_ax_radius: major_radius the tcp coordinate frame
    :return: ModelCollection
    author: weiwei
    date: 20201125, 20230926
    """
    m_col = mmc.ModelCollection(name="tcp_frame")
    mgm.gen_dashed_stick(spos=spos,
                         epos=toggle_gl_pos,
                         radius=tcp_indicator_ax_radius,
                         rgba=tcp_indicator_rgba,
                         type="round").attach_to(m_col)
    mgm.gen_frame(pos=toggle_gl_pos,
                  rotmat=toggle_gl_rotmat,
                  ax_length=tcp_frame_ax_length,
                  ax_radius=tcp_frame_ax_radius,
                  rgb_mat=tcp_frame_rgb_mat,
                  alpha=tcp_frame_alpha).attach_to(m_col)
    return m_col


def gen_lnk_gmesh(lnk,
                  rgba=None,
                  toggle_cdprim=False,
                  toggle_cdmesh=False,
                  toggel_frame=False):
    gmodel = mgm.GeometricModel()
    # if lnk.cmodel == None:
    #     raise ValueError("Collision model is unavailable.")
    # else:
    if lnk.cmodel is not None:
        if toggle_cdmesh:
            lnk.cmodel.show_cdmesh()
        if toggle_cdprim:
            lnk.cmodel.show_cdprim()
        lnk.cmodel.attach_copy_to(gmodel)
        lnk.cmodel.unshow_cdmesh()
        lnk.cmodel.unshow_cdprim()
        if rgba is not None:
            gmodel.rgba = rgba
        if toggel_frame:
            mgm.gen_frame(pos=lnk.gl_pos, rotmat=lnk.gl_rotmat).attach_to(gmodel)
    return gmodel


def gen_anchor(anchor,
               radius=rkc.ANCHOR_BALL_RADIUS,
               frame_stick_radius=rkc.FRAME_STICK_RADIUS,
               frame_stick_length=rkc.FRAME_STICK_LENGTH_SHORT,
               toggle_frame=True):
    m_col = mmc.ModelCollection()
    mgm.gen_sphere(pos=anchor.pos,
                   radius=radius,
                   rgba=bc.jnt_parent_rgba).attach_to(m_col)
    if toggle_frame:
        mgm.gen_dashed_frame(pos=anchor.pos,
                             rotmat=anchor.rotmat,
                             ax_radius=frame_stick_radius,
                             ax_length=frame_stick_length).attach_to(m_col)
    return m_col


def gen_jnt(jnt,
            radius=rkc.JNT_RADIUS,
            frame_stick_radius=rkc.FRAME_STICK_RADIUS,
            frame_stick_length=rkc.FRAME_STICK_LENGTH_SHORT,
            toggle_frame_0=True,
            toggle_frame_q=True,
            toggle_lnk_mesh=False):
    m_col = mmc.ModelCollection()
    spos = jnt._gl_pos_0 - jnt.gl_motion_ax * radius
    epos = jnt._gl_pos_0 + jnt.gl_motion_ax * radius
    if jnt.type == rkc.JntType.REVOLUTE:
        mgm.gen_stick(spos=spos,
                      epos=jnt._gl_pos_0,
                      radius=radius,
                      rgba=bc.jnt_parent_rgba).attach_to(m_col)
        mgm.gen_stick(spos=jnt._gl_pos_0,
                      epos=epos,
                      radius=radius,
                      rgba=bc.jnt_child_rgba).attach_to(m_col)
    elif jnt.type == rkc.JntType.PRISMATIC:
        mgm.gen_stick(spos=spos,
                      epos=epos,
                      radius=radius * 1.2,
                      rgba=bc.jnt_parent_rgba,
                      type="round",
                      n_sec=6).attach_to(m_col)
        mgm.gen_stick(spos=jnt._gl_pos_0,
                      epos=jnt._gl_pos_0 + jnt._gl_motion_ax * jnt.motion_value,
                      radius=radius,
                      rgba=bc.jnt_child_rgba,
                      type="round",
                      n_sec=6).attach_to(m_col)
    else:
        raise ValueError("Joint type is not available.")
    if toggle_frame_0:
        mgm.gen_dashed_frame(pos=jnt._gl_pos_0,
                             rotmat=jnt._gl_rotmat_0,
                             ax_radius=frame_stick_radius,
                             ax_length=frame_stick_length).attach_to(m_col)
    if toggle_frame_q:
        mgm.gen_frame(pos=jnt._gl_pos_q,
                      rotmat=jnt._gl_rotmat_q,
                      ax_radius=frame_stick_radius,
                      ax_length=frame_stick_length).attach_to(m_col)
    if toggle_lnk_mesh and jnt.lnk is not None:
        gen_lnk_gmesh(jnt.lnk).attach_to(m_col)
    return m_col


def gen_jlc_stick(jlc,
                  rfd_radius=0.01,
                  link_ratio=.72,
                  jnt_ratio=1,
                  stick_rgba=bc.link_stick_rgba,
                  toggle_jnt_frames=False,
                  toggle_tcp_frame=True,
                  name='jlc_stick_model'):
    """

    :param jlc:
    :param rfd_radius: basic radius for extrusion
    :param link_ratio:
    :param jnt_ratio:
    :param stick_rgba:
    :param toggle_jnt_frames:
    :param toggle_tcp_frame:
    :param name:
    :return:
    """
    m_col = mmc.ModelCollection(name=name)
    gen_anchor(jlc.anchor,
               radius=jnt_ratio * rfd_radius,
               toggle_frame=toggle_jnt_frames).attach_to(m_col)
    if jlc.n_dof >= 1:
        mgm.gen_dashed_stick(spos=jlc.anchor.pos,
                             epos=jlc.jnts[0].gl_pos_0,
                             radius=link_ratio * rfd_radius,
                             type="rect",
                             rgba=stick_rgba).attach_to(m_col)
        for i in range(jlc.n_dof - 1):
            mgm.gen_stick(spos=jlc.jnts[i].gl_pos_q,
                          epos=jlc.jnts[i + 1].gl_pos_0,
                          radius=link_ratio * rfd_radius,
                          type="rect",
                          rgba=stick_rgba).attach_to(m_col)
            gen_jnt(jlc.jnts[i],
                    radius=jnt_ratio * rfd_radius,
                    toggle_frame_0=toggle_jnt_frames,
                    toggle_frame_q=toggle_jnt_frames).attach_to(m_col)
        gen_jnt(jlc.jnts[jlc.n_dof - 1],
                radius=jnt_ratio * rfd_radius,
                toggle_frame_0=toggle_jnt_frames,
                toggle_frame_q=toggle_jnt_frames).attach_to(m_col)
    if toggle_tcp_frame:
        spos = jlc.jnts[jlc.tcp_jnt_id].gl_pos_q
        tcp_gl_pos, tcp_gl_rotmat = jlc.cvt_tcp_loc_to_gl()
        gen_tcp_frame(spos=spos, toggle_gl_pos=tcp_gl_pos, toggle_gl_rotmat=tcp_gl_rotmat).attach_to(m_col)
    return m_col


def gen_jlc_mesh(jlc,
                 toggle_tcp_frame=False,
                 toggle_jnt_frames=False,
                 rgba=None,
                 toggle_cdprim=False,
                 toggle_cdmesh=False,
                 name='jlc_mesh_model'):
    m_col = mmc.ModelCollection(name=name)
    gen_lnk_gmesh(jlc.anchor.lnk,
                  rgba=rgba,
                  toggle_cdmesh=toggle_cdmesh,
                  toggle_cdprim=toggle_cdprim).attach_to(m_col)
    if jlc.n_dof >= 1:
        for i in range(jlc.n_dof):
            if jlc.jnts[i].lnk is not None:
                gen_lnk_gmesh(jlc.jnts[i].lnk,
                              rgba=rgba,
                              toggle_cdmesh=toggle_cdmesh,
                              toggle_cdprim=toggle_cdprim).attach_to(m_col)
    if toggle_tcp_frame:
        if jlc.n_dof >= 1:
            spos = jlc.jnts[jlc.tcp_jnt_id].gl_pos_q
            tcp_gl_pos, tcp_gl_rotmat = jlc.cvt_tcp_loc_to_gl()
            gen_tcp_frame(spos=spos, toggle_gl_pos=tcp_gl_pos, toggle_gl_rotmat=tcp_gl_rotmat,
                          tcp_frame_ax_length=rkc.FRAME_STICK_LENGTH_LONG).attach_to(m_col)
    if toggle_jnt_frames:
        # anchor
        gen_anchor(jlc.anchor,
                   frame_stick_radius=rkc.FRAME_STICK_RADIUS,
                   frame_stick_length=rkc.FRAME_STICK_LENGTH_LONG,
                   toggle_frame=toggle_jnt_frames).attach_to(m_col)
        if jlc.n_dof >= 1:
            # 0 frame
            mgm.gen_dashed_frame(pos=jlc.jnts[i]._gl_pos_0,
                                 rotmat=jlc.jnts[i]._gl_rotmat_0,
                                 ax_radius=rkc.FRAME_STICK_RADIUS,
                                 ax_length=rkc.FRAME_STICK_LENGTH_LONG).attach_to(m_col)
            # q frame
            mgm.gen_frame(pos=jlc.jnts[i]._gl_pos_q,
                          rotmat=jlc.jnts[i]._gl_rotmat_q,
                          ax_radius=rkc.FRAME_STICK_RADIUS,
                          ax_length=rkc.FRAME_STICK_LENGTH_LONG).attach_to(m_col)
    return m_col
