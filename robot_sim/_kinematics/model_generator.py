import numpy as np
import basis.constant as bc
import modeling.geometric_model as mgm
import modeling.collision_model as mcm
import modeling.model_collection as mmc
import basis.robot_math as rm
import robot_sim._kinematics.constant as rkc
import motion.utils as utils


def gen_indicated_frame(spos,
                        gl_pos,
                        gl_rotmat,
                        indicator_rgba=bc.magenta,
                        indicator_ax_radius=rkc.TCP_INDICATOR_STICK_RADIUS,
                        frame_rgb_mat=bc.myc_mat,
                        frame_alpha=1,
                        frame_ax_radius=rkc.FRAME_STICK_RADIUS,
                        frame_ax_length=rkc.FRAME_STICK_LENGTH_MEDIUM,
                        name="indicated_frame"):
    """
    :param spos:
    :param gl_pos:
    :param gl_rotmat:
    :param indicator_rgba: color that used to render the frame indicator (line between pos of function id and flange)
    :param indicator_ax_radius: radius the tcp indicator
    :param frame_rgb_mat:
    :param frame_alpha:
    :param frame_ax_radius: major_radius the tcp frame
    :param frame_ax_length: length of the tcp frame axes
    :return: ModelCollection
    author: weiwei
    date: 20201125, 20230926, 20240301
    """
    m_col = mmc.ModelCollection(name=name)
    mgm.gen_dashed_stick(spos=spos,
                         epos=gl_pos,
                         radius=indicator_ax_radius,
                         rgba=indicator_rgba,
                         type="round").attach_to(m_col)
    mgm.gen_frame(pos=gl_pos,
                  rotmat=gl_rotmat,
                  ax_length=frame_ax_length,
                  ax_radius=frame_ax_radius,
                  rgb_mat=frame_rgb_mat,
                  alpha=frame_alpha).attach_to(m_col)
    return m_col


def gen_lnk_mesh(lnk,
                 rgb=None,
                 alpha=None,
                 toggle_cdprim=False,
                 toggle_cdmesh=False,
                 toggle_frame=False):
    if lnk.cmodel is not None:
        if toggle_cdmesh:
            lnk.cmodel.show_cdmesh()
        if toggle_cdprim:
            lnk.cmodel.show_cdprim()
        gmodel = mgm.GeometricModel(lnk.cmodel)
        lnk.cmodel.unshow_cdmesh()
        lnk.cmodel.unshow_cdprim()
        if rgb is not None:
            gmodel.rgb = rgb
        if alpha is not None:
            gmodel.alpha = alpha
        if toggle_frame:
            mgm.gen_frame(pos=lnk.gl_pos, rotmat=lnk.gl_rotmat).attach_to(gmodel)
        return gmodel
    else:
        return mgm.GeometricModel()


def gen_anchor(anchor,
               toggle_base_frame=True,
               toggle_flange_frame=True,
               radius=rkc.JNT_RADIUS,
               frame_stick_radius=rkc.FRAME_STICK_RADIUS,
               frame_stick_length=rkc.FRAME_STICK_LENGTH_MEDIUM):
    m_col = mmc.ModelCollection()
    mgm.gen_sphere(pos=anchor.pos,
                   radius=radius,
                   rgba=bc.jnt_parent_rgba).attach_to(m_col)
    mgm.gen_sphere(pos=anchor.gl_flange_pos,
                   radius=radius,
                   rgba=bc.jnt_parent_rgba).attach_to(m_col)
    mgm.gen_dashed_stick(spos=anchor.pos,
                         epos=anchor.gl_flange_pos,
                         radius=frame_stick_radius,
                         rgba=bc.jnt_parent_rgba).attach_to(m_col)
    if toggle_base_frame:
        mgm.gen_frame(pos=anchor.pos,
                      rotmat=anchor.rotmat,
                      ax_radius=frame_stick_radius,
                      ax_length=frame_stick_length,
                      alpha=.3).attach_to(m_col)
    if toggle_flange_frame:
        gen_indicated_frame(spos=anchor.pos, gl_pos=anchor.gl_flange_pos, gl_rotmat=anchor.gl_flange_rotmat,
                            indicator_rgba=bc.cyan,
                            frame_alpha=.3).attach_to(m_col)
    return m_col


def gen_jnt(jnt,
            toggle_frame_0=True,
            toggle_frame_q=True,
            toggle_lnk_mesh=False,
            radius=rkc.JNT_RADIUS,
            frame_stick_radius=rkc.FRAME_STICK_RADIUS,
            frame_stick_length=rkc.FRAME_STICK_LENGTH_MEDIUM):
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
        mgm.gen_arrow(spos=jnt._gl_pos_0, epos=jnt._gl_pos_0 + jnt.gl_motion_ax * rkc.FRAME_STICK_LENGTH_LONG,
                      rgba=bc.black).attach_to(m_col)
        mgm.gen_circarrow(axis=jnt.gl_motion_ax,
                          portion=(jnt.motion_range[1]-jnt.motion_range[0])/(2*np.pi),
                          center=jnt._gl_pos_0 + jnt.gl_motion_ax * rkc.FRAME_STICK_LENGTH_LONG / 3,
                          rgba=bc.black).attach_to(m_col)
    elif jnt.type == rkc.JntType.PRISMATIC:
        mgm.gen_stick(spos=spos,
                      epos=epos,
                      radius=radius * rkc.PRISMATIC_RATIO,
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
        gen_lnk_mesh(jnt.lnk).attach_to(m_col)
    return m_col


def gen_jlc_stick(jlc,
                  stick_rgba=bc.lnk_stick_rgba,
                  toggle_jnt_frames=False,
                  toggle_flange_frame=True,
                  name='jlc_stick_model',
                  jnt_radius=rkc.JNT_RADIUS,
                  lnk_radius=rkc.LNK_STICK_RADIUS):
    """
    :param jlc:
    :param jnt_radius: basic radius for extrusion
    :param lnk_radius:
    :param stick_rgba:
    :param toggle_jnt_frames:
    :param toggle_flange_frame:
    :param name:
    :return:
    """
    m_col = mmc.ModelCollection(name=name)
    # anchor
    gen_anchor(jlc.anchor,
               radius=jnt_radius * rkc.ANCHOR_RATIO,
               toggle_base_frame=toggle_jnt_frames).attach_to(m_col)
    # jlc
    if jlc.n_dof >= 1:
        mgm.gen_dashed_stick(spos=jlc.anchor.gl_flange_pos,
                             epos=jlc.jnts[0].gl_pos_0,
                             radius=lnk_radius,
                             type="rect",
                             rgba=stick_rgba).attach_to(m_col)
        for i in range(jlc.n_dof - 1):
            mgm.gen_stick(spos=jlc.jnts[i].gl_pos_q,
                          epos=jlc.jnts[i + 1].gl_pos_0,
                          radius=lnk_radius,
                          type="rect",
                          rgba=stick_rgba).attach_to(m_col)
            gen_jnt(jlc.jnts[i],
                    radius=jnt_radius,
                    toggle_frame_0=toggle_jnt_frames,
                    toggle_frame_q=toggle_jnt_frames).attach_to(m_col)
        gen_jnt(jlc.jnts[jlc.n_dof - 1],
                radius=jnt_radius,
                toggle_frame_0=toggle_jnt_frames,
                toggle_frame_q=toggle_jnt_frames).attach_to(m_col)
    if toggle_flange_frame:
        spos = jlc.jnts[jlc.flange_jnt_id].gl_pos_q
        gen_indicated_frame(spos=spos, gl_pos=jlc.gl_flange_pos, gl_rotmat=jlc.gl_flange_rotmat,
                            indicator_rgba=bc.spring_green, frame_alpha=.3).attach_to(m_col)
    return m_col


def gen_jlc_stick_by_jnt_values(jlc,
                                jnt_values,
                                stick_rgba=bc.lnk_stick_rgba,
                                toggle_jnt_frames=False,
                                toggle_flange_frame=True,
                                name='jlc_stick_model',
                                jnt_radius=rkc.JNT_RADIUS,
                                lnk_radius=rkc.LNK_STICK_RADIUS):
    jnt_values_bk = jlc.get_jnt_values()
    jlc.go_given_conf(jnt_values=jnt_values)
    m_col = gen_jlc_stick(jlc, stick_rgba, toggle_jnt_frames, toggle_flange_frame, name, jnt_radius, lnk_radius)
    jlc.go_given_conf(jnt_values=jnt_values_bk)
    return m_col


def gen_jlc_mesh(jlc,
                 rgb=None,
                 alpha=None,
                 toggle_flange_frame=False,
                 toggle_jnt_frames=False,
                 toggle_cdprim=False,
                 toggle_cdmesh=False,
                 name='jlc_mesh_model'):
    m_col = mmc.ModelCollection(name=name)
    gen_lnk_mesh(jlc.anchor.lnk,
                 rgb=rgb,
                 alpha=alpha,
                 toggle_cdmesh=toggle_cdmesh,
                 toggle_cdprim=toggle_cdprim).attach_to(m_col)
    if jlc.n_dof >= 1:
        for i in range(jlc.n_dof):
            if jlc.jnts[i].lnk is not None:
                gen_lnk_mesh(jlc.jnts[i].lnk,
                             rgb=rgb,
                             alpha=alpha,
                             toggle_cdmesh=toggle_cdmesh,
                             toggle_cdprim=toggle_cdprim).attach_to(m_col)
    if toggle_flange_frame:
        spos = jlc.jnts[jlc.flange_jnt_id].gl_pos_q
        gen_indicated_frame(spos=spos, gl_pos=jlc.gl_flange_pos, gl_rotmat=jlc.gl_flange_rotmat,
                            indicator_rgba=bc.spring_green, frame_alpha=.3,
                            frame_ax_length=rkc.FRAME_STICK_LENGTH_MEDIUM).attach_to(m_col)
    if toggle_jnt_frames:
        # anchor
        gen_anchor(jlc.anchor,
                   frame_stick_radius=rkc.FRAME_STICK_RADIUS,
                   frame_stick_length=rkc.FRAME_STICK_LENGTH_LONG,
                   toggle_base_frame=toggle_jnt_frames).attach_to(m_col)
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
