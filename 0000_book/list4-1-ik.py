from wrs import wd, rm, mgm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim._kinematics.model_generator as rkmg


option = 'c'
jlc = rkjlc.JLChain(n_dof=6)
# root
jlc.anchor.pos = rm.np.zeros(3)
jlc.anchor.rotmat = rm.np.eye(3)
# joint 0
jlc.jnts[0].loc_pos = rm.np.array([0, 0, .15])
jlc.jnts[0].loc_rotmat = rm.np.eye(3)
jlc.jnts[0].loc_motion_ax = rm.np.array([0, 0, 1])
jlc.jnts[0].motion_range = rm.np.array([-rm.pi * 5 / 6, rm.pi * 5 / 6])
# joint 1
jlc.jnts[1].loc_pos = rm.np.array([0, 0, .15])
jlc.jnts[1].loc_rotmat = rm.np.eye(3)
jlc.jnts[1].loc_motion_ax = rm.np.array([0, 1, 0])
jlc.jnts[1].motion_range = rm.np.array([-rm.pi * 5 / 6, rm.pi * 5 / 6])
# joint 2
jlc.jnts[2].loc_pos = rm.np.array([0, 0, .15])
jlc.jnts[2].loc_rotmat = rm.np.eye(3)
jlc.jnts[2].loc_motion_ax = rm.np.array([0, 1, 0])
jlc.jnts[2].motion_range = rm.np.array([-rm.pi * 5 / 6, rm.pi * 5 / 6])
# joint 3
jlc.jnts[3].loc_pos = rm.np.array([0, 0, .15])
jlc.jnts[3].loc_rotmat = rm.np.eye(3)
jlc.jnts[3].loc_motion_ax = rm.np.array([0, 0, 1])
jlc.jnts[3].motion_range = rm.np.array([-rm.pi * 5 / 6, rm.pi * 5 / 6])
# joint 4
jlc.jnts[4].loc_pos = rm.np.array([0, 0, .15])
jlc.jnts[4].loc_rotmat = rm.np.eye(3)
jlc.jnts[4].loc_motion_ax = rm.np.array([0, 1, 0])
jlc.jnts[4].motion_range = rm.np.array([-rm.pi * 5 / 6, rm.pi * 5 / 6])
# joint 5
jlc.jnts[5].loc_pos = rm.np.array([0, 0, .15])
jlc.jnts[5].loc_rotmat = rm.np.eye(3)
jlc.jnts[5].loc_motion_ax = rm.np.array([0, 0, 1])
jlc.jnts[5].motion_range = rm.np.array([-rm.pi * 5 / 6, rm.pi * 5 / 6])
jlc._loc_flange_pos = rm.np.array([0, 0, .03])
jlc.finalize()
# visualize
base = wd.World(cam_pos=rm.np.array([1, 1, 1]), lookat_pos=rm.np.array([0, 0, .3]))
if option == 'a':
    jlc.goto_given_conf(jnt_values=rm.np.array([rm.pi / 12, -rm.pi / 3, rm.pi * 2 / 3, rm.pi / 12, rm.pi / 3, 0]))
    jlc.gen_stickmodel(toggle_jnt_frames=False, toggle_actuation=False, lnk_alpha=.2, jnt_alpha=.2).attach_to(base)
    prev_pos = jlc.gl_flange_pos
    mgm.gen_dashed_stick(spos=jlc.jnts[0].gl_pos_0, epos=jlc.gl_flange_pos, rgb=rm.const.black, radius=.0015,
                         len_solid=.015, len_interval=.012, alpha=.2).attach_to(base)
    jlc.goto_given_conf(
        jnt_values=rm.np.array([rm.pi / 12 - rm.pi / 18, -rm.pi / 3, rm.pi * 2 / 3, rm.pi / 12, rm.pi / 3, 0]))
    jlc.gen_stickmodel(toggle_jnt_frames=False, toggle_actuation=False, lnk_alpha=.7, jnt_alpha=1).attach_to(base)
    rkmg.gen_jnt(jlc.jnts[0], toggle_frame_0=False, toggle_frame_q=False, toggle_actuation=True, alpha=1).attach_to(
        base)
    mgm.gen_dashed_stick(spos=jlc.jnts[0].gl_pos_0, epos=jlc.gl_flange_pos, rgb=rm.const.black, radius=.0015,
                         len_solid=.015, len_interval=.012).attach_to(base)
    mgm.gen_arrow(spos=prev_pos, epos=jlc.gl_flange_pos, rgb=rm.const.black,
                  stick_radius=rkjlc.const.FRAME_STICK_RADIUS).attach_to(base)
if option == 'b':
    jlc.goto_given_conf(
        jnt_values=rm.np.array([rm.pi / 12, -rm.pi / 3, rm.pi * 2 / 3, rm.pi / 12, rm.pi / 3, 0]))
    jlc.gen_stickmodel(toggle_jnt_frames=False, toggle_actuation=False, lnk_alpha=.2, jnt_alpha=.2).attach_to(base)
    prev_pos = jlc.gl_flange_pos
    mgm.gen_dashed_stick(spos=jlc.jnts[2].gl_pos_0, epos=jlc.gl_flange_pos, rgb=rm.const.black, radius=.0015,
                         len_solid=.015, len_interval=.012, alpha=.2).attach_to(base)
    jlc.goto_given_conf(jnt_values=rm.np.array(
        [rm.pi / 12, -rm.pi / 3, rm.pi * 2 / 3 - rm.pi / 18, rm.pi / 12, rm.pi / 3, 0]))
    jlc.gen_stickmodel(toggle_jnt_frames=False, toggle_actuation=False, lnk_alpha=.7, jnt_alpha=1).attach_to(base)
    rkmg.gen_jnt(jlc.jnts[2], toggle_frame_0=False, toggle_frame_q=False, toggle_actuation=True, alpha=1).attach_to(
        base)
    mgm.gen_dashed_stick(spos=jlc.jnts[2].gl_pos_0, epos=jlc.gl_flange_pos, rgb=rm.const.black, radius=.0015,
                         len_solid=.015, len_interval=.012).attach_to(base)
    mgm.gen_arrow(spos=prev_pos, epos=jlc.gl_flange_pos, rgb=rm.bc.black,
                  stick_radius=rkjlc.const.FRAME_STICK_RADIUS).attach_to(base)
if option == 'c':
    # original
    jlc.goto_given_conf(jnt_values=rm.np.array([rm.pi / 12, -rm.pi / 3, rm.pi * 2 / 3, rm.pi / 12, rm.pi / 3, 0]))
    jlc.gen_stickmodel(toggle_jnt_frames=False, toggle_actuation=False, lnk_alpha=.2, jnt_alpha=.2).attach_to(base)
    prev_pos = jlc.gl_flange_pos
    mgm.gen_dashed_stick(spos=jlc.jnts[0].gl_pos_0, epos=jlc.gl_flange_pos, rgb=rm.const.black, radius=.0015,
                         len_solid=.015, len_interval=.012, alpha=.2).attach_to(base)
    mgm.gen_dashed_stick(spos=jlc.jnts[2].gl_pos_0, epos=jlc.gl_flange_pos, rgb=rm.const.black, radius=.0015,
                         len_solid=.015, len_interval=.012, alpha=.2).attach_to(base)
    # first
    jlc.goto_given_conf(
        jnt_values=rm.np.array([rm.pi / 12 - rm.pi / 18, -rm.pi / 3, rm.pi * 2 / 3, rm.pi / 12, rm.pi / 3, 0]))
    mgm.gen_dashed_arrow(spos=prev_pos, epos=jlc.gl_flange_pos, rgb=rm.const.black,
                         stick_radius=rkjlc.const.FRAME_STICK_RADIUS, alpha=.2).attach_to(base)
    mgm.gen_dashed_stick(spos=jlc.jnts[0].gl_pos_0, epos=jlc.gl_flange_pos, rgb=rm.const.black, radius=.0015,
                         len_solid=.015, len_interval=.012, alpha=.2).attach_to(base)
    resulted_sum_pos = jlc.gl_flange_pos
    jlc.goto_given_conf(jnt_values=rm.np.array(
        [rm.pi / 12, -rm.pi / 3, rm.pi * 2 / 3 - rm.pi / 18, rm.pi / 12, rm.pi / 3, 0]))
    mgm.gen_dashed_arrow(spos=prev_pos, epos=jlc.gl_flange_pos, rgb=rm.const.black,
                         stick_radius=rkjlc.const.FRAME_STICK_RADIUS, alpha=.2).attach_to(base)
    mgm.gen_dashed_stick(spos=jlc.jnts[2].gl_pos_0, epos=jlc.gl_flange_pos, rgb=rm.const.black, radius=.0015,
                         len_solid=.015, len_interval=.012, alpha=.2).attach_to(base)
    resulted_sum_pos += jlc.gl_flange_pos
    mgm.gen_dashed_arrow(spos=prev_pos, epos=resulted_sum_pos - prev_pos, rgb=rm.const.black,
                         stick_radius=rkjlc.const.FRAME_STICK_RADIUS).attach_to(base)
    # third
    jlc.goto_given_conf(jnt_values=rm.np.array(
        [rm.pi / 12 - rm.pi / 18, -rm.pi / 3, rm.pi * 2 / 3 - rm.pi / 18, rm.pi / 12 - rm.pi / 18, rm.pi / 3, 0]))
    jlc.gen_stickmodel(toggle_jnt_frames=False, toggle_actuation=False, lnk_alpha=.7, jnt_alpha=1).attach_to(base)
    rkmg.gen_jnt(jlc.jnts[0], toggle_frame_0=False, toggle_frame_q=False, toggle_actuation=True, alpha=1).attach_to(
        base)
    rkmg.gen_jnt(jlc.jnts[2], toggle_frame_0=False, toggle_frame_q=False, toggle_actuation=True, alpha=1).attach_to(
        base)
    mgm.gen_arrow(spos=prev_pos, epos=jlc.gl_flange_pos, rgb=rm.const.black,
                         stick_radius=rkjlc.const.FRAME_STICK_RADIUS).attach_to(base)
base.run()
