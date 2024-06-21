import numpy as np
import robot_sim._kinematics.jl as rkjl
import robot_sim._kinematics.model_generator as rkmg
import robot_sim._kinematics.jlchain as rkjlc
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as mgm
import modeling.model_collection as mmc


class Motor(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3)):
        self._pos = pos
        self._rotmat = rotmat
        self.motion_value = 0
        self._body = mgm.GeometricModel(initor="./objects/motor_body.stl", rgb=rm.bc.jnt_parent_rgba)
        self._shaft = mgm.GeometricModel(initor="./objects/motor_shaft.stl", rgb=rm.bc.jnt_child_rgba)
        self._link = mgm.GeometricModel(initor="./objects/motor_link.stl", rgb=rm.bc.lnk_stick_rgba)

    @property
    def pos(self):
        return self._pos

    @property
    def rotmat(self):
        return self._rotmat

    @pos.setter
    def pos(self, pos):
        self._pos = pos

    @rotmat.setter
    def rotmat(self, rotmat):
        self._rotmat = rotmat

    @property
    def center_pos(self):
        return self._pos + self._rotmat @ np.array([0, 0, .0225])

    def gen_meshmodel(self,
                      toggle_stator_frame=True,
                      toggle_rotor_frame=True,
                      toggle_link=True,
                      toggle_rotax=True,
                      frame_center=None,
                      name="motor",
                      alpha=1):
        m_col = mmc.ModelCollection(name=name)
        tmp_body = self._body.copy()
        tmp_body.pos = self._pos
        tmp_body.rotmat = self._rotmat @ rm.rotmat_from_euler(0, 0, -np.pi / 4)
        tmp_body.alpha = alpha
        tmp_body.attach_to(m_col)
        tmp_shaft = self._shaft.copy()
        tmp_shaft.pos = self._pos
        tmp_shaft.alpha = alpha
        tmp_shaft.rotmat = self._rotmat @ rm.rotmat_from_euler(0, 0, self.motion_value)
        tmp_shaft.attach_to(m_col)
        in_rotmat = rm.rotmat_from_euler(np.pi / 2, 0, 0)
        in_stator_offset = np.array([0, 0, .0225])
        in_rotor_offset = np.array([0, 0, .05])
        if toggle_stator_frame:
            if frame_center is None:
                pos = self._pos + self._rotmat @ in_stator_offset
            else:
                pos = frame_center
            mgm.gen_frame(pos=pos, rotmat=self._rotmat @ in_rotmat,
                          ax_length=.05, ax_radius=.0006,
                                 alpha=alpha).attach_to(m_col)
        if toggle_rotor_frame:
            if frame_center is None:
                pos = self._pos + self._rotmat @ in_rotor_offset
            else:
                pos = frame_center
            mgm.gen_frame(pos=pos, rotmat=self._rotmat @ in_rotmat,
                          ax_length=.05, ax_radius=.0006,
                                 alpha=alpha).attach_to(m_col)
            mgm.gen_dashed_frame(pos=pos,
                                 rotmat=self._rotmat @ rm.rotmat_from_euler(0, 0, self.motion_value) @ in_rotmat,
                                 ax_length=.05, ax_radius=.0006,
                                 alpha=alpha).attach_to(m_col)
        if toggle_link:
            tmp_link = self._link.copy()
            tmp_link.pos = self._pos
            tmp_link.rotmat = self._rotmat @ rm.rotmat_from_euler(0, 0, -np.pi * 3 / 4)
            tmp_link.alpha = alpha
            tmp_link.attach_to(m_col)
        if toggle_rotax:
            mgm.gen_circarrow(self._rotmat @ np.array([0, 0, 1]), portion=5 / 6,
                              center=self._pos + self._rotmat @ np.array([0, 0, .06]), major_radius=.012,
                              minor_radius=.0006,
                              rgb=rm.bc.black, alpha=1, end_type='double').attach_to(m_col)
            mgm.gen_arrow(spos=self._pos, epos=self._pos + self._rotmat @ np.array([0, 0, .09]),
                          rgb=rm.bc.black, stick_radius=.0006).attach_to(m_col)
        return m_col


option = 'c'
if option == 'a':
    # visualize
    base = wd.World(cam_pos=np.array([.4, .4, .4]), lookat_pos=np.array([0, 0.025, 0]))
    motor = Motor(rotmat=rm.rotmat_from_euler(-np.pi / 2, 0, 0))
    motor.motion_value = np.pi / 6
    motor.gen_meshmodel(toggle_link=False, toggle_rotax=True, toggle_rotor_frame=False,
                        toggle_stator_frame=False).attach_to(base)
    base.run()
if option == 'b':
    # visualize
    base = wd.World(cam_pos=np.array([.4, .4, .4]), lookat_pos=np.array([0, 0.025, 0]))
    motor = Motor(rotmat=rm.rotmat_from_euler(-np.pi / 2, 0, 0))
    motor.motion_value = np.pi / 6
    motor.gen_meshmodel(toggle_link=False, toggle_rotax=False, toggle_rotor_frame=True,
                        toggle_stator_frame=True).attach_to(base)
    base.run()
if option == 'c':
    # visualize
    base = wd.World(cam_pos=np.array([.4, .4, .4]), lookat_pos=np.array([0, 0.025, 0]))
    # prev_motor = Motor(rotmat=rm.rotmat_from_euler(-np.pi / 2, 0, 0))
    # prev_motor.pos = np.array([0.121,0,0])
    # prev_motor.gen_meshmodel(toggle_link=True, toggle_rotax=False, toggle_rotor_frame=False,
    #                     toggle_stator_frame=False).attach_to(base)
    motor = Motor(rotmat=rm.rotmat_from_euler(-np.pi / 2, 0, 0))
    motor.motion_value = np.pi / 6
    motor.gen_meshmodel(toggle_link=True, toggle_rotax=False, toggle_rotor_frame=True,
                        toggle_stator_frame=True).attach_to(base)
    motor.motion_value = 0
    nxt_loc_pos = rm.rotmat_from_euler(0, 0, np.pi / 2) @ np.array([0, 0.121, 0])
    nxt_pos = motor.pos + motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value) @ nxt_loc_pos
    nxt_rotmat = motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value)
    nxt_motor = Motor(pos=nxt_pos, rotmat=nxt_rotmat)
    nxt_motor.gen_meshmodel(toggle_link=True, toggle_rotax=False, toggle_rotor_frame=True,
                            toggle_stator_frame=True, alpha=.2).attach_to(base)
    # nnxt_loc_pos = rm.rotmat_from_euler(0, 0, np.pi /2) @ np.array([0, 0.242, 0])
    # nnxt_pos = motor.pos + motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value) @ nnxt_loc_pos
    # nnxt_rotmat = motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value)
    # nnxt_motor = Motor(pos=nnxt_pos, rotmat=nnxt_rotmat)
    # nnxt_motor.gen_meshmodel(toggle_link=True, toggle_rotax=False, toggle_rotor_frame=False,
    #                     toggle_stator_frame=False, alpha=.3).attach_to(base)
    # nnnxt_loc_pos = rm.rotmat_from_euler(0, 0, np.pi /2) @ np.array([0, 0.363, 0])
    # nnnxt_pos = motor.pos + motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value) @ nnnxt_loc_pos
    # nnnxt_rotmat = motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value)
    # nnnxt_motor = Motor(pos=nnnxt_pos, rotmat=nnnxt_rotmat)
    # nnnxt_motor.gen_meshmodel(toggle_link=True, toggle_rotax=False, toggle_rotor_frame=False,
    #                     toggle_stator_frame=False, alpha=.3).attach_to(base)
    motor.motion_value = np.pi / 6
    nxt_loc_pos = rm.rotmat_from_euler(0, 0, np.pi / 2) @ np.array([0, 0.121, 0])
    nxt_pos = motor.pos + motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value) @ nxt_loc_pos
    nxt_rotmat = motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value)
    nxt_motor = Motor(pos=nxt_pos, rotmat=nxt_rotmat)
    nxt_motor.gen_meshmodel(toggle_link=True, toggle_rotax=False, toggle_rotor_frame=True,
                            toggle_stator_frame=True).attach_to(base)
    # nnxt_loc_pos = rm.rotmat_from_euler(0, 0, np.pi /2) @ np.array([0, 0.242, 0])
    # nnxt_pos = motor.pos + motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value) @ nnxt_loc_pos
    # nnxt_rotmat = motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value)
    # nnxt_motor = Motor(pos=nnxt_pos, rotmat=nnxt_rotmat)
    # nnxt_motor.gen_meshmodel(toggle_link=True, toggle_rotax=False, toggle_rotor_frame=True,
    #                     toggle_stator_frame=True).attach_to(base)
    base.run()
if option == 'd':
    # visualize
    base = wd.World(cam_pos=np.array([.4, .4, .4]), lookat_pos=np.array([0, 0.025, 0]))
    prev_motor = Motor(rotmat=rm.rotmat_from_euler(-np.pi / 2, 0, 0))
    prev_motor.pos = np.array([0.121, 0, 0])
    # prev_motor.gen_meshmodel(toggle_link=True, toggle_rotax=False, toggle_rotor_frame=False,
    #                     toggle_stator_frame=False).attach_to(base)
    motor = Motor(rotmat=rm.rotmat_from_euler(-np.pi / 2, 0, 0))
    motor.motion_value = np.pi / 6
    motor.gen_meshmodel(toggle_link=False, toggle_rotax=False, toggle_rotor_frame=True,
                        toggle_stator_frame=True, frame_center=motor.center_pos).attach_to(base)
    mgm.gen_stick(spos=prev_motor.center_pos, epos=motor.center_pos, rgb=rm.bc.lnk_stick_rgba,
                  radius=.0025, alpha=.7).attach_to(base)
    motor.motion_value = 0
    nxt_loc_pos = rm.rotmat_from_euler(0, 0, np.pi / 2) @ np.array([0, 0.121, 0])
    nxt_pos = motor.pos + motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value) @ nxt_loc_pos
    nxt_rotmat = motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value)
    nxt_motor = Motor(pos=nxt_pos, rotmat=nxt_rotmat)
    nxt_motor.gen_meshmodel(toggle_link=False, toggle_rotax=False, toggle_rotor_frame=True,
                            toggle_stator_frame=True, frame_center=nxt_motor.center_pos, alpha=.2).attach_to(base)
    mgm.gen_stick(spos=motor.center_pos, epos=nxt_motor.center_pos, rgb=rm.bc.lnk_stick_rgba, alpha=.2,
                  radius=.0025).attach_to(base)
    # nnxt_loc_pos = rm.rotmat_from_euler(0, 0, np.pi /2) @ np.array([0, 0.242, 0])
    # nnxt_pos = motor.pos + motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value) @ nnxt_loc_pos
    # nnxt_rotmat = motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value)
    # nnxt_motor = Motor(pos=nnxt_pos, rotmat=nnxt_rotmat)
    # nnxt_motor.gen_meshmodel(toggle_link=True, toggle_rotax=False, toggle_rotor_frame=False,
    #                     toggle_stator_frame=False, alpha=.3).attach_to(base)
    # nnnxt_loc_pos = rm.rotmat_from_euler(0, 0, np.pi /2) @ np.array([0, 0.363, 0])
    # nnnxt_pos = motor.pos + motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value) @ nnnxt_loc_pos
    # nnnxt_rotmat = motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value)
    # nnnxt_motor = Motor(pos=nnnxt_pos, rotmat=nnnxt_rotmat)
    # nnnxt_motor.gen_meshmodel(toggle_link=True, toggle_rotax=False, toggle_rotor_frame=False,
    #                     toggle_stator_frame=False, alpha=.3).attach_to(base)
    motor.motion_value = np.pi / 6
    nxt_loc_pos = rm.rotmat_from_euler(0, 0, np.pi / 2) @ np.array([0, 0.121, 0])
    nxt_pos = motor.pos + motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value) @ nxt_loc_pos
    nxt_rotmat = motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value)
    nxt_motor = Motor(pos=nxt_pos, rotmat=nxt_rotmat)
    nxt_motor.gen_meshmodel(toggle_link=False, toggle_rotax=False, toggle_rotor_frame=True,
                            toggle_stator_frame=True, frame_center=nxt_motor.center_pos).attach_to(base)
    # nnxt_loc_pos = rm.rotmat_from_euler(0, 0, np.pi /2) @ np.array([0, 0.242, 0])
    # nnxt_pos = motor.pos + motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value) @ nnxt_loc_pos
    # nnxt_rotmat = motor.rotmat @ rm.rotmat_from_euler(0, 0, motor.motion_value)
    # nnxt_motor = Motor(pos=nnxt_pos, rotmat=nnxt_rotmat)
    # nnxt_motor.gen_meshmodel(toggle_link=True, toggle_rotax=False, toggle_rotor_frame=True,
    #                     toggle_stator_frame=True).attach_to(base)
    mgm.gen_stick(spos=motor.center_pos, epos=nxt_motor.center_pos, rgb=rm.bc.lnk_stick_rgba,
                  radius=.0025, alpha=.7).attach_to(base)
    base.run()
if option == 'c':
    jnt_0 = rkjl.Joint()
    jnt_0.loc_pos = np.array([0, 0, 0])
    jnt_0.loc_rotmat = np.eye(3)
    jnt_0.loc_motion_ax = np.array([0, 1, 0])
    jnt_0.motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])
    jnt_0.update_globals(pos=np.zeros(3), rotmat=np.eye(3), motion_value=np.pi / 6)
    jnt_1 = rkjl.Joint()
    jnt_1.loc_pos = np.array([-.15, .15, 0])
    jnt_1.loc_rotmat = np.eye(3)
    jnt_1.loc_motion_ax = np.array([0, 0, 1])
    jnt_1.motion_range = np.array([-np.pi * 5 / 6, np.pi * 5 / 6])

    # visualize
    base = wd.World(cam_pos=np.array([.5, .5, .5]), lookat_pos=np.array([0, 0, 0]))
    rkmg.gen_jnt(jnt_0, toggle_frame_0=True, toggle_frame_q=True, toggle_actuation=True).attach_to(base)
    jnt_1.update_globals(pos=jnt_0.gl_pos_q, rotmat=jnt_0.gl_rotmat_q)
    rkmg.gen_jnt(jnt_1, toggle_frame_0=True, toggle_frame_q=True, toggle_actuation=False).attach_to(base)
    mgm.gen_dashed_stick(spos=jnt_0.gl_pos_0, epos=jnt_1.gl_pos_0, rgb=rm.bc.yellow, radius=.0025).attach_to(base)
    jnt_1.update_globals(pos=jnt_0.gl_pos_0, rotmat=jnt_0.gl_rotmat_0)
    rkmg.gen_jnt(jnt_1, toggle_frame_0=True, toggle_frame_q=True, toggle_actuation=False).attach_to(base)
    mgm.gen_stick(spos=jnt_0.gl_pos_0, epos=jnt_1.gl_pos_0, rgb=rm.bc.yellow, radius=.0025).attach_to(base)

base.run()
