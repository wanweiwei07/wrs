import os
import wrs.basis.robot_math as rm
import wrs.robot_sim.end_effectors.ee_interface as ei
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.grasping.grasp as gg
import wrs.robot_sim._kinematics.jlchain as rkjlc


class XHandRight(ei.EEInterface):

    def __init__(self, pos=rm.zeros(3), rotmat=rm.eye(3),
                 coupling_offset_pos=rm.zeros(3),
                 coupling_offset_rotmat=rm.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.AABB, name='xhand_right'):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        # ======= palm ======== #
        self.coupling.loc_flange_pose_list[0] = (coupling_offset_pos, coupling_offset_rotmat)
        self.palm = rkjlc.rkjl.Anchor(name=name + "_palm_anchor",
                                      pos=self.coupling.gl_flange_pose_list[0][0],
                                      rotmat=self.coupling.gl_flange_pose_list[0][1],
                                      n_flange=5)
        self.palm.loc_flange_pose_list[0] = [rm.vec(0.0228, -0.0095, 0.0305), rm.eye(3)]  # thumb
        self.palm.loc_flange_pose_list[1] = [rm.vec(0.0265, -0.0065, 0.0899), rm.eye(3)]  # index
        self.palm.loc_flange_pose_list[2] = [rm.vec(0.004, -0.0065, 0.1082), rm.eye(3)]  # middle
        self.palm.loc_flange_pose_list[3] = [rm.vec(-0.016, -0.0065, 0.1052), rm.eye(3)]  # ring
        self.palm.loc_flange_pose_list[4] = [rm.vec(-0.036, -0.0065, 0.1022), rm.eye(3)]  # pinky
        self.palm.lnk_list[0].name = name + "_palm_lnk"
        self.palm.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "right_hand_link_.stl"),
            name=name + "_palm",
            cdmesh_type=self.cdmesh_type)
        self.palm.lnk_list[0].cmodel.rgba = rm.const.dim_gray
        self.thumb_jlc = self.__define_thumb(palm=self.palm, name=self.name, file_dir=current_file_dir)
        self.index_jlc = self.__define_index(palm=self.palm, name=self.name, file_dir=current_file_dir)
        self.middle_jlc = self.__define_middle(palm=self.palm, name=self.name, file_dir=current_file_dir)
        self.ring_jlc = self.__define_ring(palm=self.palm, name=self.name, file_dir=current_file_dir)
        self.pinky_jlc = self.__define_pinky(palm=self.palm, name=self.name, file_dir=current_file_dir)
        # finalize
        self.thumb_jlc.finalize()
        self.index_jlc.finalize()
        self.middle_jlc.finalize()
        self.ring_jlc.finalize()
        self.pinky_jlc.finalize()
        # backup
        self.jaw_width_bk = []
        # acting center
        self.loc_acting_center_pos = rm.vec(0,-0.075,.075)
        self.loc_acting_center_rotmat = rm.rotmat_from_euler(rm.pi/2,rm.pi/2,0)
        # collision detection
        # collisions
        self.cdelements = (self.palm.lnk_list[0],
                           self.thumb_jlc.jnts[0].lnk,
                           self.thumb_jlc.jnts[1].lnk,
                           self.thumb_jlc.jnts[2].lnk,
                           self.index_jlc.jnts[0].lnk,
                           self.index_jlc.jnts[1].lnk,
                           self.middle_jlc.jnts[0].lnk,
                           self.middle_jlc.jnts[1].lnk,
                           self.ring_jlc.jnts[0].lnk,
                           self.ring_jlc.jnts[1].lnk,
                           self.pinky_jlc.jnts[0].lnk,
                           self.pinky_jlc.jnts[1].lnk)

    def __define_thumb(self, palm, name, file_dir):
        jlc = rkjlc.JLChain(pos=palm.gl_flange_pose_list[0][0],
                            rotmat=palm.gl_flange_pose_list[0][1],
                            n_dof=3, name=name + "_thumb_jlc")
        # joint 1
        jlc.jnts[0].loc_motion_ax = -rm.const.z_ax
        jlc.jnts[0].motion_range = [.0, 1.83]
        jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(file_dir, "meshes", "right_hand_thumb_bend_link_.stl"),
            name=name + "_thumb_bend_link",
            cdmesh_type=self.cdmesh_type)
        jlc.jnts[0].lnk.cmodel.rgba = rm.const.hug_gray
        # joint 2
        jlc.jnts[1].loc_pos = rm.vec(0.028599, 0.0083177, 0.00178)
        jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(0.2618, 0, 0.0407)
        jlc.jnts[1].loc_motion_ax = -rm.const.y_ax
        jlc.jnts[1].motion_range = [-1.05, 1.57]
        jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(file_dir, "meshes", "right_hand_thumb_rota_link1_.stl"),
            name=name + "_thumb_rota_link1",
            cdmesh_type=self.cdmesh_type)
        jlc.jnts[1].lnk.cmodel.rgba = rm.const.hug_gray
        # joint 3
        jlc.jnts[2].loc_pos = rm.vec(0.0553, .0, .0)
        jlc.jnts[2].loc_motion_ax = -rm.const.y_ax
        jlc.jnts[2].motion_range = [-0.175, 1.83]
        jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(file_dir, "meshes", "right_hand_thumb_rota_link2_.stl"),
            name=name + "_thumb_rota_link1",
            cdmesh_type=self.cdmesh_type)
        jlc.jnts[2].lnk.cmodel.rgba = rm.const.hug_gray
        jlc.set_flange(loc_flange_pos=rm.vec(0.051, 0, 0))
        return jlc

    def __define_index(self, palm, name, file_dir):
        jlc = rkjlc.JLChain(pos=palm.gl_flange_pose_list[1][0],
                            rotmat=palm.gl_flange_pose_list[1][1],
                            n_dof=3, name=name + "_index_jlc")
        # joint 1
        jlc.jnts[0].loc_motion_ax = rm.const.y_ax
        jlc.jnts[0].motion_range = [-0.175, 0.175]
        jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(file_dir, "meshes", "right_hand_index_bend_link_.stl"),
            name=name + "_index_bend_link",
            cdmesh_type=self.cdmesh_type)
        jlc.jnts[0].lnk.cmodel.rgba = rm.const.hug_gray
        # joint 2
        jlc.jnts[1].loc_pos = rm.vec(.0, .0, 0.0178)
        jlc.jnts[1].loc_motion_ax = rm.const.x_ax
        jlc.jnts[1].motion_range = [.0, 1.92]
        jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(file_dir, "meshes", "right_hand_index_rota_link1_.stl"),
            name=name + "_index_rota_link1",
            cdmesh_type=self.cdmesh_type)
        jlc.jnts[1].lnk.cmodel.rgba = rm.const.hug_gray
        # joint 3
        jlc.jnts[2].loc_pos = rm.vec(0, 0, 0.0558)
        jlc.jnts[2].loc_motion_ax = rm.const.x_ax
        jlc.jnts[2].motion_range = [.0, 1.92]
        jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(file_dir, "meshes", "right_hand_index_rota_link2_.stl"),
            name=name + "_index_rota_link1",
            cdmesh_type=self.cdmesh_type)
        jlc.jnts[2].lnk.cmodel.rgba = rm.const.hug_gray
        jlc.set_flange(loc_flange_pos=rm.vec(0, 0, 0.042107))
        return jlc

    def __define_middle(self, palm, name, file_dir):
        jlc = rkjlc.JLChain(pos=palm.gl_flange_pose_list[2][0],
                            rotmat=palm.gl_flange_pose_list[2][1],
                            n_dof=2, name=name + "_mid_jlc")
        # joint 1
        jlc.jnts[0].loc_motion_ax = rm.const.x_ax
        jlc.jnts[0].motion_range = [0, 1.92]
        jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(file_dir, "meshes", "right_hand_mid_link1_.stl"),
            name=name + "_mid_link1",
            cdmesh_type=self.cdmesh_type)
        jlc.jnts[0].lnk.cmodel.rgba = rm.const.hug_gray
        # joint 2
        jlc.jnts[1].loc_pos = rm.vec(.0, .0, 0.0558)
        jlc.jnts[1].loc_motion_ax = rm.const.x_ax
        jlc.jnts[1].motion_range = [.0, 1.92]
        jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(file_dir, "meshes", "right_hand_mid_link2_.stl"),
            name=name + "_mid_link2",
            cdmesh_type=self.cdmesh_type)
        jlc.jnts[1].lnk.cmodel.rgba = rm.const.hug_gray
        jlc.set_flange(loc_flange_pos=rm.vec(0, 0, 0.0425))
        return jlc

    def __define_ring(self, palm, name, file_dir):
        jlc = rkjlc.JLChain(pos=palm.gl_flange_pose_list[3][0],
                            rotmat=palm.gl_flange_pose_list[3][1],
                            n_dof=2, name=name + "_ring_jlc")
        # joint 1
        jlc.jnts[0].loc_motion_ax = rm.const.x_ax
        jlc.jnts[0].motion_range = [0, 1.92]
        jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(file_dir, "meshes", "right_hand_ring_link1_.stl"),
            name=name + "_ring_link1",
            cdmesh_type=self.cdmesh_type)
        jlc.jnts[0].lnk.cmodel.rgba = rm.const.hug_gray
        # joint 2
        jlc.jnts[1].loc_pos = rm.vec(.0, .0, 0.0558)
        jlc.jnts[1].loc_motion_ax = rm.const.x_ax
        jlc.jnts[1].motion_range = [.0, 1.92]
        jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(file_dir, "meshes", "right_hand_ring_link2_.stl"),
            name=name + "_ring_link2",
            cdmesh_type=self.cdmesh_type)
        jlc.jnts[1].lnk.cmodel.rgba = rm.const.hug_gray
        jlc.set_flange(loc_flange_pos=rm.vec(0, 0, 0.0425))
        return jlc

    def __define_pinky(self, palm, name, file_dir):
        jlc = rkjlc.JLChain(pos=palm.gl_flange_pose_list[4][0],
                            rotmat=palm.gl_flange_pose_list[4][1],
                            n_dof=2, name=name + "_pinky_jlc")
        # joint 1
        jlc.jnts[0].loc_motion_ax = rm.const.x_ax
        jlc.jnts[0].motion_range = [0, 1.92]
        jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(file_dir, "meshes", "right_hand_pinky_link1_.stl"),
            name=name + "_pinky_link1",
            cdmesh_type=self.cdmesh_type)
        jlc.jnts[0].lnk.cmodel.rgba = rm.const.hug_gray
        # joint 2
        jlc.jnts[1].loc_pos = rm.vec(.0, .0, 0.0558)
        jlc.jnts[1].loc_motion_ax = rm.const.x_ax
        jlc.jnts[1].motion_range = [.0, 1.92]
        jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(file_dir, "meshes", "right_hand_pinky_link2_.stl"),
            name=name + "_pinky_link2",
            cdmesh_type=self.cdmesh_type)
        jlc.jnts[1].lnk.cmodel.rgba = rm.const.hug_gray
        jlc.set_flange(loc_flange_pos=rm.vec(0, 0, 0.0425))
        return jlc

    @property
    def thumb(self):
        return self.thumb_jlc

    @property
    def index(self):
        return self.index_jlc

    @property
    def middle(self):
        return self.middle_jlc

    @property
    def ring(self):
        return self.ring_jlc

    @property
    def pinky(self):
        return self.pinky_jlc

    def rand_conf(self):
        thumb_conf = self.thumb_jlc.rand_conf()
        index_conf = self.index_jlc.rand_conf()
        mid_conf = self.middle_jlc.rand_conf()
        ring_conf = self.ring_jlc.rand_conf()
        pinky_conf = self.pinky_jlc.rand_conf()
        return rm.np.hstack([thumb_conf, index_conf, mid_conf, ring_conf, pinky_conf])

    def goto_given_conf(self, conf):
        self.thumb_jlc.goto_given_conf(conf[:3])
        self.index_jlc.goto_given_conf(conf[3:6])
        self.middle_jlc.goto_given_conf(conf[6:8])
        self.ring_jlc.goto_given_conf(conf[8:10])
        self.pinky_jlc.goto_given_conf(conf[10:12])

    def backup_state(self):
        self.oiee_list_bk.append(self.oiee_list.copy())
        self.oiee_pose_list_bk.append([oiee.loc_pose for oiee in self.oiee_list])
        self.jaw_width_bk.append(self.get_jaw_width())

    def restore_state(self):
        self.oiee_list = self.oiee_list_bk.pop()
        oiee_pose_list = self.oiee_pose_list_bk.pop()
        for i, oiee in enumerate(self.oiee_list):
            oiee.loc_pose = oiee_pose_list[i]
        self.update_oiee()
        jaw_width = self.jaw_width_bk.pop()
        if jaw_width != self.get_jaw_width():
            self.change_jaw_width(jaw_width=jaw_width)

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.coupling.pos = self._pos
        self.coupling.rotmat = self.rotmat
        self.palm.fix_to(self.coupling.gl_flange_pose_list[0][0], self.coupling.gl_flange_pose_list[0][1])
        self.thumb_jlc.fix_to(self.palm.gl_flange_pose_list[0][0], self.palm.gl_flange_pose_list[0][1])
        self.index_jlc.fix_to(self.palm.gl_flange_pose_list[1][0], self.palm.gl_flange_pose_list[1][1])
        self.middle_jlc.fix_to(self.palm.gl_flange_pose_list[2][0], self.palm.gl_flange_pose_list[2][1])
        self.ring_jlc.fix_to(self.palm.gl_flange_pose_list[3][0], self.palm.gl_flange_pose_list[3][1])
        self.pinky_jlc.fix_to(self.palm.gl_flange_pose_list[4][0], self.palm.gl_flange_pose_list[4][1])
        self.update_oiee()

    def get_jaw_width(self):
        raise NotImplementedError

    @ei.EEInterface.assert_oiee_decorator
    def change_jaw_width(self, jaw_width):
        pass

    def get_ee_values(self):
        return self.get_jaw_width()

    @ei.EEInterface.assert_oiee_decorator
    def change_ee_values(self, ee_values):
        """
        interface
        :param ee_values:
        :return:
        """
        self.change_jaw_width(jaw_width=ee_values)

    def hold(self, obj_cmodel, jaw_width=None):
        if jaw_width is None:
            jaw_width = self.jaw_range[0]
        self.change_jaw_width(jaw_width=jaw_width)
        return super().hold(obj_cmodel=obj_cmodel)

    def release(self, obj_cmodel, jaw_width=None):
        obj_lnk = super().release(obj_cmodel=obj_cmodel)
        if jaw_width is None:
            jaw_width = self.jaw_range[1]
        self.change_jaw_width(jaw_width=jaw_width)
        return obj_lnk

    def grip_at_by_twovecs(self,
                           jaw_center_pos,
                           approaching_direction,
                           thumb_opening_direction,
                           jaw_width):
        """
        specifying the gripping pose using two axes -- approaching vector and opening vector
        the thumb is the finger at the y+ direction
        :param jaw_center_pos:
        :param approaching_direction: jaw_center's approaching motion_vec
        :param thumb_opening_direction: jaw_center's opening motion_vec (thumb is the left finger, or finger 0)
        :param jaw_width: [ee_values, jaw_center_pos, jaw_center_rotmat, eef_root_pos, eef_root_rotmat]
        :return:
        """
        self.change_jaw_width(jaw_width)
        param_list = self.align_acting_center_by_twovecs(acting_center_pos=jaw_center_pos,
                                                         approaching_vec=approaching_direction,
                                                         side_vec=thumb_opening_direction)
        return gg.Grasp(ee_values=jaw_width, ac_pos=param_list[0], ac_rotmat=param_list[1])

    def grip_at_by_pose(self, jaw_center_pos, jaw_center_rotmat, jaw_width):
        """
        :param jaw_center_pos:
        :param jaw_center_rotmat:
        :param jaw_width:
        :return: [ee_values, jaw_center_pos, gl_jaw_center_rotmat, eef_root_pos, eef_root_rotmat]
        """
        self.change_jaw_width(jaw_width)
        param_list = self.align_acting_center_by_pose(acting_center_pos=jaw_center_pos,
                                                      acting_center_rotmat=jaw_center_rotmat)
        return gg.Grasp(ee_values=jaw_width, ac_pos=param_list[0], ac_rotmat=param_list[1])

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_fingertip_frames=False):
        m_col = mmc.ModelCollection(name=self.name + '_stickmodel')
        self.coupling.gen_stickmodel(toggle_root_frame=False, toggle_flange_frame=False).attach_to(m_col)
        self.palm.gen_stickmodel(toggle_root_frame=toggle_jnt_frames, toggle_flange_frame=toggle_fingertip_frames).attach_to(
            m_col)
        self.thumb_jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                      toggle_flange_frame=toggle_fingertip_frames).attach_to(m_col)
        self.index_jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                      toggle_flange_frame=toggle_fingertip_frames).attach_to(m_col)
        self.middle_jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                       toggle_flange_frame=toggle_fingertip_frames).attach_to(m_col)
        self.ring_jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                     toggle_flange_frame=toggle_fingertip_frames).attach_to(m_col)
        self.pinky_jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                      toggle_flange_frame=toggle_fingertip_frames).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False):
        m_col = mmc.ModelCollection(name=self.name + '_meshmodel')
        self.coupling.gen_meshmodel(rgb=rgb,
                                    alpha=alpha,
                                    toggle_flange_frame=False,
                                    toggle_root_frame=False,
                                    toggle_cdmesh=toggle_cdmesh,
                                    toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.palm.gen_meshmodel(rgb=rgb,
                                alpha=alpha,
                                toggle_root_frame=toggle_jnt_frames,
                                toggle_flange_frame=False,
                                toggle_cdmesh=toggle_cdmesh,
                                toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.thumb_jlc.gen_meshmodel(rgb=rgb,
                                     alpha=alpha,
                                     toggle_jnt_frames=toggle_jnt_frames,
                                     toggle_flange_frame=False,
                                     toggle_cdmesh=toggle_cdmesh,
                                     toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.index_jlc.gen_meshmodel(rgb=rgb,
                                     alpha=alpha,
                                     toggle_jnt_frames=toggle_jnt_frames,
                                     toggle_flange_frame=False,
                                     toggle_cdmesh=toggle_cdmesh,
                                     toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.middle_jlc.gen_meshmodel(rgb=rgb,
                                      alpha=alpha,
                                      toggle_jnt_frames=toggle_jnt_frames,
                                      toggle_flange_frame=False,
                                      toggle_cdmesh=toggle_cdmesh,
                                      toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.ring_jlc.gen_meshmodel(rgb=rgb,
                                    alpha=alpha,
                                    toggle_jnt_frames=toggle_jnt_frames,
                                    toggle_flange_frame=False,
                                    toggle_cdmesh=toggle_cdmesh,
                                    toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.pinky_jlc.gen_meshmodel(rgb=rgb,
                                     alpha=alpha,
                                     toggle_jnt_frames=toggle_jnt_frames,
                                     toggle_flange_frame=False,
                                     toggle_cdmesh=toggle_cdmesh,
                                     toggle_cdprim=toggle_cdprim).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        self._gen_oiee_meshmodel(m_col=m_col, rgb=rgb, alpha=alpha, toggle_cdprim=toggle_cdprim,
                                 toggle_cdmesh=toggle_cdmesh)
        return m_col


if __name__ == '__main__':
    from wrs import wd, mgm

    base = wd.World(cam_pos=rm.vec(.5, -.5, .5), lookat_pos=rm.vec(0, 0, 0.05))
    mgm.gen_frame().attach_to(base)
    xhand = XHandRight()
    # xhand.goto_given_conf(xhand.rand_conf())
    # xhand.gen_stickmodel(toggle_tcp_frame=True).attach_to(base)
    xhand.gen_meshmodel(toggle_jnt_frames=True, toggle_cdprim=True, toggle_tcp_frame=True, alpha=.3).attach_to(base)
    # xhand.gen_meshmodel(toggle_cdprim=True).attach_to(base)
    base.run()
