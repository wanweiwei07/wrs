from wrs import basis as rm, robot_sim as rkjlc, robot_sim as gi, modeling as mcm, modeling as mgm

import os
import numpy as np
import wrs.modeling.model_collection as mmc
import wrs.basis.robot_math as rm


class Robotiq140(gi.GripperInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 coupling_offset_pos=np.zeros(3),
                 coupling_offset_rotmat=np.eye(3),
                 cdmesh_type=mcm.mc.CDMeshType.DEFAULT,
                 name='robotiq140'):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        # flange
        self.coupling.loc_flange_pose_list[0] = (coupling_offset_pos, coupling_offset_rotmat)
        if np.any(self.coupling.loc_flange_pose_list[0][0]):
            self.coupling.lnk_list[0].cmodel = mcm.gen_stick(spos=np.zeros(3),
                                                             epos=self.coupling.loc_flange_pose_list[0][0],
                                                             type="rect",
                                                             radius=0.035,
                                                             rgb=np.array([.35, .35, .35]),
                                                             alpha=1,
                                                             n_sec=24)
        # jaw range
        self.jaw_range = np.array([.0, .140])  # Robotiq 140's max opening
        # palm
        self.palm = rkjlc.rkjl.Anchor(name=name + "_palm",
                                      pos=self.coupling.gl_flange_pose_list[0][0],
                                      rotmat=self.coupling.gl_flange_pose_list[0][1],
                                      n_flange=4)
        self.palm.loc_flange_pose_list[0] = [np.array([0, .0306011, .054904]), np.eye(3)]
        self.palm.loc_flange_pose_list[1] = [np.array([0, .0127, .06142]), np.eye(3)]
        self.palm.loc_flange_pose_list[2] = [np.array([0, -.0306011, .054904]), np.eye(3)]
        self.palm.loc_flange_pose_list[3] = [np.array([0, -.0127, .06142]), np.eye(3)]
        self.palm.lnk_list[0].name = name + "_palm_lnk"
        self.palm.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "robotiq_arg2f_base_link.stl"),
            cdmesh_type=self.cdmesh_type)
        self.palm.lnk_list[0].cmodel.rgba = rm.bc.dim_gray
        # ======= left finger ======== #
        # left finger outer
        self.lft_outer_jlc = rkjlc.JLChain(pos=self.palm.gl_flange_pose_list[0][0],
                                           rotmat=self.palm.gl_flange_pose_list[0][1],
                                           n_dof=4, name=name + "_left_outer")
        # left finger outer (joint 0 / outer_knuckle)
        self.lft_outer_jlc.jnts[0].loc_pos = np.zeros(3)
        self.lft_outer_jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(np.pi / 2 + .725, 0, np.pi)
        self.lft_outer_jlc.jnts[0].loc_motion_ax = rm.bc.x_ax
        self.lft_outer_jlc.jnts[0].motion_range = [.0, .7]
        self.lft_outer_jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_140_outer_knuckle.stl"),
            cdmesh_type=self.cdmesh_type)
        self.lft_outer_jlc.jnts[0].lnk.cmodel.rgba = rm.bc.hug_gray
        # left finger outer (joint 1 / outer_finger)
        self.lft_outer_jlc.jnts[1].loc_pos = np.array([0, .01821998610742, .0260018192872234])
        self.lft_outer_jlc.jnts[1].loc_motion_ax = np.array([1, 0, 0])
        self.lft_outer_jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_140_outer_finger.stl"),
            cdmesh_type=self.cdmesh_type)
        self.lft_outer_jlc.jnts[1].lnk.cmodel.rgba = rm.bc.dim_gray
        # left finger outer (joint 2 / inner_finger)
        self.lft_outer_jlc.jnts[2].loc_pos = np.array([0, .0817554015893473, -.0282203446692936])
        self.lft_outer_jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(-.725, 0, 0)
        self.lft_outer_jlc.jnts[2].loc_motion_ax = np.array([1, 0, 0])
        self.lft_outer_jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_140_inner_finger.stl"),
            cdmesh_type=self.cdmesh_type)
        self.lft_outer_jlc.jnts[2].lnk.cmodel.rgba = rm.bc.dim_gray
        # left finger outer (joint 3 / inner_finger_pad)
        self.lft_outer_jlc.jnts[3].loc_pos = np.array([0, 0.0420203446692936, -.028])
        self.lft_outer_jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_140_pad.stl"), cdmesh_type=self.cdmesh_type)
        self.lft_outer_jlc.jnts[3].lnk.cmodel.rgba = rm.bc.hug_gray
        # left finger inner
        self.lft_inner_jlc = rkjlc.JLChain(pos=self.palm.gl_flange_pose_list[1][0],
                                           rotmat=self.palm.gl_flange_pose_list[1][1],
                                           n_dof=1, name=name + "_left_inner")
        self.lft_inner_jlc.jnts[0].loc_pos = np.zeros(3)
        self.lft_inner_jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(np.pi / 2 + .725, 0, np.pi)
        self.lft_inner_jlc.jnts[0].loc_motion_ax = np.array([1, 0, 0])
        self.lft_inner_jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_140_inner_knuckle.stl"),
            cdmesh_type=self.cdmesh_type)
        self.lft_inner_jlc.jnts[0].lnk.cmodel.rgba = rm.bc.dim_gray
        # ======= right finger ======== #
        # rgt finger outer
        self.rgt_outer_jlc = rkjlc.JLChain(pos=self.palm.gl_flange_pose_list[2][0],
                                           rotmat=self.palm.gl_flange_pose_list[2][1],
                                           n_dof=4, name=name + "_right_outer")
        # right finger outer (joint 0 / outer_knuckle)
        self.rgt_outer_jlc.jnts[0].loc_pos = np.zeros(3)
        self.rgt_outer_jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(np.pi / 2 + 0.725, 0, 0)
        self.rgt_outer_jlc.jnts[0].loc_motion_ax = np.array([1, 0, 0])
        self.rgt_outer_jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_140_outer_knuckle.stl"),
            cdmesh_type=self.cdmesh_type)
        self.rgt_outer_jlc.jnts[0].lnk.cmodel.rgba = rm.bc.hug_gray
        # right finger outer (joint 1 / outer_finger)
        self.rgt_outer_jlc.jnts[1].loc_pos = np.array([0, .01821998610742, .0260018192872234])
        self.rgt_outer_jlc.jnts[1].loc_motion_ax = np.array([1, 0, 0])
        self.rgt_outer_jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_140_outer_finger.stl"),
            cdmesh_type=self.cdmesh_type)
        self.rgt_outer_jlc.jnts[1].lnk.cmodel.rgba = rm.bc.dim_gray
        # right finger outer (joint 2 / inner_finger)
        self.rgt_outer_jlc.jnts[2].loc_pos = np.array([0, 0.0817554015893473, -0.0282203446692936])
        self.rgt_outer_jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(-.725, 0, 0)
        self.rgt_outer_jlc.jnts[2].loc_motion_ax = np.array([1, 0, 0])
        self.rgt_outer_jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_140_inner_finger.stl"),
            cdmesh_type=self.cdmesh_type)
        self.rgt_outer_jlc.jnts[2].lnk.cmodel.rgba = rm.bc.dim_gray
        # right finger outer (joint 3 / inner_finger_pad)
        self.rgt_outer_jlc.jnts[3].loc_pos = np.array([0, 0.0420203446692936, -.028])
        self.rgt_outer_jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_140_pad.stl"), cdmesh_type=self.cdmesh_type)
        self.rgt_outer_jlc.jnts[3].lnk.cmodel.rgba = rm.bc.hug_gray
        # right finger inner
        self.rgt_inner_jlc = rkjlc.JLChain(pos=self.palm.gl_flange_pose_list[3][0],
                                           rotmat=self.palm.gl_flange_pose_list[3][1],
                                           n_dof=1, name=name + "_right_inner")
        self.rgt_inner_jlc.jnts[0].loc_pos = np.zeros(3)
        self.rgt_inner_jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(np.pi / 2 + .725, 0, 0)
        self.rgt_inner_jlc.jnts[0].loc_motion_ax = np.array([1, 0, 0])
        self.rgt_inner_jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_140_inner_knuckle.stl"),
            cdmesh_type=self.cdmesh_type)
        self.rgt_inner_jlc.jnts[0].lnk.cmodel.rgba = rm.bc.dim_gray
        # finalize
        self.lft_outer_jlc.finalize()
        self.lft_inner_jlc.finalize()
        self.rgt_outer_jlc.finalize()
        self.rgt_inner_jlc.finalize()
        # acting center
        self.loc_acting_center_pos = np.array([0, 0, .2])
        # collisions
        self.cdmesh_elements = (self.palm.lnk_list[0],
                                self.lft_outer_jlc.jnts[0].lnk,
                                self.lft_outer_jlc.jnts[1].lnk,
                                self.lft_outer_jlc.jnts[2].lnk,
                                self.lft_outer_jlc.jnts[3].lnk,
                                self.lft_inner_jlc.jnts[0].lnk,
                                self.rgt_outer_jlc.jnts[0].lnk,
                                self.rgt_outer_jlc.jnts[1].lnk,
                                self.rgt_outer_jlc.jnts[2].lnk,
                                self.rgt_outer_jlc.jnts[3].lnk,
                                self.rgt_inner_jlc.jnts[0].lnk)

    def fix_to(self, pos, rotmat, jaw_width=None):
        self._pos = pos
        self.rotmat = rotmat
        if jaw_width is not None:
            self.change_jaw_width(jaw_width=jaw_width)
        self.coupling.pos = self._pos
        self.coupling.rotmat = self.rotmat
        self.palm.pos = self.coupling.gl_flange_pose_list[0][0]
        self.palm.rotmat = self.coupling.gl_flange_pose_list[0][1]
        self.lft_outer_jlc.fix_to(self.palm.gl_flange_pose_list[0][0], self.palm.gl_flange_pose_list[0][1])
        self.lft_inner_jlc.fix_to(self.palm.gl_flange_pose_list[1][0], self.palm.gl_flange_pose_list[1][1])
        self.rgt_outer_jlc.fix_to(self.palm.gl_flange_pose_list[2][0], self.palm.gl_flange_pose_list[2][1])
        self.rgt_inner_jlc.fix_to(self.palm.gl_flange_pose_list[3][0], self.palm.gl_flange_pose_list[3][1])
        self.update_oiee()

    def change_jaw_width(self, jaw_width):
        """
        :param jaw_width:
        :return:
        author: dyanamic changes made by junbo 20240912
        """
        if jaw_width > 0.140:
            raise ValueError("ee_values must be 0mm~140mm!")
        angle = np.clip(self.lft_outer_jlc.jnts[0].motion_range[1] - rm.math.asin(
            (np.sin(self.lft_outer_jlc.jnts[0].motion_range[1]) / self.jaw_range[1]) * jaw_width),
                        self.lft_outer_jlc.jnts[0].motion_range[0], self.lft_outer_jlc.jnts[0].motion_range[1])
        if self.lft_outer_jlc.jnts[0].motion_range[0] <= angle <= self.lft_outer_jlc.jnts[0].motion_range[1]:
            self.lft_outer_jlc.goto_given_conf(jnt_values=np.array([-angle, 0.0, angle, 0.0]))
            self.lft_inner_jlc.goto_given_conf(jnt_values=np.array([-angle]))
            self.rgt_outer_jlc.goto_given_conf(jnt_values=np.array([-angle, 0.0, angle, 0.0]))
            self.rgt_inner_jlc.goto_given_conf(jnt_values=np.array([-angle]))
        # dynamic adjustment
        homo_flange_gl = rm.homomat_from_posrot(self._pos, self.rotmat)
        homo_pad_gl = self.rgt_outer_jlc.jnts[3].gl_homomat_0
        homo_pad_flange = np.linalg.inv(homo_flange_gl) @ homo_pad_gl
        self.loc_acting_center_pos[2] = homo_pad_flange[2,3]

    def get_jaw_width(self):
        angle = -self.lft_inner_jlc.jnts[0].motion_value
        return self.jaw_range[1] * np.sin(self.lft_outer_jlc.jnts[0].motion_range[1] - angle) / np.sin(
            self.lft_outer_jlc.jnts[0].motion_range[1])

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       name="xarm_gripper_stickmodel"):
        m_col = mmc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_root_frame=False, toggle_flange_frame=False).attach_to(m_col)
        self.palm.gen_stickmodel(toggle_root_frame=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        self.lft_outer_jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                          toggle_flange_frame=False).attach_to(m_col)
        self.lft_inner_jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                          toggle_flange_frame=False).attach_to(m_col)
        self.rgt_outer_jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                          toggle_flange_frame=False).attach_to(m_col)
        self.rgt_inner_jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                          toggle_flange_frame=False).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name="xarm_gripper_meshmodel"):
        m_col = mmc.ModelCollection(name=name)
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
        self.lft_outer_jlc.gen_meshmodel(rgb=rgb,
                                         alpha=alpha,
                                         toggle_jnt_frames=toggle_jnt_frames,
                                         toggle_flange_frame=False,
                                         toggle_cdmesh=toggle_cdmesh,
                                         toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.lft_inner_jlc.gen_meshmodel(rgb=rgb,
                                         alpha=alpha,
                                         toggle_jnt_frames=toggle_jnt_frames,
                                         toggle_flange_frame=False,
                                         toggle_cdmesh=toggle_cdmesh,
                                         toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.rgt_outer_jlc.gen_meshmodel(rgb=rgb,
                                         alpha=alpha,
                                         toggle_jnt_frames=toggle_jnt_frames,
                                         toggle_flange_frame=False,
                                         toggle_cdmesh=toggle_cdmesh,
                                         toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.rgt_inner_jlc.gen_meshmodel(rgb=rgb,
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
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[2, 0, 0], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    gripper = Robotiq140()
    gripper.change_jaw_width(0.0)
    print(gripper.loc_acting_center_pos)
    gripper.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    model = gripper.gen_meshmodel(alpha=.3)
    model.attach_to(base)
    base.run()