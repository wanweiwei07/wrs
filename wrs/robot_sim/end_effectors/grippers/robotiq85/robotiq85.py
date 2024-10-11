import os
import math
import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi


class Robotiq85(gpi.GripperInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 coupling_offset_pos=np.zeros(3),
                 coupling_offset_rotmat=np.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT,
                 name='robotiq85'):
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
        self.jaw_range = np.array([.0, .085])
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
            initor=os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_base_link.stl"),
            cdmesh_type=self.cdmesh_type)
        self.palm.lnk_list[0].cmodel.rgba = rm.const.dim_gray
        # ======= left finger ======== #
        # left finger outer
        self.lft_outer_jlc = rkjlc.JLChain(pos=self.palm.gl_flange_pose_list[0][0],
                                           rotmat=self.palm.gl_flange_pose_list[0][1],
                                           n_dof=4, name=name + "_left_outer")
        # left finger outer (joint 0 / outer_knuckle)
        self.lft_outer_jlc.jnts[0].loc_pos = np.zeros(3)
        self.lft_outer_jlc.jnts[0].loc_motion_ax = np.array([1, 0, 0])
        self.lft_outer_jlc.jnts[0].motion_range = [.0, .8]
        self.lft_outer_jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_outer_knuckle.stl"),
            cdmesh_type=self.cdmesh_type)
        self.lft_outer_jlc.jnts[0].lnk.cmodel.rgba = rm.const.hug_gray
        # left finger outer (joint 1 / outer_finger)
        self.lft_outer_jlc.jnts[1].loc_pos = np.array([0, .0315, -.0041])
        self.lft_outer_jlc.jnts[1].loc_motion_ax = np.array([1, 0, 0])
        self.lft_outer_jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_outer_finger.stl"),
            cdmesh_type=self.cdmesh_type)
        self.lft_outer_jlc.jnts[1].lnk.cmodel.rgba = rm.const.dim_gray
        # left finger outer (joint 2 / inner_finger)
        self.lft_outer_jlc.jnts[2].loc_pos = np.array([0, .0061, .0471])
        self.lft_outer_jlc.jnts[2].loc_motion_ax = np.array([1, 0, 0])
        self.lft_outer_jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_inner_finger.stl"),
            cdmesh_type=self.cdmesh_type)
        self.lft_outer_jlc.jnts[2].lnk.cmodel.rgba = rm.const.dim_gray
        # left finger outer (joint 3 / inner_finger_pad)
        self.lft_outer_jlc.jnts[3].loc_pos = np.zeros(3)
        self.lft_outer_jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_pad.stl"), cdmesh_type=self.cdmesh_type)
        self.lft_outer_jlc.jnts[3].lnk.cmodel.rgba = rm.const.hug_gray
        # left finger inner
        self.lft_inner_jlc = rkjlc.JLChain(pos=self.palm.gl_flange_pose_list[1][0],
                                           rotmat=self.palm.gl_flange_pose_list[1][1],
                                           n_dof=1, name=name + "_left_inner")
        self.lft_inner_jlc.jnts[0].loc_pos = np.zeros(3)
        self.lft_inner_jlc.jnts[0].loc_motion_ax = np.array([1, 0, 0])
        self.lft_inner_jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_inner_knuckle.stl"),
            cdmesh_type=self.cdmesh_type)
        self.lft_inner_jlc.jnts[0].lnk.cmodel.rgba = rm.const.dim_gray
        # ======= right finger ======== #
        # rgt finger outer
        self.rgt_outer_jlc = rkjlc.JLChain(pos=self.palm.gl_flange_pose_list[2][0],
                                           rotmat=self.palm.gl_flange_pose_list[2][1],
                                           n_dof=4, name=name + "_right_outer")
        # right finger outer (joint 0 / outer_knuckle)
        self.rgt_outer_jlc.jnts[0].loc_pos = np.zeros(3)
        self.rgt_outer_jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0, 0, math.pi)
        self.rgt_outer_jlc.jnts[0].loc_motion_ax = np.array([1, 0, 0])
        self.rgt_outer_jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_outer_knuckle.stl"),
            cdmesh_type=self.cdmesh_type)
        self.rgt_outer_jlc.jnts[0].lnk.cmodel.rgba = rm.const.hug_gray
        # right finger outer (joint 1 / outer_finger)
        self.rgt_outer_jlc.jnts[1].loc_pos = np.array([0, .0315, -.0041])
        self.rgt_outer_jlc.jnts[1].loc_motion_ax = np.array([1, 0, 0])
        self.rgt_outer_jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_outer_finger.stl"),
            cdmesh_type=self.cdmesh_type)
        self.rgt_outer_jlc.jnts[1].lnk.cmodel.rgba = rm.const.dim_gray
        # right finger outer (joint 2 / inner_finger)
        self.rgt_outer_jlc.jnts[2].loc_pos = np.array([0, .0061, .0471])
        self.rgt_outer_jlc.jnts[2].loc_motion_ax = np.array([1, 0, 0])
        self.rgt_outer_jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_inner_finger.stl"),
            cdmesh_type=self.cdmesh_type)
        self.rgt_outer_jlc.jnts[2].lnk.cmodel.rgba = rm.const.dim_gray
        # right finger outer (joint 3 / inner_finger_pad)
        self.rgt_outer_jlc.jnts[3].loc_pos = np.zeros(3)
        self.rgt_outer_jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_pad.stl"), cdmesh_type=self.cdmesh_type)
        self.rgt_outer_jlc.jnts[3].lnk.cmodel.rgba = rm.const.hug_gray
        # right finger inner
        self.rgt_inner_jlc = rkjlc.JLChain(pos=self.palm.gl_flange_pose_list[3][0],
                                           rotmat=self.palm.gl_flange_pose_list[3][1],
                                           n_dof=1, name=name + "_right_inner")
        self.rgt_inner_jlc.jnts[0].loc_pos = np.zeros(3)
        self.rgt_inner_jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0, 0, math.pi)
        self.rgt_inner_jlc.jnts[0].loc_motion_ax = np.array([1, 0, 0])
        self.rgt_inner_jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_inner_knuckle.stl"),
            cdmesh_type=self.cdmesh_type)
        self.rgt_inner_jlc.jnts[0].lnk.cmodel.rgba = rm.const.dim_gray
        # finalize
        self.lft_outer_jlc.finalize()
        self.lft_inner_jlc.finalize()
        self.rgt_outer_jlc.finalize()
        self.rgt_inner_jlc.finalize()
        # acting center
        self.loc_acting_center_pos = np.array([0, 0, .15])
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
        self._rotmat = rotmat
        if jaw_width is not None:
            self.change_jaw_width(jaw_width=jaw_width)
        self.coupling.pos = self._pos
        self.coupling.rotmat = self._rotmat
        self.palm.pos = self.coupling.gl_flange_pose_list[0][0]
        self.palm.rotmat = self.coupling.gl_flange_pose_list[0][1]
        self.lft_outer_jlc.fix_to(self.palm.gl_flange_pose_list[0][0], self.palm.gl_flange_pose_list[0][1])
        self.lft_inner_jlc.fix_to(self.palm.gl_flange_pose_list[1][0], self.palm.gl_flange_pose_list[1][1])
        self.rgt_outer_jlc.fix_to(self.palm.gl_flange_pose_list[2][0], self.palm.gl_flange_pose_list[2][1])
        self.rgt_inner_jlc.fix_to(self.palm.gl_flange_pose_list[3][0], self.palm.gl_flange_pose_list[3][1])
        self.update_oiee()

    def change_jaw_width(self, jaw_width):
        if jaw_width > 0.085:
            raise ValueError("ee_values must be 0mm~85mm!")
        angle = math.asin((self.jaw_range[1] / 2.0 + .0064 - .0306011) / 0.055) - math.asin(
            (jaw_width / 2.0 + .0064 - .0306011) / 0.055)
        if angle < 0:
            angle = 0
        self.lft_outer_jlc.goto_given_conf(jnt_values=np.array([angle, 0.0, -angle, 0.0]))
        self.lft_inner_jlc.goto_given_conf(jnt_values=np.array([angle]))
        self.rgt_outer_jlc.goto_given_conf(jnt_values=np.array([angle, 0.0, -angle, 0.0]))
        self.rgt_inner_jlc.goto_given_conf(jnt_values=np.array([angle]))

    def get_jaw_width(self):
        angle = self.lft_inner_jlc.jnts[0].motion_value
        return (math.sin(
            math.asin((self.jaw_range[1] / 2.0 + .0064 - .0306011) / 0.055) - angle) * 0.055 - 0.0064 + 0.0306011) * 2.0

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

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    mcm.mgm.gen_frame().attach_to(base)
    gripper = Robotiq85()
    gripper.change_jaw_width(.025)
    model = gripper.gen_meshmodel(toggle_cdprim=True)
    model.attach_to(base)
    base.run()
