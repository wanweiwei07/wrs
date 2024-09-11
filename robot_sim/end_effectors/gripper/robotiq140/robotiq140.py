import os
import math
import numpy as np

import basis.robot_math as rm
import modeling.geometric_model as gm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.end_effectors.gripper.gripper_interface as gp

import os
import math
import numpy as np
import modeling.model_collection as mmc
import modeling.collision_model as mcm
import modeling.geometric_model as mgm
import robot_sim._kinematics.jlchain as jl
import basis.robot_math as rm
import robot_sim._kinematics.jlchain as rkjlc
import robot_sim.end_effectors.gripper.gripper_interface as gi


class Robotiq140(gi.GripperInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 coupling_offset_pos=np.zeros(3),
                 coupling_offset_rotmat=np.eye(3),
                 cdmesh_type=mcm.mc.CDMType.DEFAULT,
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
        self.lft_outer_jlc.jnts[3].loc_pos = np.array([0, 0.0420203446692936, -.03242])
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
        self.rgt_outer_jlc.jnts[1].loc_pos = np.array([0, 0.01821998610742, 0.0260018192872234])
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
        self.rgt_outer_jlc.jnts[3].loc_pos = np.array([0, 0.0420203446692936, -.03242])
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
        self.pos = pos
        self.rotmat = rotmat
        if jaw_width is not None:
            self.change_jaw_width(jaw_width=jaw_width)
        self.coupling.pos = self.pos
        self.coupling.rotmat = self.rotmat
        self.palm.pos = self.coupling.gl_flange_pose_list[0][0]
        self.palm.rotmat = self.coupling.gl_flange_pose_list[0][1]
        self.lft_outer_jlc.fix_to(self.palm.gl_flange_pose_list[0][0], self.palm.gl_flange_pose_list[0][1])
        self.lft_inner_jlc.fix_to(self.palm.gl_flange_pose_list[1][0], self.palm.gl_flange_pose_list[1][1])
        self.rgt_outer_jlc.fix_to(self.palm.gl_flange_pose_list[2][0], self.palm.gl_flange_pose_list[2][1])
        self.rgt_inner_jlc.fix_to(self.palm.gl_flange_pose_list[3][0], self.palm.gl_flange_pose_list[3][1])
        self.update_oiee()

    def change_jaw_width(self, jaw_width):
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
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    gripper = Robotiq140()
    gripper.change_jaw_width(.14)
    model = gripper.gen_meshmodel(toggle_cdprim=True)
    model.attach_to(base)
    base.run()

# class Robotiq140(gp.GripperInterface):
#     """
#     author: kiyokawa, revised by weiwei
#     date: 2020212
#     """
#
#     def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='box', name='robotiq140', enable_cc=True):
#
#         super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
#         this_dir, this_filename = os.path.split(__file__)
#         cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
#         cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
#         # - lft_outer
#         self.lft_outer = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(4), name='lft_outer')
#         self.lft_outer.jnts[1]['loc_pos'] = np.array([0, -.0306011, .054904])
#         self.lft_outer.jnts[1]['motion_range'] = [.0, .7]
#         self.lft_outer.jnts[1]['gl_rotmat'] = rm.rotmat_from_euler((math.pi / 2.0 + .725), 0, 0)
#         self.lft_outer.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
#         self.lft_outer.jnts[2]['loc_pos'] = np.array([0, 0.01821998610742, 0.0260018192872234])  # passive
#         self.lft_outer.jnts[2]['loc_motionax'] = np.array([1, 0, 0])
#         self.lft_outer.jnts[3]['loc_pos'] = np.array([0, 0.0817554015893473, -0.0282203446692936])
#         self.lft_outer.jnts[3]['gl_rotmat'] = rm.rotmat_from_euler(-0.725, 0, 0)
#         self.lft_outer.jnts[3]['loc_motionax'] = np.array([1, 0, 0])
#         self.lft_outer.jnts[4]['loc_pos'] = np.array([0, 0.0420203446692936, -.03242])
#         # - lft_inner
#         self.lft_inner = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(1), name='lft_inner')
#         self.lft_inner.jnts[1]['loc_pos'] = np.array([0, -.0127, .06142])
#         self.lft_inner.jnts[1]['gl_rotmat'] = rm.rotmat_from_euler((math.pi / 2.0 + .725), 0, 0)
#         self.lft_inner.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
#         # - rgt_outer
#         self.rgt_outer = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(4), name='rgt_outer')
#         self.rgt_outer.jnts[1]['loc_pos'] = np.array([0, .0306011, .054904])
#         self.rgt_outer.jnts[1]['gl_rotmat'] = rm.rotmat_from_euler((math.pi / 2.0 + .725), 0, math.pi)
#         self.rgt_outer.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
#         self.rgt_outer.jnts[2]['loc_pos'] = np.array([0, 0.01821998610742, 0.0260018192872234])  # passive
#         self.rgt_outer.jnts[2]['loc_motionax'] = np.array([1, 0, 0])
#         self.rgt_outer.jnts[3]['loc_pos'] = np.array([0, 0.0817554015893473, -0.0282203446692936])
#         self.rgt_outer.jnts[3]['gl_rotmat'] = rm.rotmat_from_euler(-0.725, 0, 0)
#         self.rgt_outer.jnts[3]['loc_motionax'] = np.array([1, 0, 0])
#         self.rgt_outer.jnts[4]['loc_pos'] = np.array([0, 0.0420203446692936, -.03242])
#         # - rgt_inner
#         self.rgt_inner = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(1), name='rgt_inner')
#         self.rgt_inner.jnts[1]['loc_pos'] = np.array([0, 0.0127, 0.06142])
#         self.rgt_inner.jnts[1]['gl_rotmat'] = rm.rotmat_from_euler((math.pi / 2.0 + .725), 0, math.pi)
#         self.rgt_inner.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
#         # links
#         # - lft_outer
#         self.lft_outer.lnks[0]['name'] = "robotiq140_gripper_base"
#         self.lft_outer.lnks[0]['loc_pos'] = np.zeros(3)
#         self.lft_outer.lnks[0]['com'] = np.array([8.625e-08, -4.6583e-06, 0.03145])
#         self.lft_outer.lnks[0]['mass'] = 0.22652
#         self.lft_outer.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_base_link.stl")
#         self.lft_outer.lnks[0]['rgba'] = [.2, .2, .2, 1]
#         self.lft_outer.lnks[1]['name'] = "left_outer_knuckle"
#         self.lft_outer.lnks[1]['loc_pos'] = np.zeros(3)
#         self.lft_outer.lnks[1]['com'] = np.array([-0.000200000000003065, 0.0199435877845359, 0.0292245259211331])
#         self.lft_outer.lnks[1]['mass'] = 0.00853198276973456
#         self.lft_outer.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_outer_knuckle.stl")
#         self.lft_outer.lnks[1]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
#         self.lft_outer.lnks[2]['name'] = "left_outer_finger"
#         self.lft_outer.lnks[2]['loc_pos'] = np.zeros(3)
#         self.lft_outer.lnks[2]['com'] = np.array([0.00030115855001899, 0.0373907951953854, -0.0208027427000385])
#         self.lft_outer.lnks[2]['mass'] = 0.022614240507152
#         self.lft_outer.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_outer_finger.stl")
#         self.lft_outer.lnks[2]['rgba'] = [.2, .2, .2, 1]
#         self.lft_outer.lnks[3]['name'] = "left_inner_finger"
#         self.lft_outer.lnks[3]['loc_pos'] = np.zeros(3)
#         self.lft_outer.lnks[3]['com'] = np.array([0.000299999999999317, 0.0160078233491243, -0.0136945669206257])
#         self.lft_outer.lnks[3]['mass'] = 0.0104003125914103
#         self.lft_outer.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_inner_finger.stl")
#         self.lft_outer.lnks[3]['rgba'] = [.2, .2, .2, 1]
#         self.lft_outer.lnks[4]['name'] = "left_inner_finger_pad"
#         self.lft_outer.lnks[4]['loc_pos'] = np.zeros(3)
#         self.lft_outer.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_pad.stl")
#         self.lft_outer.lnks[4]['scale'] = [1e-3, 1e-3, 1e-3]
#         self.lft_outer.lnks[4]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
#         # - lft_inner
#         self.lft_inner.lnks[1]['name'] = "left_inner_knuckle"
#         self.lft_inner.lnks[1]['loc_pos'] = np.zeros(3)
#         self.lft_inner.lnks[1]['com'] = np.array([0.000123011831763771, 0.0507850843201817, 0.00103968640075166])
#         self.lft_inner.lnks[1]['mass'] = 0.0271177346495152
#         self.lft_inner.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_inner_knuckle.stl")
#         self.lft_inner.lnks[1]['rgba'] = [.2, .2, .2, 1]
#         # - rgt_outer
#         self.rgt_outer.lnks[1]['name'] = "right_outer_knuckle"
#         self.rgt_outer.lnks[1]['loc_pos'] = np.zeros(3)
#         self.rgt_outer.lnks[1]['com'] = np.array([-0.000200000000003065, 0.0199435877845359, 0.0292245259211331])
#         self.rgt_outer.lnks[1]['mass'] = 0.00853198276973456
#         self.rgt_outer.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_outer_knuckle.stl")
#         self.rgt_outer.lnks[1]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
#         self.rgt_outer.lnks[2]['name'] = "right_outer_finger"
#         self.rgt_outer.lnks[2]['loc_pos'] = np.zeros(3)
#         self.rgt_outer.lnks[2]['com'] = np.array([0.00030115855001899, 0.0373907951953854, -0.0208027427000385])
#         self.rgt_outer.lnks[2]['mass'] = 0.022614240507152
#         self.rgt_outer.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_outer_finger.stl")
#         self.rgt_outer.lnks[2]['rgba'] = [.2, .2, .2, 1]
#         self.rgt_outer.lnks[3]['name'] = "right_inner_finger"
#         self.rgt_outer.lnks[3]['loc_pos'] = np.zeros(3)
#         self.rgt_outer.lnks[3]['com'] = np.array([0.000299999999999317, 0.0160078233491243, -0.0136945669206257])
#         self.rgt_outer.lnks[3]['mass'] = 0.0104003125914103
#         self.rgt_outer.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_inner_finger.stl")
#         self.rgt_outer.lnks[3]['rgba'] = [.2, .2, .2, 1]
#         self.rgt_outer.lnks[4]['name'] = "right_inner_finger_pad"
#         self.rgt_outer.lnks[4]['loc_pos'] = np.zeros(3)
#         self.rgt_outer.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_pad.stl")
#         self.rgt_outer.lnks[4]['scale'] = [1e-3, 1e-3, 1e-3]
#         self.rgt_outer.lnks[4]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
#         # - rgt_inner
#         self.rgt_inner.lnks[1]['name'] = "right_inner_knuckle"
#         self.rgt_inner.lnks[1]['loc_pos'] = np.zeros(3)
#         self.rgt_inner.lnks[1]['com'] = np.array([0.000123011831763771, 0.0507850843201817, 0.00103968640075166])
#         self.rgt_inner.lnks[1]['mass'] = 0.0271177346495152
#         self.rgt_inner.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_inner_knuckle.stl")
#         self.rgt_inner.lnks[1]['rgba'] = [.2, .2, .2, 1]
#         # reinitialize
#         self.lft_outer.finalize()
#         self.lft_inner.finalize()
#         self.rgt_outer.finalize()
#         self.rgt_inner.finalize()
#         # jaw range
#         self.jaw_range = [0.0, .140]
#         # jaw center
#         self.jaw_center_pos = np.array([0, 0, .19])  # position for initial state (fully open)
#         # relative jaw center pos
#         self.jaw_center_pos_rel = self.jaw_center_pos - self.lft_outer.jnts[4]['pos']
#         # collision detection
#         self.all_cdelements = []
#         self.enable_cc(toggle_cdprimit=enable_cc)
#
#     def enable_cc(self, toggle_cdprimit):
#         if toggle_cdprimit:
#             super().enable_cc()
#             # cdprimit
#             self.cc.add_cdlnks(self.lft_outer, [0, 1, 2, 3, 4])
#             self.cc.add_cdlnks(self.lft_inner, [1])
#             self.cc.add_cdlnks(self.rgt_outer, [1, 2, 3, 4])
#             self.cc.add_cdlnks(self.rgt_inner, [1])
#             activelist = [self.lft_outer.lnks[0],
#                           self.lft_outer.lnks[1],
#                           self.lft_outer.lnks[2],
#                           self.lft_outer.lnks[3],
#                           self.lft_outer.lnks[4],
#                           self.lft_inner.lnks[1],
#                           self.rgt_outer.lnks[1],
#                           self.rgt_outer.lnks[2],
#                           self.rgt_outer.lnks[3],
#                           self.rgt_outer.lnks[4],
#                           self.rgt_inner.lnks[1]]
#             self.cc.set_active_cdlnks(activelist)
#             self.all_cdelements = self.cc.cce_dict
#         # cdmesh
#         for cdelement in self.all_cdelements:
#             cdmesh = cdelement['collision_model'].copy()
#             self.cdmesh_collection.add_cm(cdmesh)
#
#     def fix_to(self, pos, rotmat, angle=None):
#         self.pos = pos
#         self.rotmat = rotmat
#         if angle is not None:
#             self.lft_outer.jnts[1]['motion_value'] = angle
#             self.lft_outer.jnts[3]['motion_value'] = self.lft_outer.jnts[1]['motion_value']
#             self.lft_inner.jnts[1]['motion_value'] = -self.lft_outer.jnts[1]['motion_value']
#             self.rgt_outer.jnts[1]['motion_value'] = -self.lft_outer.jnts[1]['motion_value']
#             self.rgt_outer.jnts[3]['motion_value'] = self.lft_outer.jnts[1]['motion_value']
#             self.rgt_inner.jnts[1]['motion_value'] = -self.lft_outer.jnts[1]['motion_value']
#         self.coupling.fix_to(self.pos, self.rotmat)
#         cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
#         cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
#         self.lft_outer.fix_to(cpl_end_pos, cpl_end_rotmat)
#         self.lft_inner.fix_to(cpl_end_pos, cpl_end_rotmat)
#         self.rgt_inner.fix_to(cpl_end_pos, cpl_end_rotmat)
#         self.rgt_outer.fix_to(cpl_end_pos, cpl_end_rotmat)
#
#     def fk(self, motion_val):
#         """
#         lft_outer is the only active joint, all others mimic this one
#         :param: angle, radian
#         """
#         if self.lft_outer.jnts[1]['motion_range'][0] <= motion_val <= self.lft_outer.jnts[1]['motion_range'][1]:
#             self.lft_outer.jnts[1]['motion_value'] = motion_val
#             self.lft_outer.jnts[3]['motion_value'] = self.lft_outer.jnts[1]['motion_value']
#             self.lft_inner.jnts[1]['motion_value'] = -self.lft_outer.jnts[1]['motion_value']
#             self.rgt_outer.jnts[1]['motion_value'] = -self.lft_outer.jnts[1]['motion_value']
#             self.rgt_outer.jnts[3]['motion_value'] = self.lft_outer.jnts[1]['motion_value']
#             self.rgt_inner.jnts[1]['motion_value'] = -self.lft_outer.jnts[1]['motion_value']
#             self.lft_outer.fk()
#             self.lft_inner.fk()
#             self.rgt_outer.fk()
#             self.rgt_inner.fk()
#         else:
#             raise ValueError("The angle parameter is out of range!")
#
#     def _from_distance_to_radians(self, distance):
#         """
#         private helper function to convert a command in meters to radians (joint value)
#         """
#         # return np.clip(
#         #   self.lft_outer.joints[1]['motion_range'][1] - ((self.lft_outer.joints[1]['motion_range'][1]/self.jaw_range[1]) * linear_distance),
#         #   self.lft_outer.joints[1]['motion_range'][0], self.lft_outer.joints[1]['motion_range'][1]) # kiyokawa, commented out by weiwei
#         return np.clip(self.lft_outer.jnts[1]['motion_range'][1] - math.asin(
#             (math.sin(self.lft_outer.jnts[1]['motion_range'][1]) / self.jaw_range[1]) * distance),
#                        self.lft_outer.jnts[1]['motion_range'][0], self.lft_outer.jnts[1]['motion_range'][1])
#
#     def change_jaw_width(self, jaw_width):
#         if jaw_width > self.jaw_range[1]:
#             raise ValueError(f"Jawwidth must be {self.jaw_range[0]}mm~{self.jaw_range[1]}mm!")
#         motion_val = self._from_distance_to_radians(jaw_width)
#         self.fk(motion_val)
#         # TODO dynamically change jaw center
#         # print(self.jaw_center_pos_rel)
#         self.jaw_center_pos=np.array([0, 0, self.lft_outer.jnts[4]['gl_posq'][2] + self.jaw_center_pos_rel[2]])
#
#     def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='ee_stickmodel'):
#         sm_collection = mc.ModelCollection(name=name)
#         self.coupling.gen_stickmodel(toggle_tcp_frame=False,
#                                      toggle_jnt_frames=toggle_jnt_frames).attach_to(sm_collection)
#         self.lft_outer.gen_stickmodel(toggle_tcpcs=False,
#                                       toggle_jntscs=toggle_jnt_frames,
#                                       toggle_connjnt=toggle_connjnt).attach_to(sm_collection)
#         self.lft_inner.gen_stickmodel(toggle_tcpcs=False,
#                                       toggle_jntscs=toggle_jnt_frames,
#                                       toggle_connjnt=toggle_connjnt).attach_to(sm_collection)
#         self.rgt_outer.gen_stickmodel(toggle_tcpcs=False,
#                                       toggle_jntscs=toggle_jnt_frames,
#                                       toggle_connjnt=toggle_connjnt).attach_to(sm_collection)
#         self.rgt_inner.gen_stickmodel(toggle_tcpcs=False,
#                                       toggle_jntscs=toggle_jnt_frames,
#                                       toggle_connjnt=toggle_connjnt).attach_to(sm_collection)
#         if toggle_tcp_frame:
#             jaw_center_gl_pos = self.rotmat.dot(grpr.jaw_center_pos) + self.pos
#             jaw_center_gl_rotmat = self.rotmat.dot(grpr.loc_acting_center_rotmat)
#             gm.gen_dashed_stick(spos=self.pos,
#                                 epos=jaw_center_gl_pos,
#                                 radius=.0062,
#                                 rgba=[.5, 0, 1, 1],
#                                 type="round").attach_to(sm_collection)
#             gm.gen_myc_frame(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(sm_collection)
#         return sm_collection
#
#     def gen_meshmodel(self,
#                       toggle_tcp_frame=False,
#                       toggle_jnt_frames=False,
#                       rgba=None,
#                       name='robotiq140_meshmodel'):
#         mm_collection = mc.ModelCollection(name=name)
#         self.coupling.gen_mesh_model(toggle_tcpcs=False,
#                                      toggle_jntscs=toggle_jnt_frames,
#                                      rgba=rgba).attach_to(mm_collection)
#         self.lft_outer.gen_mesh_model(toggle_tcpcs=False,
#                                       toggle_jntscs=toggle_jnt_frames,
#                                       rgba=rgba).attach_to(mm_collection)
#         self.lft_inner.gen_mesh_model(toggle_tcpcs=False,
#                                       toggle_jntscs=toggle_jnt_frames,
#                                       rgba=rgba).attach_to(mm_collection)
#         self.rgt_outer.gen_mesh_model(toggle_tcpcs=False,
#                                       toggle_jntscs=toggle_jnt_frames,
#                                       rgba=rgba).attach_to(mm_collection)
#         self.rgt_inner.gen_mesh_model(toggle_tcpcs=False,
#                                       toggle_jntscs=toggle_jnt_frames,
#                                       rgba=rgba).attach_to(mm_collection)
#         if toggle_tcp_frame:
#             jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
#             jaw_center_gl_rotmat = self.rotmat.dot(self.loc_acting_center_rotmat)
#             gm.gen_dashed_stick(spos=self.pos,
#                                 epos=jaw_center_gl_pos,
#                                 radius=.0062,
#                                 rgba=[.5, 0, 1, 1],
#                                 type="round").attach_to(mm_collection)
#             gm.gen_myc_frame(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(mm_collection)
#         return mm_collection
#
#
# if __name__ == '__main__':
#     import visualization.panda.world as wd
#
#     base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
#     gm.gen_frame().attach_to(base)
#     grpr = Robotiq140(enable_cc=True)
#     # grpr.cdmesh_type='convexhull'
#     grpr.change_jaw_width(.1)
#     grpr.gen_meshmodel(toggle_tcp_frame=True, rgba=[.3, .3, .0, .5], toggle_jnt_frames=True).attach_to(base)
#     base.run()
