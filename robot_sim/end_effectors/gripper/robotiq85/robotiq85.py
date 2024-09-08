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


class Robotiq85(gi.GripperInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 cdmesh_type=mcm.mc.CDMType.DEFAULT,
                 name='robotiq85'):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        # flange
        self.coupling.loc_flange_pose_list[0] = [np.zeros(3), np.eye(3)]
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
        self.palm.lnk_list[0].cmodel.rgba = rm.bc.tab20_list[15]
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
        self.lft_outer_jlc.jnts[0].lnk.cmodel.rgba = rm.bc.tab20_list[14]
        # left finger outer (joint 1 / outer_finger)
        self.lft_outer_jlc.jnts[1].loc_pos = np.array([0, .0315, -.0041])
        self.lft_outer_jlc.jnts[1].loc_motion_ax = np.array([1, 0, 0])
        self.lft_outer_jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_outer_finger.stl"),
            cdmesh_type=self.cdmesh_type)
        self.lft_outer_jlc.jnts[1].lnk.cmodel.rgba = rm.bc.tab20_list[14]
        # left finger outer (joint 2 / inner_finger)
        self.lft_outer_jlc.jnts[2].loc_pos = np.array([0, .0061, .0471])
        self.lft_outer_jlc.jnts[2].loc_motion_ax = np.array([1, 0, 0])
        self.lft_outer_jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_inner_finger.stl"),
            cdmesh_type=self.cdmesh_type)
        self.lft_outer_jlc.jnts[2].lnk.cmodel.rgba = rm.bc.tab20_list[14]
        # left finger outer (joint 3 / inner_finger_pad)
        self.lft_outer_jlc.jnts[3].loc_pos = np.zeros(3)
        self.lft_outer_jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_pad.stl"), cdmesh_type=self.cdmesh_type)
        self.lft_outer_jlc.jnts[3].lnk.cmodel.rgba = rm.bc.tab20_list[14]
        # left finger inner
        self.lft_inner_jlc = rkjlc.JLChain(pos=self.palm.gl_flange_pose_list[1][0],
                                           rotmat=self.palm.gl_flange_pose_list[1][1],
                                           n_dof=1, name=name + "_left_inner")
        self.lft_inner_jlc.jnts[0].loc_pos = np.zeros(3)
        self.lft_inner_jlc.jnts[0].loc_motion_ax = np.array([1, 0, 0])
        self.lft_inner_jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_inner_knuckle.stl"),
            cdmesh_type=self.cdmesh_type)
        self.lft_inner_jlc.jnts[0].lnk.cmodel.rgba = rm.bc.tab20_list[14]
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
        self.rgt_outer_jlc.jnts[0].lnk.cmodel.rgba = rm.bc.tab20_list[14]
        # right finger outer (joint 1 / outer_finger)
        self.rgt_outer_jlc.jnts[1].loc_pos = np.array([0, .0315, -.0041])
        self.rgt_outer_jlc.jnts[1].loc_motion_ax = np.array([1, 0, 0])
        self.rgt_outer_jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_outer_finger.stl"),
            cdmesh_type=self.cdmesh_type)
        self.rgt_outer_jlc.jnts[1].lnk.cmodel.rgba = rm.bc.tab20_list[14]
        # right finger outer (joint 2 / inner_finger)
        self.rgt_outer_jlc.jnts[2].loc_pos = np.array([0, .0061, .0471])
        self.rgt_outer_jlc.jnts[2].loc_motion_ax = np.array([1, 0, 0])
        self.rgt_outer_jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_inner_finger.stl"),
            cdmesh_type=self.cdmesh_type)
        self.rgt_outer_jlc.jnts[2].lnk.cmodel.rgba = rm.bc.tab20_list[14]
        # right finger outer (joint 3 / inner_finger_pad)
        self.rgt_outer_jlc.jnts[3].loc_pos = np.zeros(3)
        self.rgt_outer_jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "robotiq_arg2f_85_pad.stl"), cdmesh_type=self.cdmesh_type)
        self.rgt_outer_jlc.jnts[3].lnk.cmodel.rgba = rm.bc.tab20_list[14]
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
        self.rgt_inner_jlc.jnts[0].lnk.cmodel.rgba = rm.bc.tab20_list[14]

        # # - rgt_outer
        # self.rgt_outer.lnks[1]['name'] = "left_outer_knuckle"
        # self.rgt_outer.lnks[1]['loc_pos'] = np.zeros(3)
        # self.rgt_outer.lnks[1]['com'] = np.array([-0.000200000000003065, 0.0199435877845359, 0.0292245259211331])
        # self.rgt_outer.lnks[1]['mass'] = 0.00853198276973456
        # self.rgt_outer.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_outer_knuckle.stl")
        # self.rgt_outer.lnks[1]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        # self.rgt_outer.lnks[2]['name'] = "left_outer_finger"
        # self.rgt_outer.lnks[2]['loc_pos'] = np.zeros(3)
        # self.rgt_outer.lnks[2]['com'] = np.array([0.00030115855001899, 0.0373907951953854, -0.0208027427000385])
        # self.rgt_outer.lnks[2]['mass'] = 0.022614240507152
        # self.rgt_outer.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_outer_finger_cvt.stl")
        # self.rgt_outer.lnks[2]['rgba'] = [.2, .2, .2, 1]
        # self.rgt_outer.lnks[3]['name'] = "left_inner_finger"
        # self.rgt_outer.lnks[3]['loc_pos'] = np.zeros(3)
        # self.rgt_outer.lnks[3]['com'] = np.array([0.000299999999999317, 0.0160078233491243, -0.0136945669206257])
        # self.rgt_outer.lnks[3]['mass'] = 0.0104003125914103
        # self.rgt_outer.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_inner_finger_cvt2.stl")
        # self.rgt_outer.lnks[3]['rgba'] = [.2, .2, .2, 1]
        # self.rgt_outer.lnks[4]['name'] = "left_inner_finger_pad"
        # self.rgt_outer.lnks[4]['loc_pos'] = np.zeros(3)
        # self.rgt_outer.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_pad.dae")
        # self.rgt_outer.lnks[4]['scale'] = [1e-3, 1e-3, 1e-3]
        # self.rgt_outer.lnks[4]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        # # - rgt_inner
        # self.rgt_inner.lnks[1]['name'] = "left_inner_knuckle"
        # self.rgt_inner.lnks[1]['loc_pos'] = np.zeros(3)
        # self.rgt_inner.lnks[1]['com'] = np.array([0.000123011831763771, 0.0507850843201817, 0.00103968640075166])
        # self.rgt_inner.lnks[1]['mass'] = 0.0271177346495152
        # self.rgt_inner.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_inner_knuckle_cvt.stl")
        # self.rgt_inner.lnks[1]['rgba'] = [.2, .2, .2, 1]

        # cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        # cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        #
        # # - rgt_outer
        # self.rgt_outer = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(4), name='rgt_outer')
        # self.rgt_outer.jnts[1]['loc_pos'] = np.array([0, .0306011, .054904])
        # self.rgt_outer.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        # self.rgt_outer.jnts[2]['loc_pos'] = np.array([0, .0315, -.0041])  # passive
        # self.rgt_outer.jnts[2]['loc_motionax'] = np.array([1, 0, 0])
        # self.rgt_outer.jnts[3]['loc_pos'] = np.array([0, .0061, .0471])
        # self.rgt_outer.jnts[3]['loc_motionax'] = np.array([1, 0, 0])
        # self.rgt_outer.jnts[4]['loc_pos'] = np.zeros(3)
        # # https://github.com/Danfoa uses geometry instead of the dae mesh. The following coordiante is needed
        # # self.rgt_outer.joints[4]['loc_pos'] = np.array([0, -0.0220203446692936, .03242])
        # # - rgt_inner
        # self.rgt_inner = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(1), name='rgt_inner')
        # self.rgt_inner.jnts[1]['loc_pos'] = np.array([0, .0127, .06142])
        # self.rgt_inner.jnts[1]['loc_motionax'] = np.array([1, 0, 0])

        # # - lft_outer
        # self.lft_outer = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(4), name='lft_outer')
        # self.lft_outer.jnts[1]['loc_pos'] = np.array([0, -.0306011, .054904])
        # self.lft_outer.jnts[1]['motion_range'] = [.0, .8]
        # self.lft_outer.jnts[1]['gl_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi)
        # self.lft_outer.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        # self.lft_outer.jnts[2]['loc_pos'] = np.array([0, .0315, -.0041])  # passive
        # self.lft_outer.jnts[2]['loc_motionax'] = np.array([1, 0, 0])
        # self.lft_outer.jnts[3]['loc_pos'] = np.array([0, .0061, .0471])
        # self.lft_outer.jnts[3]['loc_motionax'] = np.array([1, 0, 0])
        # self.lft_outer.jnts[4]['loc_pos'] = np.zeros(3)
        # # https://github.com/Danfoa uses geometry instead of the dae mesh. The following coordiante is needed
        # # self.lft_outer.joints[4]['loc_pos'] = np.array([0, -0.0220203446692936, .03242])
        # # - lft_inner
        # self.lft_inner = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(1), name='lft_inner')
        # self.lft_inner.jnts[1]['loc_pos'] = np.array([0, -.0127, .06142])
        # self.lft_inner.jnts[1]['gl_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi)
        # self.lft_inner.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        #
        # # links
        # # - lft_outer
        # self.lft_outer.lnks[0]['name'] = "robotiq85_gripper_base"
        # self.lft_outer.lnks[0]['loc_pos'] = np.zeros(3)
        # self.lft_outer.lnks[0]['com'] = np.array([8.625e-08, -4.6583e-06, 0.03145])
        # self.lft_outer.lnks[0]['mass'] = 0.22652
        # self.lft_outer.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_base_link_cvt.stl")
        # self.lft_outer.lnks[0]['rgba'] = [.2, .2, .2, 1]
        # self.lft_outer.lnks[1]['name'] = "left_outer_knuckle"
        # self.lft_outer.lnks[1]['loc_pos'] = np.zeros(3)
        # self.lft_outer.lnks[1]['com'] = np.array([-0.000200000000003065, 0.0199435877845359, 0.0292245259211331])
        # self.lft_outer.lnks[1]['mass'] = 0.00853198276973456
        # self.lft_outer.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_outer_knuckle.stl")
        # self.lft_outer.lnks[1]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        # self.lft_outer.lnks[2]['name'] = "left_outer_finger"
        # self.lft_outer.lnks[2]['loc_pos'] = np.zeros(3)
        # self.lft_outer.lnks[2]['com'] = np.array([0.00030115855001899, 0.0373907951953854, -0.0208027427000385])
        # self.lft_outer.lnks[2]['mass'] = 0.022614240507152
        # self.lft_outer.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_outer_finger_cvt.stl")
        # self.lft_outer.lnks[2]['rgba'] = [.2, .2, .2, 1]
        # self.lft_outer.lnks[3]['name'] = "left_inner_finger"
        # self.lft_outer.lnks[3]['loc_pos'] = np.zeros(3)
        # self.lft_outer.lnks[3]['com'] = np.array([0.000299999999999317, 0.0160078233491243, -0.0136945669206257])
        # self.lft_outer.lnks[3]['mass'] = 0.0104003125914103
        # self.lft_outer.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_inner_finger_cvt2.stl")
        # self.lft_outer.lnks[3]['rgba'] = [.2, .2, .2, 1]
        # self.lft_outer.lnks[4]['name'] = "left_inner_finger_pad"
        # self.lft_outer.lnks[4]['loc_pos'] = np.zeros(3)
        # self.lft_outer.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_pad.dae")
        # self.lft_outer.lnks[4]['scale'] = [1e-3, 1e-3, 1e-3]
        # self.lft_outer.lnks[4]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        # # - lft_inner
        # self.lft_inner.lnks[1]['name'] = "left_inner_knuckle"
        # self.lft_inner.lnks[1]['loc_pos'] = np.zeros(3)
        # self.lft_inner.lnks[1]['com'] = np.array([0.000123011831763771, 0.0507850843201817, 0.00103968640075166])
        # self.lft_inner.lnks[1]['mass'] = 0.0271177346495152
        # self.lft_inner.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_inner_knuckle_cvt.stl")
        # self.lft_inner.lnks[1]['rgba'] = [.2, .2, .2, 1]
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
                                self.rgt_outer_jlc.jnts[0].lnk,
                                self.rgt_outer_jlc.jnts[1].lnk)

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
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    gripper = Robotiq85()
    gripper.change_jaw_width(.025)
    model = gripper.gen_meshmodel(toggle_cdprim=True)
    model.attach_to(base)
    base.run()
