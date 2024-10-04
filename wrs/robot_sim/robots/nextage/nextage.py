import os
import math
import numpy as np
from wrs import basis as rm, basis, robot_sim as mi, robot_sim as jl, modeling as cm, modeling as gm


class Left_Manipulator(mi.ManipulatorInterface):
    """
    the left manipulator is a manipulator basis for a waist + single arm robot
    author: weiwei
    date: 20230809
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(7), name='nextage_left_manipulator'):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, home_conf=homeconf, name=name)
        # seven joints, n_jnts = 7+2 (tgt ranges from 1-7), nlinks = 7+1
        self.jlc.jnts[1]['loc_pos'] = np.array([0, 0, 0])
        self.jlc.jnts[1]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[2]['loc_pos'] = np.array([0, 0.145, 0.370296])
        self.jlc.jnts[2]['gl_rotmat'] = rm.rotmat_from_euler(-0.261799, 0, 0)
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[2]['motion_range'] = [-1.53589, 1.53589]
        self.jlc.jnts[3]['loc_pos'] = np.array([0, 0, 0])
        self.jlc.jnts[3]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.jnts[3]['motion_range'] = [-2.44346, 1.0472]
        self.jlc.jnts[4]['loc_pos'] = np.array([0, 0.095, -0.25])
        self.jlc.jnts[4]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.jnts[4]['motion_range'] = [-2.75762, 0]
        self.jlc.jnts[5]['loc_pos'] = np.array([-0.03, 0, 0])
        self.jlc.jnts[5]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[5]['motion_range'] = [-1.8326, 2.87979]
        self.jlc.jnts[6]['loc_pos'] = np.array([0, 0, -0.235])
        self.jlc.jnts[6]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.jnts[6]['motion_range'] = [-1.74533, 1.74533]
        self.jlc.jnts[7]['loc_pos'] = np.array([-0.047, 0, -0.09])
        self.jlc.jnts[7]['loc_motionax'] = np.array([1, 0, 0])
        self.jlc.jnts[7]['motion_range'] = [-2.84489, 2.84489]
        self.jlc.lnks[2]['name'] = "jlc_joint0"
        self.jlc.lnks[2]['loc_pos'] = np.array([0, 0.145, 0.370296])
        self.jlc.lnks[2]['gl_rotmat'] = rm.rotmat_from_euler(-0.261799, 0, 0)
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint0_link_mesh.dae")
        self.jlc.lnks[2]['rgba'] = [.35, .35, .35, 1]
        self.jlc.lnks[3]['name'] = "jlc_joint1"
        self.jlc.lnks[3]['loc_pos'] = np.array([0, 0, 0])
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint1_link_mesh.dae")
        self.jlc.lnks[3]['rgba'] = [.57, .57, .57, 1]
        self.jlc.lnks[4]['name'] = "jlc_joint2"
        self.jlc.lnks[4]['loc_pos'] = np.array([0, 0.095, -0.25])
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint2_link_mesh.dae")
        self.jlc.lnks[4]['rgba'] = [.35, .35, .35, 1]
        self.jlc.lnks[5]['name'] = "jlc_joint3"
        self.jlc.lnks[5]['loc_pos'] = np.array([-0.03, 0, 0])
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint3_link_mesh.dae")
        self.jlc.lnks[5]['rgba'] = [.35, .35, .35, 1]
        self.jlc.lnks[6]['name'] = "jlc_joint4"
        self.jlc.lnks[6]['loc_pos'] = np.array([0, 0, -0.235])
        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint4_link_mesh.dae")
        self.jlc.lnks[6]['rgba'] = [.7, .7, .7, 1]
        self.jlc.lnks[7]['name'] = "jlc_joint5"
        self.jlc.lnks[7]['loc_pos'] = np.array([-0.047, 0, -0.09])
        self.jlc.lnks[7]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint5_link_mesh.dae")
        self.jlc.lnks[7]['rgba'] = [.57, .57, .57, 1]
        self.jlc.finalize()
        self.toggle_waist(token=False)

    def toggle_waist(self, token=True):
        if token:
            self.tgt_jnts = range(1, 7)
        else:
            self.tgt_jnts = range(2, 7)

    def ik(self,
           tgt_pos,
           tgt_rotmat,
           seed_jnt_values=None,
           max_niter=100,
           tcp_jnt_id=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           local_minima="accept",
           toggle_dbg=False):
        self.jlc.ik(tgt_pos,
                    tgt_rotmat,
                    seed_jnt_values=seed_jnt_values,
                    tcp_joint_id=tcp_jnt_id,
                    tcp_loc_pos=tcp_loc_pos,
                    tcp_loc_rotmat=tcp_loc_rotmat,
                    max_niter=max_niter,
                    local_minima=local_minima,
                    toggle_debug=toggle_dbg)
#
#
# class Body_Manipulator(mi.ManipulatorInterface):
#
#
# class Nextage_WSA(ai.SglArmRobotInterface):
#     """
#     7 DoF half robot, WSA = waist + single arm
#     author: weiwei
#     date: 20230809
#     """
#
#     def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='nextage_waist_arm', enable_cc=False):
#         super().__init__(pos=pos, rotmat=rotmat, name=name)
#         self.manipulator = rkjlc.JLChain(pos=self.central_body.joints[1]['gl_posq'],
#                                       rotmat=self.central_body.joints[1]['gl_rotmatq'],
#                                       home=lft_arm_homeconf, name='lft_arm')
#
#
# class Nextage(ri.SglArmRobotInterface):
#
#     def _decorator_switch_tgt_jnts(foo):
#         """
#         decorator function for switching tgt_jnts
#         :return:
#         author: weiwei
#         date: 20220404
#         """
#
#         def wrapper(self, component_name, *args, **kwargs):
#
#             if component_name == 'lft_arm' or component_name == 'rgt_arm':
#                 old_tgt_jnts = self.manipulator_dict[component_name].tgtjnts
#                 self.manipulator_dict[component_name].tgtjnts = range(2, self.manipulator_dict[component_name].n_dof + 1)
#                 result = foo(self, component_name, *args, **kwargs)
#                 self.manipulator_dict[component_name].tgtjnts = old_tgt_jnts
#                 return result
#             else:
#                 return foo(self, component_name, *args, **kwargs)
#
#         return wrapper
#
#     def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='nextage', enable_cc=True):
#         super().__init__(pos=pos, rotmat=rotmat, name=name)
#         this_dir, this_filename = os.path.split(__file__)
#         central_homeconf = np.radians(np.array([.0, .0, .0]))
#         lft_arm_homeconf = np.radians(np.array([central_homeconf[0], 15, 0, -143, 0, 0, 0]))
#         rgt_arm_homeconf = np.radians(np.array([central_homeconf[0], -15, 0, -143, 0, 0, 0]))
#         # central
#         self.central_body = rkjlc.JLChain(pos=pos, rotmat=rotmat, home=central_homeconf, name='centeral_body')
#         self.central_body.joints[1]['loc_pos'] = np.array([0, 0, 0])
#         self.central_body.joints[1]['loc_motionax'] = np.array([0, 0, 1])
#         self.central_body.joints[1]['motion_range'] = [-2.84489, 2.84489]
#         self.central_body.joints[2]['loc_pos'] = np.array([0, 0, 0.5695])
#         self.central_body.joints[2]['loc_motionax'] = np.array([0, 0, 1])
#         self.central_body.joints[2]['motion_range'] = [-1.22173, 1.22173]
#         self.central_body.joints[3]['loc_pos'] = np.array([0, 0, 0])
#         self.central_body.joints[3]['loc_motionax'] = np.array([0, 1, 0])
#         self.central_body.joints[3]['motion_range'] = [-0.349066, 1.22173]
#         self.central_body.lnks[0]['name'] = "nextage_base"
#         self.central_body.lnks[0]['loc_pos'] = np.array([0, 0, 0.97])
#         self.central_body.lnks[0]['collision_model'] = mcm.CollisionModel(
#             os.path.join(this_dir, "meshes", "waist_link_mesh.dae"),
#             cdprim_type="user_defined", ex_radius=.005,
#             userdef_cdprim_fn=self._waist_combined_cdnp)
#         self.central_body.lnks[0]['rgba'] = [.77, .77, .77, 1.0]
#         self.central_body.lnks[1]['name'] = "nextage_chest"
#         self.central_body.lnks[1]['loc_pos'] = np.array([0, 0, 0])
#         self.central_body.lnks[1]['collision_model'] = mcm.CollisionModel(
#             os.path.join(this_dir, "meshes", "chest_joint0_link_mesh.dae"),
#             cdprim_type="user_defined", ex_radius=.005,
#             userdef_cdprim_fn=self._chest_combined_cdnp)
#         self.central_body.lnks[1]['rgba'] = [1, .65, .5, 1]
#         self.central_body.lnks[2]['name'] = "head_joint0_link_mesh"
#         self.central_body.lnks[2]['loc_pos'] = np.array([0, 0, 0.5695])
#         self.central_body.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "head_joint0_link_mesh.dae")
#         self.central_body.lnks[2]['rgba'] = [.35, .35, .35, 1]
#         self.central_body.lnks[3]['name'] = "nextage_chest"
#         self.central_body.lnks[3]['loc_pos'] = np.array([0, 0, 0])
#         self.central_body.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "head_joint1_link_mesh.dae")
#         self.central_body.lnks[3]['rgba'] = [.63, .63, .63, 1]
#         self.central_body.reinitialize()
#         # lft
#         self.lft_arm = rkjlc.JLChain(pos=self.central_body.joints[1]['gl_posq'],
#                                   rotmat=self.central_body.joints[1]['gl_rotmatq'],
#                                   home=lft_arm_homeconf, name='lft_arm')
#         self.lft_arm.joints[1]['loc_pos'] = np.array([0, 0, 0])
#         self.lft_arm.joints[1]['loc_motionax'] = np.array([0, 0, 1])
#         self.lft_arm.joints[2]['loc_pos'] = np.array([0, 0.145, 0.370296])
#         self.lft_arm.joints[2]['gl_rotmat'] = rm.rotmat_from_euler(-0.261799, 0, 0)
#         self.lft_arm.joints[2]['loc_motionax'] = np.array([0, 0, 1])
#         self.lft_arm.joints[2]['motion_range'] = [-1.53589, 1.53589]
#         self.lft_arm.joints[3]['loc_pos'] = np.array([0, 0, 0])
#         self.lft_arm.joints[3]['loc_motionax'] = np.array([0, 1, 0])
#         self.lft_arm.joints[3]['motion_range'] = [-2.44346, 1.0472]
#         self.lft_arm.joints[4]['loc_pos'] = np.array([0, 0.095, -0.25])
#         self.lft_arm.joints[4]['loc_motionax'] = np.array([0, 1, 0])
#         self.lft_arm.joints[4]['motion_range'] = [-2.75762, 0]
#         self.lft_arm.joints[5]['loc_pos'] = np.array([-0.03, 0, 0])
#         self.lft_arm.joints[5]['loc_motionax'] = np.array([0, 0, 1])
#         self.lft_arm.joints[5]['motion_range'] = [-1.8326, 2.87979]
#         self.lft_arm.joints[6]['loc_pos'] = np.array([0, 0, -0.235])
#         self.lft_arm.joints[6]['loc_motionax'] = np.array([0, 1, 0])
#         self.lft_arm.joints[6]['motion_range'] = [-1.74533, 1.74533]
#         self.lft_arm.joints[7]['loc_pos'] = np.array([-0.047, 0, -0.09])
#         self.lft_arm.joints[7]['loc_motionax'] = np.array([1, 0, 0])
#         self.lft_arm.joints[7]['motion_range'] = [-2.84489, 2.84489]
#         self.lft_arm.lnks[2]['name'] = "lft_arm_joint0"
#         self.lft_arm.lnks[2]['loc_pos'] = np.array([0, 0.145, 0.370296])
#         self.lft_arm.lnks[2]['gl_rotmat'] = rm.rotmat_from_euler(-0.261799, 0, 0)
#         self.lft_arm.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint0_link_mesh.dae")
#         self.lft_arm.lnks[2]['rgba'] = [.35, .35, .35, 1]
#         self.lft_arm.lnks[3]['name'] = "lft_arm_joint1"
#         self.lft_arm.lnks[3]['loc_pos'] = np.array([0, 0, 0])
#         self.lft_arm.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint1_link_mesh.dae")
#         self.lft_arm.lnks[3]['rgba'] = [.57, .57, .57, 1]
#         self.lft_arm.lnks[4]['name'] = "lft_arm_joint2"
#         self.lft_arm.lnks[4]['loc_pos'] = np.array([0, 0.095, -0.25])
#         self.lft_arm.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint2_link_mesh.dae")
#         self.lft_arm.lnks[4]['rgba'] = [.35, .35, .35, 1]
#         self.lft_arm.lnks[5]['name'] = "lft_arm_joint3"
#         self.lft_arm.lnks[5]['loc_pos'] = np.array([-0.03, 0, 0])
#         self.lft_arm.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint3_link_mesh.dae")
#         self.lft_arm.lnks[5]['rgba'] = [.35, .35, .35, 1]
#         self.lft_arm.lnks[6]['name'] = "lft_arm_joint4"
#         self.lft_arm.lnks[6]['loc_pos'] = np.array([0, 0, -0.235])
#         self.lft_arm.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint4_link_mesh.dae")
#         self.lft_arm.lnks[6]['rgba'] = [.7, .7, .7, 1]
#         self.lft_arm.lnks[7]['name'] = "lft_arm_joint5"
#         self.lft_arm.lnks[7]['loc_pos'] = np.array([-0.047, 0, -0.09])
#         self.lft_arm.lnks[7]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint5_link_mesh.dae")
#         self.lft_arm.lnks[7]['rgba'] = [.57, .57, .57, 1]
#         self.lft_arm.reinitialize()
#         # rgt
#         self.rgt_arm = rkjlc.JLChain(pos=self.central_body.joints[1]['gl_posq'],
#                                   rotmat=self.central_body.joints[1]['gl_rotmatq'],
#                                   home=rgt_arm_homeconf, name='rgt_arm')
#         self.rgt_arm.joints[1]['loc_pos'] = np.array([0, 0, 0])
#         self.rgt_arm.joints[1]['loc_motionax'] = np.array([0, 0, 1])
#         self.rgt_arm.joints[2]['loc_pos'] = np.array([0, -0.145, 0.370296])
#         self.rgt_arm.joints[2]['gl_rotmat'] = rm.rotmat_from_euler(0.261799, 0, 0)
#         self.rgt_arm.joints[2]['loc_motionax'] = np.array([0, 0, 1])
#         self.rgt_arm.joints[2]['motion_range'] = [-1.53589, 1.53589]
#         self.rgt_arm.joints[3]['loc_pos'] = np.array([0, 0, 0])
#         self.rgt_arm.joints[3]['loc_motionax'] = np.array([0, 1, 0])
#         self.rgt_arm.joints[3]['motion_range'] = [-2.44346, 1.0472]
#         self.rgt_arm.joints[4]['loc_pos'] = np.array([0, -0.095, -0.25])
#         self.rgt_arm.joints[4]['loc_motionax'] = np.array([0, 1, 0])
#         self.rgt_arm.joints[4]['motion_range'] = [-2.75762, 0]
#         self.rgt_arm.joints[5]['loc_pos'] = np.array([-0.03, 0, 0])
#         self.rgt_arm.joints[5]['loc_motionax'] = np.array([0, 0, 1])
#         self.rgt_arm.joints[5]['motion_range'] = [-1.8326, 2.87979]
#         self.rgt_arm.joints[6]['loc_pos'] = np.array([0, 0, -0.235])
#         self.rgt_arm.joints[6]['loc_motionax'] = np.array([0, 1, 0])
#         self.rgt_arm.joints[6]['motion_range'] = [-1.74533, 1.74533]
#         self.rgt_arm.joints[7]['loc_pos'] = np.array([-0.047, 0, -0.09])
#         self.rgt_arm.joints[7]['loc_motionax'] = np.array([1, 0, 0])
#         self.rgt_arm.joints[7]['motion_range'] = [-2.84489, 2.84489]
#         self.rgt_arm.lnks[2]['name'] = "rgt_arm_joint0"
#         self.rgt_arm.lnks[2]['loc_pos'] = np.array([0, -0.145, 0.370296])
#         self.rgt_arm.lnks[2]['gl_rotmat'] = rm.rotmat_from_euler(0.261799, 0, 0)
#         self.rgt_arm.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "rarm_joint0_link_mesh.dae")
#         self.rgt_arm.lnks[2]['rgba'] = [.35, .35, .35, 1]
#         self.rgt_arm.lnks[3]['name'] = "rgt_arm_joint1"
#         self.rgt_arm.lnks[3]['loc_pos'] = np.array([0, 0, 0])
#         self.rgt_arm.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "rarm_joint1_link_mesh.dae")
#         self.rgt_arm.lnks[3]['rgba'] = [.57, .57, .57, 1]
#         self.rgt_arm.lnks[4]['name'] = "rgt_arm_joint2"
#         self.rgt_arm.lnks[4]['loc_pos'] = np.array([0, -0.095, -0.25])
#         self.rgt_arm.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "rarm_joint2_link_mesh.dae")
#         self.rgt_arm.lnks[4]['rgba'] = [.35, .35, .35, 1]
#         self.rgt_arm.lnks[5]['name'] = "rgt_arm_joint3"
#         self.rgt_arm.lnks[5]['loc_pos'] = np.array([-0.03, 0, 0])
#         self.rgt_arm.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "rarm_joint3_link_mesh.dae")
#         self.rgt_arm.lnks[5]['rgba'] = [.35, .35, .35, 1]
#         self.rgt_arm.lnks[6]['name'] = "rgt_arm_joint4"
#         self.rgt_arm.lnks[6]['loc_pos'] = np.array([0, 0, -0.235])
#         self.rgt_arm.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "rarm_joint4_link_mesh.dae")
#         self.rgt_arm.lnks[6]['rgba'] = [.7, .7, .7, 1]
#         self.rgt_arm.lnks[7]['name'] = "rgt_arm_joint5"
#         self.rgt_arm.lnks[7]['loc_pos'] = np.array([-0.047, 0, -0.09])
#         self.rgt_arm.lnks[7]['mesh_file'] = os.path.join(this_dir, "meshes", "rarm_joint5_link_mesh.dae")
#         self.rgt_arm.lnks[7]['rgba'] = [.57, .57, .57, 1]
#         self.rgt_arm.reinitialize()
#         # tool center point
#         # lft
#         self.lft_arm.tcp_joint_id = -1
#         # self.lft_arm._loc_flange_pos = self.lft_hnd.jaw_center_pos
#         # self.lft_arm._loc_flange_rotmat = self.lft_hnd.jaw_center_rotmat
#         self.lft_arm._loc_flange_pos = np.zeros(3)
#         self.lft_arm._loc_flange_rotmat = np.eye(3)
#         # rgt
#         self.rgt_arm.tcp_joint_id = -1
#         # self.rgt_arm._loc_flange_pos = self.rgt_hnd.jaw_center_pos
#         # self.rgt_arm._loc_flange_rotmat = self.rgt_hnd.jaw_center_rotmat
#         self.rgt_arm._loc_flange_pos = np.zeros(3)
#         self.rgt_arm._loc_flange_rotmat = np.eye(3)
#         # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
#         self.lft_oih_infos = []
#         self.rgt_oih_infos = []
#         # collision detection
#         if enable_cc:
#             self.enable_cc()
#         # component map
#         self.manipulator_dict['rgt_arm'] = self.rgt_arm
#         self.manipulator_dict['lft_arm'] = self.lft_arm
#         self.manipulator_dict['rgt_arm_waist'] = self.rgt_arm
#         self.manipulator_dict['lft_arm_waist'] = self.lft_arm
#         # self.hnd_dict['rgt_hnd'] = self.rgt_hnd
#         # self.hnd_dict['lft_hnd'] = self.lft_hnd
#
#     @staticmethod
#     def _waist_combined_cdnp(name, major_radius):
#         collision_node = CollisionNode(name)
#         collision_primitive_c0 = CollisionBox(Point3(-.183, 0, -1.68),
#                                               x=.3 + major_radius, y=.3 + major_radius, z=.26 + major_radius)
#         collision_node.addSolid(collision_primitive_c0)
#         collision_primitive_c1 = CollisionBox(Point3(-.183, 0, -1.28),
#                                               x=.3 + major_radius, y=.135 + major_radius, z=.15 + major_radius)
#         collision_node.addSolid(collision_primitive_c1)
#         collision_primitive_c2 = CollisionBox(Point3(0, 0, -.93),
#                                               x=.06 + major_radius, y=.06 + major_radius, z=.2 + major_radius)
#         collision_node.addSolid(collision_primitive_c2)
#         return collision_node
#
#     @staticmethod
#     def _chest_combined_cdnp(name, major_radius):
#         collision_node = CollisionNode(name)
#         collision_primitive_c0 = CollisionBox(Point3(-.0505, 0, .45),
#                                               x=.136 + major_radius, y=.12 + major_radius, z=.09 + major_radius)
#         collision_node.addSolid(collision_primitive_c0)
#         collision_primitive_c1 = CollisionBox(Point3(-.028, 0, .3),
#                                               x=.1 + major_radius, y=.07 + major_radius, z=.05 + major_radius)
#         collision_node.addSolid(collision_primitive_c1)
#         collision_primitive_l0 = CollisionBox(Point3(0.005, 0.16, .515),
#                                               x=.037 + major_radius, y=.055 + major_radius, z=.02 + major_radius)
#         collision_node.addSolid(collision_primitive_l0)
#         collision_primitive_r0 = CollisionBox(Point3(0.005, -0.16, .515),
#                                               x=.037 + major_radius, y=.055 + major_radius, z=.02 + major_radius)
#         collision_node.addSolid(collision_primitive_r0)
#         return collision_node
#
#     def enable_cc(self):
#         # TODO when pose is changed, oih info goes wrong
#         super().enable_cc()
#         self.cc.add_cdlnks(self.central_body, [0, 1, 2, 3])
#         self.cc.add_cdlnks(self.lft_arm, [2, 3, 4, 5, 6, 7])
#         self.cc.add_cdlnks(self.rgt_arm, [2, 3, 4, 5, 6, 7])
#         activelist = [self.lft_arm.lnks[2],
#                       self.lft_arm.lnks[3],
#                       self.lft_arm.lnks[4],
#                       self.lft_arm.lnks[5],
#                       self.lft_arm.lnks[6],
#                       self.lft_arm.lnks[7],
#                       self.rgt_arm.lnks[2],
#                       self.rgt_arm.lnks[3],
#                       self.rgt_arm.lnks[4],
#                       self.rgt_arm.lnks[5],
#                       self.rgt_arm.lnks[6],
#                       self.rgt_arm.lnks[7]]
#         self.cc.set_active_cdlnks(activelist)
#         fromlist = [self.central_body.lnks[0],
#                     self.central_body.lnks[1],
#                     self.central_body.lnks[3],
#                     self.lft_arm.lnks[2],
#                     self.rgt_arm.lnks[2]]
#         intolist = [self.lft_arm.lnks[5],
#                     self.lft_arm.lnks[6],
#                     self.lft_arm.lnks[7],
#                     self.rgt_arm.lnks[5],
#                     self.rgt_arm.lnks[6],
#                     self.rgt_arm.lnks[7]]
#         self.cc.set_cdpair(fromlist, intolist)
#         fromlist = [self.lft_arm.lnks[5],
#                     self.lft_arm.lnks[6],
#                     self.lft_arm.lnks[7]]
#         intolist = [self.rgt_arm.lnks[5],
#                     self.rgt_arm.lnks[6],
#                     self.rgt_arm.lnks[7]]
#         self.cc.set_cdpair(fromlist, intolist)
#
#     def get_hnd_on_manipulator(self, manipulator_name):
#         pass
#         # if hnd_name == 'rgt_arm':
#         #     return self.rgt_hnd
#         # elif hnd_name == 'lft_arm':
#         #     return self.lft_hnd
#         # else:
#         #     raise ValueError("The given jlc does not have a hand!")
#
#     def fix_to(self, pos, rotmat):
#         super().fix_to(pos, rotmat)
#         self.pos = pos
#         self.rotmat = rotmat
#         self.central_body.fix_to(self.pos, self.rotmat)
#         self.lft_arm.fix_to(self.pos, self.rotmat)
#         # self.lft_hnd.fix_to(pos=self.lft_arm.joints[-1]['gl_posq'],
#         #                     rotmat=self.lft_arm.joints[-1]['gl_rotmatq'])
#         self.rgt_arm.fix_to(self.pos, self.rotmat)
#         # self.rgt_hnd.fix_to(pos=self.rgt_arm.joints[-1]['gl_posq'],
#         #                     rotmat=self.rgt_arm.joints[-1]['gl_rotmatq'])
#
#     def fk(self, component_name, jnt_values):
#         """
#         waist angle is transmitted to arms
#         :param jnt_values: nparray 1x6 or 1x14 depending on component_names
#         :hnd_name 'lft_arm', 'rgt_arm', 'lft_arm_waist', 'rgt_arm_wasit', 'both_arm'
#         :param component_name:
#         :return:
#         author: weiwei
#         date: 20201208toyonaka
#         """
#
#         def update_oih(component_name='rgt_arm_waist'):
#             # inline function for update objects in hand
#             if component_name[:7] == 'rgt_arm':
#                 oih_info_list = self.rgt_oih_infos
#             elif component_name[:7] == 'lft_arm':
#                 oih_info_list = self.lft_oih_infos
#             for obj_info in oih_info_list:
#                 gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
#                 obj_info['gl_pos'] = gl_pos
#                 obj_info['gl_rotmat'] = gl_rotmat
#
#         def update_component(component_name, jnt_values):
#             status = self.manipulator_dict[component_name].fk(jnt_values=jnt_values)
#             hnd_on_manipulator = self.get_hnd_on_manipulator(component_name)
#             if hnd_on_manipulator is not None:
#                 hnd_on_manipulator.fix_to(pos=self.manipulator_dict[component_name].joints[-1]['gl_posq'],
#                                           rotmat=self.manipulator_dict[component_name].joints[-1]['gl_rotmatq'])
#             update_oih(component_name=component_name)
#             return status
#
#         # examine axis_length
#         if component_name == 'lft_arm' or component_name == 'rgt_arm':
#             if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 6:
#                 raise ValueError("An 1x6 npdarray must be specified to move a single arm!")
#             waist_value = self.central_body.joints[1]['motion_value']
#             return update_component(component_name, np.append(waist_value, jnt_values))
#         elif component_name == 'lft_arm_waist' or component_name == 'rgt_arm_waist':
#             if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 7:
#                 raise ValueError("An 1x7 npdarray must be specified to move a single arm plus the waist!")
#             status = update_component(component_name, jnt_values)
#             self.central_body.joints[1]['motion_value'] = jnt_values[0]
#             self.central_body.fk()
#             the_other_manipulator_name = 'lft_arm' if component_name[:7] == 'rgt_arm' else 'rgt_arm'
#             self.manipulator_dict[the_other_manipulator_name].joints[1]['motion_value'] = jnt_values[0]
#             self.manipulator_dict[the_other_manipulator_name].fk()
#             return status  # if waist is out of range, the first status will always be out of range
#         elif component_name == 'both_arm':
#             raise NotImplementedError
#         elif component_name == 'all':
#             raise NotImplementedError
#         else:
#             raise ValueError("The given component name is not available!")
#
#     @_decorator_switch_tgt_jnts
#     def ik(self,
#            component_name,
#            tgt_pos,
#            tgt_rotmat,
#            seed_jnt_values=None,
#            tcp_joint_id=None,
#            _loc_flange_pos=None,
#            _loc_flange_rotmat=None,
#            max_n_iter=100,
#            policy_for_local_minima="accept",
#            toggle_dbg=False):
#         # if component_name == 'lft_arm' or component_name == 'rgt_arm':
#         #     old_tgt_jnts = self.manipulator_dict[component_name].tgtjnts
#         #     self.manipulator_dict[component_name].tgtjnts = range(2, self.manipulator_dict[component_name].n_dof + 1)
#         #     ik_results = self.manipulator_dict[component_name].ik(tgt_pos,
#         #                                                           tgt_rotmat,
#         #                                                           seed_jnt_values=seed_jnt_values,
#         #                                                           tcp_joint_id=tcp_joint_id,
#         #                                                           _loc_flange_pos=_loc_flange_pos,
#         #                                                           _loc_flange_rotmat=_loc_flange_rotmat,
#         #                                                           max_n_iter=max_n_iter,
#         #                                                           policy_for_local_minima=policy_for_local_minima,
#         #                                                           toggle_dbg=toggle_dbg)
#         #     self.manipulator_dict[component_name].tgtjnts = old_tgt_jnts
#         #     return ik_results
#         # elif component_name == 'lft_arm_waist' or component_name == 'rgt_arm_waist':
#         #     return self.manipulator_dict[component_name].ik(tgt_pos,
#         #                                                     tgt_rotmat,
#         #                                                     seed_jnt_values=seed_jnt_values,
#         #                                                     tcp_joint_id=tcp_joint_id,
#         #                                                     _loc_flange_pos=_loc_flange_pos,
#         #                                                     _loc_flange_rotmat=_loc_flange_rotmat,
#         #                                                     max_n_iter=max_n_iter,
#         #                                                     policy_for_local_minima=policy_for_local_minima,
#         #                                                     toggle_dbg=toggle_dbg)
#         if component_name in ['lft_arm', 'rgt_arm', 'lft_arm_waist', 'rgt_arm_waist']:
#             return self.manipulator_dict[component_name].ik(tgt_pos,
#                                                             tgt_rotmat,
#                                                             seed_jnt_values=seed_jnt_values,
#                                                             tcp_joint_id=tcp_joint_id,
#                                                             _loc_flange_pos=_loc_flange_pos,
#                                                             _loc_flange_rotmat=_loc_flange_rotmat,
#                                                             max_n_iter=max_n_iter,
#                                                             policy_for_local_minima=policy_for_local_minima,
#                                                             toggle_dbg=toggle_dbg)
#         elif component_name == 'both_arm':
#             raise NotImplementedError
#         elif component_name == 'all':
#             raise NotImplementedError
#         else:
#             raise ValueError("The given component name is not available!")
#
#     @_decorator_switch_tgt_jnts
#     def get_jnt_values(self, component_name):
#         return self.manipulator_dict[component_name].get_jnt_values()
#
#     @_decorator_switch_tgt_jnts
#     def is_jnt_values_in_ranges(self, component_name, jnt_values):
#         # if component_name == 'lft_arm' or component_name == 'rgt_arm':
#         #     old_tgt_jnts = self.manipulator_dict[component_name].tgtjnts
#         #     self.manipulator_dict[component_name].tgtjnts = range(2, self.manipulator_dict[component_name].n_dof + 1)
#         #     result = self.manipulator_dict[component_name].is_jnt_values_in_ranges(jnt_values)
#         #     self.manipulator_dict[component_name].tgtjnts = old_tgt_jnts
#         #     return result
#         # else:
#         return self.manipulator_dict[component_name].is_jnt_values_in_ranges(jnt_values)
#
#     def rand_conf(self, component_name):
#         """
#         override robot_interface.rand_conf
#         :param component_name:
#         :return:
#         author: weiwei
#         date: 20210406
#         """
#         if component_name == 'lft_arm' or component_name == 'rgt_arm':
#             return super().rand_conf(component_name)[1:]
#         elif component_name == 'lft_arm_waist' or component_name == 'rgt_arm_waist':
#             return super().rand_conf(component_name)
#         elif component_name == 'both_arm':
#             return np.hstack((super().rand_conf('lft_arm')[1:], super().rand_conf('rgt_arm')[1:]))
#         else:
#             raise NotImplementedError
#
#     def hold(self, obj_cmodel, ee_values=None, hnd_name='lft_hnd'):
#         """
#         the obj_cmodel is added as a part of the robot_s to the cd checker
#         :param ee_values:
#         :param obj_cmodel:
#         :return:
#         """
#         # if hnd_name == 'lft_hnd':
#         #     rel_pos, rel_rotmat = self.lft_arm.cvt_gl_to_loc_tcp(obj_cmodel.get_pos(), obj_cmodel.get_rotmat())
#         #     into_list = [self.lft_body.lnks[0],
#         #                 self.lft_body.lnks[1],
#         #                 self.lft_arm.lnks[1],
#         #                 self.lft_arm.lnks[2],
#         #                 self.lft_arm.lnks[3],
#         #                 self.lft_arm.lnks[4],
#         #                 self.rgt_arm.lnks[1],
#         #                 self.rgt_arm.lnks[2],
#         #                 self.rgt_arm.lnks[3],
#         #                 self.rgt_arm.lnks[4],
#         #                 self.rgt_arm.lnks[5],
#         #                 self.rgt_arm.lnks[6],
#         #                 self.rgt_hnd.lft.lnks[0],
#         #                 self.rgt_hnd.lft.lnks[1],
#         #                 self.rgt_hnd.rgt.lnks[1]]
#         #     self.lft_oih_infos.append(self.cc.add_cdobj(obj_cmodel, rel_pos, rel_rotmat, into_list))
#         # elif hnd_name == 'rgt_hnd':
#         #     rel_pos, rel_rotmat = self.rgt_arm.cvt_gl_to_loc_tcp(obj_cmodel.get_pos(), obj_cmodel.get_rotmat())
#         #     into_list = [self.lft_body.lnks[0],
#         #                 self.lft_body.lnks[1],
#         #                 self.rgt_arm.lnks[1],
#         #                 self.rgt_arm.lnks[2],
#         #                 self.rgt_arm.lnks[3],
#         #                 self.rgt_arm.lnks[4],
#         #                 self.lft_arm.lnks[1],
#         #                 self.lft_arm.lnks[2],
#         #                 self.lft_arm.lnks[3],
#         #                 self.lft_arm.lnks[4],
#         #                 self.lft_arm.lnks[5],
#         #                 self.lft_arm.lnks[6],
#         #                 self.lft_hnd.lft.lnks[0],
#         #                 self.lft_hnd.lft.lnks[1],
#         #                 self.lft_hnd.rgt.lnks[1]]
#         #     self.rgt_oih_infos.append(self.cc.add_cdobj(obj_cmodel, rel_pos, rel_rotmat, into_list))
#         # else:
#         #     raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
#         # if ee_values is not None:
#         #     self.jaw_to(hnd_name, ee_values)
#         # return rel_pos, rel_rotmat
#
#     def get_loc_pose_from_hio(self, hio_pos, hio_rotmat, component_name='lft_arm'):
#         """
#         get the loc pose of an object from a grasp pose described in an object's local frame
#         :param hio_pos: a grasp pose described in an object's local frame -- pos
#         :param hio_rotmat: a grasp pose described in an object's local frame -- rotmat
#         :return:
#         author: weiwei
#         date: 20210302
#         """
#         if component_name == 'lft_arm':
#             arm = self.lft_arm
#         elif component_name == 'rgt_arm':
#             arm = self.rgt_arm
#         gripper_root_pos = arm.joints[-1]['gl_posq']
#         gripper_root_rotmat = arm.joints[-1]['gl_rotmatq']
#         hnd_homomat = rm.homomat_from_posrot(gripper_root_pos, gripper_root_rotmat)
#         hio_homomat = rm.homomat_from_posrot(hio_pos, hio_rotmat)
#         oih_homomat = rm.homomat_inverse(hio_homomat)
#         gl_obj_homomat = hnd_homomat.dot(oih_homomat)
#         return self.cvt_gl_to_loc_tcp(component_name, gl_obj_homomat[:3, 3], gl_obj_homomat[:3, :3])
#
#     def get_gl_pose_from_hio(self, hio_pos, hio_rotmat, component_name='lft_arm'):
#         """
#         get the loc pose of an object from a grasp pose described in an object's local frame
#         :param hio_pos: a grasp pose described in an object's local frame -- pos
#         :param hio_rotmat: a grasp pose described in an object's local frame -- rotmat
#         :return:
#         author: weiwei
#         date: 20210302
#         """
#         if component_name == 'lft_arm':
#             arm = self.lft_arm
#         elif component_name == 'rgt_arm':
#             arm = self.rgt_arm
#         gripper_root_pos = arm.joints[-1]['gl_posq']
#         gripper_root_rotmat = arm.joints[-1]['gl_rotmatq']
#         hnd_homomat = rm.homomat_from_posrot(gripper_root_pos, gripper_root_rotmat)
#         hio_homomat = rm.homomat_from_posrot(hio_pos, hio_rotmat)
#         oih_homomat = rm.homomat_inverse(hio_homomat)
#         gl_obj_homomat = hnd_homomat.dot(oih_homomat)
#         return gl_obj_homomat[:3, 3], gl_obj_homomat[:3, :3]
#
#     def get_oih_cm_list(self, hnd_name='lft_hnd'):
#         """
#         oih = object in hand list
#         :param hnd_name:
#         :return:
#         """
#         if hnd_name == 'lft_hnd':
#             oih_infos = self.lft_oih_infos
#         elif hnd_name == 'rgt_hnd':
#             oih_infos = self.rgt_oih_infos
#         else:
#             raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
#         return_list = []
#         for obj_info in oih_infos:
#             obj_cmodel = obj_info['collision_model']
#             obj_cmodel.set_pos(obj_info['gl_pos'])
#             obj_cmodel.set_rotmat(obj_info['gl_rotmat'])
#             return_list.append(obj_cmodel)
#         return return_list
#
#     def get_oih_glhomomat_list(self, hnd_name='lft_hnd'):
#         """
#         oih = object in hand list
#         :param hnd_name:
#         :return:
#         author: weiwei
#         date: 20210302
#         """
#         if hnd_name == 'lft_hnd':
#             oih_infos = self.lft_oih_infos
#         elif hnd_name == 'rgt_hnd':
#             oih_infos = self.rgt_oih_infos
#         else:
#             raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
#         return_list = []
#         for obj_info in oih_infos:
#             return_list.append(rm.homomat_from_posrot(obj_info['gl_pos']), obj_info['gl_rotmat'])
#         return return_list
#
#     def get_oih_relhomomat(self, obj_cmodel, hnd_name='lft_hnd'):
#         """
#         TODO: useless? 20210320
#         oih = object in hand list
#         :param obj_cmodel
#         :param hnd_name:
#         :return:
#         author: weiwei
#         date: 20210302
#         """
#         if hnd_name == 'lft_hnd':
#             oih_info_list = self.lft_oih_infos
#         elif hnd_name == 'rgt_hnd':
#             oih_info_list = self.rgt_oih_infos
#         else:
#             raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
#         for obj_info in oih_info_list:
#             if obj_info['collision_model'] is obj_cmodel:
#                 return rm.homomat_from_posrot(obj_info['rel_pos']), obj_info['rel_rotmat']
#
#     def release(self, hnd_name, obj_cmodel, ee_values=None):
#         """
#         the obj_cmodel is added as a part of the robot_s to the cd checker
#         :param ee_values:
#         :param obj_cmodel:
#         :param hnd_name:
#         :return:
#         """
#         if hnd_name == 'lft_hnd':
#             oih_infos = self.lft_oih_infos
#         elif hnd_name == 'rgt_hnd':
#             oih_infos = self.rgt_oih_infos
#         else:
#             raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
#         if ee_values is not None:
#             self.jaw_to(hnd_name, ee_values)
#         for obj_info in oih_infos:
#             if obj_info['collision_model'] is obj_cmodel:
#                 self.cc.delete_cdobj(obj_info)
#                 oih_infos.remove(obj_info)
#                 break
#
#     def release_all(self, ee_values=None, hnd_name='lft_hnd'):
#         """
#         release all objects from the specified hand
#         :param ee_values:
#         :param hnd_name:
#         :return:
#         author: weiwei
#         date: 20210125
#         """
#         if hnd_name == 'lft_hnd':
#             oih_infos = self.lft_oih_infos
#         elif hnd_name == 'rgt_hnd':
#             oih_infos = self.rgt_oih_infos
#         else:
#             raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
#         if ee_values is not None:
#             self.jaw_to(hnd_name, ee_values)
#         for obj_info in oih_infos:
#             self.cc.delete_cdobj(obj_info)
#         oih_infos.clear()
#
#     def gen_stickmodel(self,
#                        tcp_joint_id=None,
#                        _loc_flange_pos=None,
#                        _loc_flange_rotmat=None,
#                        toggle_flange_frame=False,
#                        toggle_jnt_frames=False,
#                        toggle_flange_frame=False,
#                        name='yumi'):
#         stickmodel = mmc.ModelCollection(name=name)
#         self.central_body.gen_stickmodel(_loc_flange_pos=None,
#                                          _loc_flange_rotmat=None,
#                                          toggle_flange_frame=False,
#                                          toggle_jnt_frames=toggle_jnt_frames).attach_to(stickmodel)
#         self.lft_arm.gen_stickmodel(tcp_joint_id=tcp_joint_id,
#                                     _loc_flange_pos=_loc_flange_pos,
#                                     _loc_flange_rotmat=_loc_flange_rotmat,
#                                     toggle_flange_frame=toggle_flange_frame,
#                                     toggle_jnt_frames=toggle_jnt_frames,
#                                     toggle_flange_frame=toggle_flange_frame).attach_to(stickmodel)
#         # self.lft_hnd.gen_stickmodel(toggle_flange_frame=False,
#         #                             toggle_jnt_frames=toggle_jnt_frames,
#         #                             toggle_flange_frame=toggle_flange_frame).attach_to(stickmodel)
#         self.rgt_arm.gen_stickmodel(tcp_joint_id=tcp_joint_id,
#                                     _loc_flange_pos=_loc_flange_pos,
#                                     _loc_flange_rotmat=_loc_flange_rotmat,
#                                     toggle_flange_frame=toggle_flange_frame,
#                                     toggle_jnt_frames=toggle_jnt_frames,
#                                     toggle_flange_frame=toggle_flange_frame).attach_to(stickmodel)
#         # self.rgt_hnd.gen_stickmodel(toggle_flange_frame=False,
#         #                             toggle_jnt_frames=toggle_jnt_frames,
#         #                             toggle_flange_frame=toggle_flange_frame).attach_to(stickmodel)
#         return stickmodel
#
#     def gen_meshmodel(self,
#                       tcp_joint_id=None,
#                       _loc_flange_pos=None,
#                       _loc_flange_rotmat=None,
#                       toggle_flange_frame=False,
#                       toggle_jnt_frames=False,
#                       rgba=None,
#                       name='xarm_gripper_meshmodel'):
#         meshmodel = mmc.ModelCollection(name=name)
#         self.central_body.gen_meshmodel(_loc_flange_pos=None,
#                                         _loc_flange_rotmat=None,
#                                         toggle_flange_frame=False,
#                                         toggle_jnt_frames=toggle_jnt_frames,
#                                         rgba=rgba).attach_to(meshmodel)
#         self.lft_arm.gen_meshmodel(tcp_joint_id=tcp_joint_id,
#                                    _loc_flange_pos=_loc_flange_pos,
#                                    _loc_flange_rotmat=_loc_flange_rotmat,
#                                    toggle_flange_frame=toggle_flange_frame,
#                                    toggle_jnt_frames=toggle_jnt_frames,
#                                    rgba=rgba).attach_to(meshmodel)
#         # self.lft_hnd.gen_meshmodel(toggle_flange_frame=False,
#         #                            toggle_jnt_frames=toggle_jnt_frames,
#         #                            rgba=rgba).attach_to(meshmodel)
#         self.rgt_arm.gen_meshmodel(tcp_joint_id=tcp_joint_id,
#                                    _loc_flange_pos=_loc_flange_pos,
#                                    _loc_flange_rotmat=_loc_flange_rotmat,
#                                    toggle_flange_frame=toggle_flange_frame,
#                                    toggle_jnt_frames=toggle_jnt_frames,
#                                    rgba=rgba).attach_to(meshmodel)
#         # self.rgt_hnd.gen_meshmodel(toggle_flange_frame=False,
#         #                            toggle_jnt_frames=toggle_jnt_frames,
#         #                            rgba=rgba).attach_to(meshmodel)
#         for obj_info in self.lft_oih_infos:
#             obj_cmodel = obj_info['collision_model']
#             obj_cmodel.set_pos(obj_info['gl_pos'])
#             obj_cmodel.set_rotmat(obj_info['gl_rotmat'])
#             obj_cmodel.copy().attach_to(meshmodel)
#         for obj_info in self.rgt_oih_infos:
#             obj_cmodel = obj_info['collision_model']
#             obj_cmodel.set_pos(obj_info['gl_pos'])
#             obj_cmodel.set_rotmat(obj_info['gl_rotmat'])
#             obj_cmodel.copy().attach_to(meshmodel)
#         return meshmodel


if __name__ == '__main__':
    import time
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[3, 1, 2], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    nxt_instance = Left_Manipulator()
    nxt_meshmodel = nxt_instance.gen_meshmodel(toggle_tcp_frame=True)
    nxt_meshmodel.attach_to(base)
    # nxt_instance.show_cdprimit()
    base.run()

    # tgt_pos = np.array([.4, 0, .2])
    # tgt_rotmat = rm.rotmat_from_euler(0, math.pi * 2 / 3, -math.pi / 4)
    # ik test
    component_name = 'lft_arm_waist'
    tgt_pos = np.array([-.3, .45, .55])
    tgt_rotmat = rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
    # tgt_rotmat = np.eye(3)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    tic = time.time()
    jnt_values = nxt_instance.ik(component_name, tgt_pos, tgt_rotmat, toggle_dbg=True)
    toc = time.time()
    print(toc - tic)
    nxt_instance.fk(component_name, jnt_values)
    nxt_meshmodel = nxt_instance.gen_meshmodel()
    nxt_meshmodel.attach_to(base)
    nxt_instance.gen_stickmodel().attach_to(base)
    # tic = time.time()
    # result = nxt_instance.is_collided()
    # toc = time.time()
    # print(result, toc - tic)
    base.run()

    # hold test
    component_name = 'lft_arm'
    obj_pos = np.array([-.1, .3, .3])
    obj_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    objfile = os.path.join(basis.__path__[0], 'objects', 'tubebig.stl')
    objcm = cm.CollisionModel(objfile, cdprim_type='cylinder')
    objcm.set_pos(obj_pos)
    objcm.set_rotmat(obj_rotmat)
    objcm.attach_to(base)
    objcm_copy = objcm.copy()
    nxt_instance.hold(objcm=objcm_copy, jaw_width=0.03, hnd_name='lft_hnd')
    tgt_pos = np.array([.4, .5, .4])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 3)
    jnt_values = nxt_instance.ik(component_name, tgt_pos, tgt_rotmat)
    nxt_instance.fk(component_name, jnt_values)
    # nxt_instance.show_cdprimit()
    nxt_meshmodel = nxt_instance.gen_meshmodel()
    nxt_meshmodel.attach_to(base)

    base.run()
