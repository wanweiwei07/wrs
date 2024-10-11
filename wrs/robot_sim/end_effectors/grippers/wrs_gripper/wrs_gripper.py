"""
The simulation for the WRS grippers designed by Wan sensei
Author: Chen Hao (chen960216@gmail.com), 20220924, osaka
Reference: The code is implemented referring to the 'robot_sim/end_effectors/yumi_gripper/'
"""
import os
import math
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, Point3
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi


class Lite6WRSGripper(gpi.GripperInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='convex_hull', name='wrs_gripper',
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        # lft finger and base
        self.lft = rkjlc.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(1), name='base_lft_finger')
        self.lft.lnks[0]['name'] = "base"
        self.lft.lnks[0]['loc_pos'] = np.zeros(3)
        self.lft.lnks[0]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "base.stl"),
                                                                cdprim_type="user_defined", ex_radius=.001,
                                                                userdef_cdprim_fn=self._hnd_base_cdnp)
        self.lft.lnks[0]['rgba'] = [.77, .77, .77, 1]
        self.lft.jnts[1]['loc_pos'] = np.array([0, 0.0040, 0.089])
        self.lft.jnts[1]['gl_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi)
        self.lft.jnts[1]['end_type'] = 'prismatic'
        self.lft.jnts[1]['motion_range'] = [.0, .025]
        self.lft.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.lft.lnks[1]['name'] = "finger1"
        self.lft.lnks[1]['mesh_file'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "finger.stl"), cdprim_type="user_defined",
            userdef_cdprim_fn=self._finger_cdnp, ex_radius=.001)
        self.lft.lnks[1]['rgba'] = [.35, .35, .35, 1]
        # # rgt finger
        self.rgt = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(1), name='rgt_finger')
        self.rgt.jnts[1]['loc_pos'] = np.array([0, -0.0040, 0.089])
        self.rgt.jnts[1]['end_type'] = 'prismatic'
        self.rgt.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt.lnks[1]['name'] = "finger2"
        self.rgt.lnks[1]['mesh_file'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "finger.stl"), cdprim_type="user_defined",
            userdef_cdprim_fn=self._finger_cdnp, ex_radius=.000)
        self.rgt.lnks[1]['rgba'] = [.35, .35, .35, 1]
        # # reinitialize
        self.lft.finalize(cdmesh_type=cdmesh_type)
        self.rgt.finalize(cdmesh_type=cdmesh_type)
        # jaw width range
        self.jaw_range = [0.0, .04]
        # jaw center
        self.jaw_center_pos = np.array([0, 0, .139])
        # collision detection
        self.all_cdelements = []
        self.enable_cc(toggle_cdprimit=enable_cc)

    @staticmethod
    def _finger_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-.005, 0.004, .025),
                                              x=.005 + radius, y=0.004 + radius, z=.025 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(.008, 0.028 - .002, -.011),
                                              x=.018 + radius, y=0.008 + radius, z=.011 + radius)
        collision_node.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(-.005, 0.012 - .002, -.002 + .0025),
                                              x=.005 + radius, y=0.008 + radius, z=.002 + .0025 + radius)
        collision_node.addSolid(collision_primitive_c2)
        return collision_node

    @staticmethod
    def _hnd_base_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0, 0, .031),
                                              x=.036 + radius, y=0.038 + radius, z=.031 + radius)
        collision_node.addSolid(collision_primitive_c0)  # 0.62
        collision_primitive_c1 = CollisionBox(Point3(0, 0, .067),
                                              x=.036 + radius, y=0.027 + radius, z=.003 + radius)
        collision_node.addSolid(collision_primitive_c1)  # 0.06700000
        #
        collision_primitive_c2 = CollisionBox(Point3(.006, .049, .0485),
                                              x=.02 + radius, y=.02 + radius, z=.015 + radius)
        collision_node.addSolid(collision_primitive_c2)
        collision_primitive_c3 = CollisionBox(Point3(0, 0, .08),
                                              x=.013 + radius, y=0.013 + radius, z=.005 + radius)
        collision_node.addSolid(collision_primitive_c3)

        return collision_node

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            # cdprimit
            self.cc.add_cdlnks(self.lft, [0, 1])
            self.cc.add_cdlnks(self.rgt, [1])
            activelist = [self.lft.lnks[0],
                          self.lft.lnks[1],
                          self.rgt.lnks[1]]
            self.cc.set_active_cdlnks(activelist)
            self.all_cdelements = self.cc.cce_dict
        else:
            self.all_cdelements = [self.lft.lnks[0],
                                   self.lft.lnks[1],
                                   self.rgt.lnks[1]]
        # cdmesh
        for cdelement in self.all_cdelements:
            cdmesh = cdelement['collision_model'].copy()
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.coupling.fix_to(self._pos, self._rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.lft.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt.fix_to(cpl_end_pos, cpl_end_rotmat)

    def fk(self, motion_val):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: motion_value, meter or radian
        """
        if self.lft.jnts[1]['motion_range'][0] <= -motion_val <= self.lft.jnts[1]['motion_range'][1]:
            self.lft.jnts[1]['motion_value'] = motion_val
            self.rgt.jnts[1]['motion_value'] = self.lft.jnts[1]['motion_value']
            self.lft.fk()
            self.rgt.fk()
        else:
            raise ValueError("The motion_value parameter is out of range!")

    def change_jaw_width(self, jaw_width):
        if jaw_width > self.jaw_range[1]:
            raise ValueError("The ee_values parameter is out of range!")
        self.fk(motion_val=-jaw_width / 2.0)

    def get_jaw_width(self):
        return -self.lft.jnts[1]['motion_value'] * 2

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='ee_stickmodel'):
        stickmodel = mmc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_tcp_frame=False, toggle_jnt_frames=toggle_jnt_frames).attach_to(stickmodel)
        self.lft.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jnt_frames,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt.gen_stickmodel(tcp_loc_pos=None,
                                tcp_loc_rotmat=None,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jnt_frames,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        if toggle_tcp_frame:
            jaw_center_gl_pos = self._rotmat.dot(self.jaw_center_pos) + self._pos
            jaw_center_gl_rotmat = self._rotmat.dot(self.loc_acting_center_rotmat)
            gm.gen_dashed_stick(spos=self._pos,
                                epos=jaw_center_gl_pos,
                                radius=.0062,
                                rgba=[.5, 0, 1, 1],
                                type="round").attach_to(stickmodel)
            gm.gen_myc_frame(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(stickmodel)

        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      rgba=None,
                      name='lite6wrs_gripper_meshmodel'):
        meshmodel = mmc.ModelCollection(name=name)
        self.coupling.gen_mesh_model(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jnt_frames,
                                     rgba=rgba).attach_to(meshmodel)
        self.lft.gen_mesh_model(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jnt_frames,
                                rgba=rgba).attach_to(meshmodel)
        self.rgt.gen_mesh_model(tcp_loc_pos=None,
                                tcp_loc_rotmat=None,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jnt_frames,
                                rgba=rgba).attach_to(meshmodel)
        if toggle_tcp_frame:
            jaw_center_gl_pos = self._rotmat.dot(self.jaw_center_pos) + self._pos
            jaw_center_gl_rotmat = self._rotmat.dot(self.loc_acting_center_rotmat)
            gm.gen_dashed_stick(spos=self._pos,
                                epos=jaw_center_gl_pos,
                                radius=.0062,
                                rgba=[.5, 0, 1, 1],
                                type="round").attach_to(meshmodel)
            gm.gen_myc_frame(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    grpr = Lite6WRSGripper(enable_cc=True)
    grpr.change_jaw_width(.03)
    grpr.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    grpr.show_cdprimit()
    base.run()
