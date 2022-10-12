"""
Definition for XArm Lite 6
Author: Chen Hao (chen960216@gmail.com), 20220909, osaka
Reference: https://github.com/xArm-Developer/xarm_ros/blob/d6bae33affd402cdf56c7651c6d35314c37fe3de/xarm_description/urdf/lite6.urdf.xacro
"""

import os
import math

import numpy as np
from panda3d.core import (CollisionNode,
                          CollisionBox,
                          Point3)

import basis.robot_math as rm
import modeling.collision_model as cm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.manipulator_interface as mi


class XArmLite6(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.array([0., 0.173311, 0.555015, 0., 0.381703, 0.]),
                 name='xarm_lite6', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=homeconf, name=name)
        # six joints
        jnt_saferngmargin = math.pi / 18.0
        # jnt 1
        self.jlc.jnts[1]['loc_pos'] = np.array([0., 0., .2433])
        self.jlc.jnts[1]['motion_rng'] = [-2. * math.pi + jnt_saferngmargin, 2. * math.pi - jnt_saferngmargin]
        # jnt 2
        self.jlc.jnts[2]['loc_pos'] = np.array([0., 0., 0.])
        self.jlc.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(1.5708, -1.5708, 3.1416)
        self.jlc.jnts[2]['motion_rng'] = [-2.61799 + jnt_saferngmargin, 2.61799 - jnt_saferngmargin]
        # jnt 3
        self.jlc.jnts[3]['loc_pos'] = np.array([.2, 0., 0.])
        self.jlc.jnts[3]['loc_rotmat'] = rm.rotmat_from_euler(-3.1416, 0., 1.5708)
        self.jlc.jnts[3]['motion_rng'] = [-0.061087 + jnt_saferngmargin, 5.235988 - jnt_saferngmargin]
        # jnt 4
        self.jlc.jnts[4]['loc_pos'] = np.array([.087, -.2276, 0.])
        self.jlc.jnts[4]['loc_rotmat'] = rm.rotmat_from_euler(1.5708, 0., 0.)
        self.jlc.jnts[4]['motion_rng'] = [-2. * math.pi + jnt_saferngmargin, 2. * math.pi - jnt_saferngmargin]
        # jnt 5
        self.jlc.jnts[5]['loc_pos'] = np.array([0., 0., 0.])
        self.jlc.jnts[5]['loc_rotmat'] = rm.rotmat_from_euler(1.5708, 0., 0.)
        self.jlc.jnts[5]['motion_rng'] = [-2.1642 + jnt_saferngmargin, 2.1642 - jnt_saferngmargin]
        # jnt 6
        self.jlc.jnts[6]['loc_pos'] = np.array([0., .0615, 0.])
        self.jlc.jnts[6]['loc_rotmat'] = rm.rotmat_from_euler(-1.5708, 0., 0.)
        self.jlc.jnts[6]['motion_rng'] = [-2. * math.pi + jnt_saferngmargin, 2. * math.pi - jnt_saferngmargin]
        # links
        # link base
        self.jlc.lnks[0]['name'] = "link_base"
        self.jlc.lnks[0]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[0]['com'] = np.array([-0.00829544579053192, 3.26357432323433e-05, 0.0631194584987089])
        self.jlc.lnks[0]['mass'] = 2.11
        self.jlc.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "base.stl"),
            cdprimit_type="user_defined",
            userdefined_cdprimitive_fn=self._base_cdnp)
        self.jlc.lnks[0]['rgba'] = [.7, .7, .7, 1.0]
        # link1
        self.jlc.lnks[1]['name'] = "link1"
        self.jlc.lnks[1]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[1]['com'] = np.array([-.00036, .042, -.0025])
        self.jlc.lnks[1]['mass'] = 1.411
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "link1.stl")
        # link2
        self.jlc.lnks[2]['name'] = "link2"
        self.jlc.lnks[2]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[2]['com'] = np.array([0.179, .0, .0584])
        self.jlc.lnks[2]['mass'] = 1.34
        self.jlc.lnks[2]['mesh_file'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "link2.stl"),
                                                          cdprimit_type="user_defined",
                                                          userdefined_cdprimitive_fn=self._link2_cdnp)
        # os.path.join(this_dir, "meshes", "link2.stl")
        self.jlc.lnks[2]['rgba'] = [.7, .7, .7, 1.0]
        # link 3
        self.jlc.lnks[3]['name'] = "link3"
        self.jlc.lnks[3]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[3]['com'] = np.array([0.072, -0.036, -0.001])
        self.jlc.lnks[3]['mass'] = 0.953
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "link3.stl")
        # link 4
        self.jlc.lnks[4]['name'] = "link4"
        self.jlc.lnks[4]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[4]['com'] = np.array([-0.002, -0.0285, -0.0813])
        self.jlc.lnks[4]['mass'] = 1.284
        self.jlc.lnks[4]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "link4.stl"),
            cdprimit_type="user_defined",
            userdefined_cdprimitive_fn=self._link4_cdnp)
        self.jlc.lnks[4]['rgba'] = [.7, .7, .7, 1.0]
        # link 5
        self.jlc.lnks[5]['name'] = "link5"
        self.jlc.lnks[5]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[5]['com'] = np.array([0.0, 0.010, 0.0019])
        self.jlc.lnks[5]['mass'] = 0.804
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "link5.stl")
        self.jlc.lnks[5]['rgba'] = [.7, .7, .7, 1.0]
        # link 6
        self.jlc.lnks[6]['name'] = "link6"
        self.jlc.lnks[6]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[6]['com'] = np.array([0.0, -0.00194, -0.0102])
        self.jlc.lnks[6]['mass'] = 0.180
        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "link6.stl")
        self.jlc.lnks[6]['rgba'] = [.57, .57, .57, 1.0]
        # reinitialization
        # self.jlc.setinitvalues(np.array([-math.pi/2, math.pi/3, math.pi/6, 0, 0, 0, 0]))
        # self.jlc.setinitvalues(np.array([-math.pi/2, 0, math.pi/3, math.pi/10, 0, 0, 0]))
        self.jlc.reinitialize()
        # collision detection
        if enable_cc:
            self.enable_cc()

    # self-defined collison model for the base link
    @staticmethod
    def _base_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-0.008, 0, 0.0375),
                                              x=.07 + radius, y=.065 + radius, z=0.0375 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0, 0, .124),
                                              x=.043 + radius, y=.043 + radius, z=.049 + radius)
        collision_node.addSolid(collision_primitive_c1)
        return collision_node

    @staticmethod
    def _link4_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0, 0, -0.124009),
                                              x=.041 + radius, y=.042 + radius, z=0.0682075 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0, -0.063315, -0.0503),
                                              x=.041 + radius, y=.021315 + radius, z=.087825 + radius)
        collision_node.addSolid(collision_primitive_c1)
        return collision_node

    @staticmethod
    def _link2_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0, 0, 0.1065),
                                              x=.041 + radius, y=.042 + radius, z=0.0315 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.100, 0, 0.1065),
                                              x=.059 + radius, y=.042 + radius, z=0.0315 + radius)
        collision_node.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(.2, 0, 0.0915),
                                              x=.041 + radius, y=.042 + radius, z=0.0465 + radius)
        collision_node.addSolid(collision_primitive_c2)
        return collision_node

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 6])
        activelist = [self.jlc.lnks[0],
                      self.jlc.lnks[1],
                      self.jlc.lnks[2],
                      self.jlc.lnks[3],
                      self.jlc.lnks[4],
                      self.jlc.lnks[5],
                      self.jlc.lnks[6], ]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.jlc.lnks[0],
                    self.jlc.lnks[1],
                    self.jlc.lnks[2]]
        intolist = [self.jlc.lnks[5],
                    self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    manipulator_instance = XArmLite6(enable_cc=True)
    random_conf = manipulator_instance.rand_conf()
    random_conf = np.array([0, 0, np.radians(180), 0, 0, 0])
    manipulator_instance.fk(random_conf)
    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    manipulator_instance.show_cdprimit()
    tic = time.time()
    print(manipulator_instance.is_collided())
    toc = time.time()
    print(toc - tic)

    base.run()
