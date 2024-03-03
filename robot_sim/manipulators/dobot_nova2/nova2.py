"""
Definition for Dobot Nova 2
Author: Chen Hao <chen960216@gmail.com>, 20230214, osaka
TODO:
    - Kinematics parameter of the robot seems not exactly the same as the official document
"""

import os
import math
import numpy as np
from panda3d.core import (CollisionNode,
                          CollisionBox,
                          Point3)
import modeling.collision_model as cm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.manipulator_interface as mi


class Nova2(mi.ManipulatorInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(6), name='nova2', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, home_conf=homeconf, name=name)
        # six joints
        jnt_saferngmargin = math.pi / 18.0
        # joint 1
        self.jlc.jnts[1]['loc_pos'] = np.array([0., -0., 0.22339219])
        self.jlc.jnts[1]['motion_range'] = [-2. * math.pi + jnt_saferngmargin, 2. * math.pi - jnt_saferngmargin]
        # joint 2
        self.jlc.jnts[2]['loc_pos'] = np.zeros(3)
        self.jlc.jnts[2]['gl_rotmat'] = np.array([[1.15684894e-04, -9.99999993e-01, 0.00000000e+00],
                                                  [-1.05396766e-03, -1.21928137e-07, -9.99999445e-01],
                                                  [9.99999438e-01, 1.15684829e-04, -1.05396766e-03]])
        self.jlc.jnts[2]['motion_range'] = [-math.pi + jnt_saferngmargin, math.pi - jnt_saferngmargin]
        # joint 3
        self.jlc.jnts[3]['loc_pos'] = np.array([0.28053836, -0., 0.])
        self.jlc.jnts[3]['motion_range'] = [-2.722713633111154 + jnt_saferngmargin, 2.722713633111154 - jnt_saferngmargin]
        # joint 4
        self.jlc.jnts[4]['loc_pos'] = np.array([0.22461726, -0., 0.11752977])
        self.jlc.jnts[4]['gl_rotmat'] = np.array([[-9.54658556e-05, 9.99999995e-01, 0.00000000e+00],
                                                  [-9.99999995e-01, -9.54658556e-05, -0.00000000e+00],
                                                  [-0.00000000e+00, -0.00000000e+00, 1.00000000e+00]])
        self.jlc.jnts[4]['motion_range'] = [-2. * math.pi + jnt_saferngmargin, 2. * math.pi - jnt_saferngmargin]
        # joint 5
        self.jlc.jnts[5]['loc_pos'] = np.array([0, 1.19844295e-01, 9.28531056e-05])
        self.jlc.jnts[5]['gl_rotmat'] = np.array([[1.00000000e+00, -0.00000000e+00, 0.00000000e+00],
                                                  [0.00000000e+00, 7.74780958e-04, 9.99999700e-01],
                                                  [-0.00000000e+00, -9.99999700e-01, 7.74780958e-04]])
        self.jlc.jnts[5]['motion_range'] = [-2. * math.pi + jnt_saferngmargin, 2. * math.pi - jnt_saferngmargin]
        # joint 6
        self.jlc.jnts[6]['loc_pos'] = np.array([0., -8.80028118e-02, 0])
        self.jlc.jnts[6]['gl_rotmat'] = np.array([[-9.99996405e-01, -2.68157574e-03, 0.00000000e+00],
                                                  [1.64199157e-19, -6.12321198e-17, -1.00000000e+00],
                                                  [2.68157574e-03, -9.99996405e-01, 6.12323400e-17]])
        self.jlc.jnts[6]['motion_range'] = [-2. * math.pi + jnt_saferngmargin, 2. * math.pi - jnt_saferngmargin]
        # links
        # link base
        self.jlc.lnks[0]['name'] = "link_base"
        self.jlc.lnks[0]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[0]['mass'] = 2.11
        self.jlc.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "base.stl"),
            cdprim_type="user_defined",
            userdef_cdprim_fn=self._base_cdnp)
        self.jlc.lnks[0]['rgba'] = [.7, .7, .7, 1.0]
        # # link1
        self.jlc.lnks[1]['name'] = "link1"
        self.jlc.lnks[1]['loc_pos'] = np.array([0., 0., -0.22339129])
        self.jlc.lnks[1]['mass'] = 1.411
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "link1.stl")
        # # link2
        self.jlc.lnks[2]['name'] = "link2"
        self.jlc.lnks[2]['loc_pos'] = np.array([0, 0, 0.06175])
        self.jlc.lnks[2]['mass'] = 1.34
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "link2.stl")
        # self.jlc.lnks[2]['rgba'] = [.7, .7, .7, 1.0]
        # # link 3
        self.jlc.lnks[3]['name'] = "link3"
        self.jlc.lnks[3]['loc_pos'] = np.array([0, 0, 0.0387])
        self.jlc.lnks[3]['mass'] = 0.953
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "link3.stl")
        # # link 4
        self.jlc.lnks[4]['name'] = "link4"
        self.jlc.lnks[4]['loc_pos'] = np.array([0, .004, +0.044 - 1.15025404e-01])
        self.jlc.lnks[4]['mass'] = 1.284
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "link4.stl")
        # self.jlc.lnks[4]['rgba'] = [.7, .7, .7, 1.0]
        # # link 5
        self.jlc.lnks[5]['name'] = "link5"
        self.jlc.lnks[5]['loc_pos'] = np.array([0, 0, -1.15025404e-01 + .065])
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "link5.stl")
        self.jlc.lnks[5]['rgba'] = [.7, .7, .7, 1.0]
        # # link 6
        self.jlc.lnks[6]['name'] = "link6"
        self.jlc.lnks[6]['loc_pos'] = np.array([0., -0.0038, -0.036])
        self.jlc.lnks[6]['gl_rotmat'] = np.array([[-9.99996404e-01, 1.64199157e-19, 2.68157574e-03],
                                                   [-2.68157574e-03, -6.12321198e-17, -9.99996404e-01],
                                                   [2.60423050e-28, -1.00000000e+00, 6.12323399e-17]])
        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "link6.stl")
        self.jlc.lnks[6]['rgba'] = [.57, .57, .57, 1.0]
        # reinitialization
        self.jlc.finalize()
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
                    self.jlc.lnks[1]]
        intolist = [self.jlc.lnks[3],
                    self.jlc.lnks[5],
                    self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.jlc.lnks[2]]
        intolist = [self.jlc.lnks[4],
                    self.jlc.lnks[5],
                    self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.jlc.lnks[3]]
        intolist = [self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    manipulator_instance = Nova2(enable_cc=True)
    random_conf = manipulator_instance.rand_conf()
    manipulator_instance.fk(random_conf)
    print(manipulator_instance.get_gl_tcp())
    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    manipulator_instance.show_cdprim()
    tic = time.time()
    print(manipulator_instance.is_collided())
    toc = time.time()
    print(toc - tic)
    base.run()
