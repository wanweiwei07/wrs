"""
Definition for Dobot Nova 2
Author: Chen Hao <chen960216@gmail.com>, 20230214, osaka
TODO:
    - Kinematics parameter of the robot seems not exactly the same as the official document
"""

import os
import math
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import modeling.collision_model as mcm
import robot_sim.manipulators.manipulator_interface as mi


class Nova2(mi.ManipulatorInterface):
    """
    Definition for Dobot Nova 2
    author: chen hao <chen960216@gmail.com>, 20230214osaka; weiwei20240611
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), ik_solver='d', name='nova2', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=np.zeros(6), name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "base.stl"),
            cdprim_type=mcm.mc.CDPType.USER_DEFINED,
            userdef_cdprim_fn=self._base_cdprim)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # first joint and link
        self.jlc.jnts[0].loc_pos = np.array([.0, .0, .22339219])
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-2 * np.pi, 2 * np.pi])
        self.jlc.jnts[0].lnk.loc_pos = np.array([0., 0., -0.22339129])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link1.stl"))
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # second joint and link
        self.jlc.jnts[1].loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[1].loc_rotmat = np.array([[1.15684894e-04, -9.99999993e-01, 0.00000000e+00],
                                                [-1.05396766e-03, -1.21928137e-07, -9.99999445e-01],
                                                [9.99999438e-01, 1.15684829e-04, -1.05396766e-03]])
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[1].motion_range = np.array([-np.pi, np.pi])
        self.jlc.jnts[1].lnk.loc_pos = np.array([0, 0, 0.06175])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link2.stl"))
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # third joint and link
        self.jlc.jnts[2].loc_pos = np.array([.28053836, -.0, .0])
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[2].motion_range = np.array([-2.722713633111154, 2.722713633111154])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link3.stl"))
        self.jlc.jnts[2].lnk.loc_pos = np.array([0, 0, 0.0387])
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = np.array([0.22461726, -0., 0.11752977])
        self.jlc.jnts[3].loc_rotmat = np.array([[-9.54658556e-05, 9.99999995e-01, 0.00000000e+00],
                                                [-9.99999995e-01, -9.54658556e-05, -0.00000000e+00],
                                                [-0.00000000e+00, -0.00000000e+00, 1.00000000e+00]])
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[3].motion_range = np.array([-2 * np.pi, 2 * np.pi])
        self.jlc.jnts[3].lnk.loc_pos = np.array([0, .004, +0.044 - 1.15025404e-01])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link4.stl"))
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = np.array([0, 1.19844295e-01, 9.28531056e-05])
        self.jlc.jnts[4].loc_rotmat = np.array([[1.00000000e+00, -0.00000000e+00, 0.00000000e+00],
                                                [0.00000000e+00, 7.74780958e-04, 9.99999700e-01],
                                                [-0.00000000e+00, -9.99999700e-01, 7.74780958e-04]])
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[4].motion_range = np.array([-2 * np.pi, 2 * np.pi])
        self.jlc.jnts[4].lnk.loc_pos = np.array([0, 0, -1.15025404e-01 + .065])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link5.stl"))
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = np.array([0., -8.80028118e-02, 0])
        self.jlc.jnts[5].loc_rotmat = np.array([[-9.99996405e-01, -2.68157574e-03, 0.00000000e+00],
                                                [1.64199157e-19, -6.12321198e-17, -1.00000000e+00],
                                                [2.68157574e-03, -9.99996405e-01, 6.12323400e-17]])
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[5].motion_range = np.array([-2 * np.pi, 2 * np.pi])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link6.stl"))
        self.jlc.jnts[5].lnk.loc_pos = np.array([0., -0.0038, -0.036])
        self.jlc.jnts[5].lnk.loc_rotmat = np.array([[-9.99996404e-01, 1.64199157e-19, 2.68157574e-03],
                                                   [-2.68157574e-03, -6.12321198e-17, -9.99996404e-01],
                                                   [2.60423050e-28, -1.00000000e+00, 6.12323399e-17]])
        self.jlc.finalize(ik_solver=ik_solver, identifier_str=name)
        # tcp
        self.loc_tcp_pos = np.array([0, 0, 0])
        self.loc_tcp_rotmat = np.eye(3)
        # set up cc
        if self.cc is not None:
            self.setup_cc()

    # self-defined collison model for the base link
    @staticmethod
    def _base_cdprim(ex_radius):
        pdcnd = CollisionNode("nova_base")
        collision_primitive_c0 = CollisionBox(Point3(-0.008, 0, 0.0375),
                                              x=.07 + ex_radius, y=.065 + ex_radius, z=0.0375 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0, 0, .124),
                                              x=.043 + ex_radius, y=.043 + ex_radius, z=.049 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        cdprim = NodePath("user_defined")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    def setup_cc(self):
        lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)
        from_list = [l3, l4, l5]
        into_list = [lb, l0]
        self.cc.set_cdpair_by_ids(from_list, into_list)


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    arm = Nova2(enable_cc=True)
    random_conf = arm.rand_conf()
    arm.fk(random_conf)
    print(arm.gl_flange_pos, arm.gl_flange_rotmat)
    arm_meshmodel = arm.gen_meshmodel()
    arm_meshmodel.attach_to(base)
    arm.gen_stickmodel(toggle_jnt_frames=True).attach_to(base)
    arm_meshmodel.show_cdprim()
    tic = time.time()
    print(arm.is_collided())
    toc = time.time()
    print(toc - tic)
    base.run()
