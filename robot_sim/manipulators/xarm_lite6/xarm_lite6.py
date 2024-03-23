import os
import math
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import basis.robot_math as rm
import modeling.collision_model as mcm
import robot_sim.manipulators.manipulator_interface as mi


class XArmLite6(mi.ManipulatorInterface):
    """
    Definition for XArm Lite 6
    Author: Chen Hao (chen960216@gmail.com), Updated by Weiwei
    Date: 20220909osaka, 20240318
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),
                 home_conf=np.array([0., 0.173311, 0.555015, 0., 0.381703, 0.]),
                 name='xarm_lite6', enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=home_conf, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "base.stl"),
                                                                cdprim_type=mcm.mc.CDPType.USER_DEFINED, ex_radius=.005,
                                                                userdef_cdprim_fn=self._base_cdprim)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.bc.tab20_list[15]
        # first joint and link
        self.jlc.jnts[0].loc_pos = np.array([.0, .0, .2433])
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-2 * math.pi, 2 * math.pi])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link1.stl"))
        self.jlc.jnts[0].lnk.cmodel.rgba = rm.bc.tab20_list[15]
        # second joint and link
        self.jlc.jnts[1].loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(1.5708, -1.5708, 3.1416)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[1].motion_range = np.array([-2.61799, 2.61799])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link2.stl"),
                                                         cdprim_type=mcm.mc.CDPType.USER_DEFINED, ex_radius=.005,
                                                         userdef_cdprim_fn=self._link2_cdprim)
        self.jlc.jnts[1].lnk.cmodel.rgba = rm.bc.tab20_list[15]
        # third joint and link
        self.jlc.jnts[2].loc_pos = np.array([.2, .0, .0])
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(-3.1416, 0., 1.5708)
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[2].motion_range = np.array([-0.061087, 5.235988])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link3.stl"))
        self.jlc.jnts[2].lnk.cmodel.rgba = rm.bc.tab20_list[15]
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = np.array([.087, -.2276, .0])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(1.5708, 0., 0.)
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[3].motion_range = np.array([-2 * math.pi, 2 * math.pi])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link4.stl"),
                                                         cdprim_type=mcm.mc.CDPType.USER_DEFINED, ex_radius=.005,
                                                         userdef_cdprim_fn=self._link4_cdprim)
        self.jlc.jnts[3].lnk.cmodel.rgba = rm.bc.tab20_list[15]
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(1.5708, 0., 0.)
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[4].motion_range = np.array([-2.1642, 2.1642])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link5.stl"))
        self.jlc.jnts[4].lnk.cmodel.rgba = rm.bc.tab20_list[15]
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = np.array([.0, .0615, .0])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(-1.5708, 0., 0.)
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[5].motion_range = np.array([-2 * math.pi, 2 * math.pi])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link6.stl"))
        self.jlc.jnts[5].lnk.cmodel.rgba = rm.bc.tab20_list[15]
        self.jlc.finalize(ik_solver='d', identifier_str=name)
        self.jlc.finalize()
        # tcp
        self.loc_tcp_pos = np.array([0, 0, 0])
        self.loc_tcp_rotmat = np.eye(3)
        # set up cc
        if self.cc is not None:
            self.setup_cc()

    # self-defined collison model for the base link
    @staticmethod
    def _base_cdprim(ex_radius):
        pdcnd = CollisionNode("base")
        collision_primitive_c0 = CollisionBox(Point3(-0.008, 0, 0.0375),
                                              x=.07 + ex_radius, y=.065 + ex_radius, z=0.0375 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0, 0, .124),
                                              x=.043 + ex_radius, y=.043 + ex_radius, z=.049 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        cdprim = NodePath("user_defined_base")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    @staticmethod
    def _link2_cdprim(ex_radius):
        pdcnd = CollisionNode("link2")
        collision_primitive_c0 = CollisionBox(Point3(0, 0, 0.1065),
                                              x=.041 + ex_radius, y=.042 + ex_radius, z=0.0315 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.100, 0, 0.1065),
                                              x=.059 + ex_radius, y=.042 + ex_radius, z=0.0315 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(.2, 0, 0.0915),
                                              x=.041 + ex_radius, y=.042 + ex_radius, z=0.0465 + ex_radius)
        pdcnd.addSolid(collision_primitive_c2)
        cdprim = NodePath("user_defined_link2")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    @staticmethod
    def _link4_cdprim(ex_radius):
        pdcnd = CollisionNode("link4")
        collision_primitive_c0 = CollisionBox(Point3(0, 0, -0.124009),
                                              x=.041 + ex_radius, y=.042 + ex_radius, z=0.0682075 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0, -0.063315, -0.0503),
                                              x=.041 + ex_radius, y=.021315 + ex_radius, z=.087825 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        cdprim = NodePath("user_defined_link4")
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
        into_list = [lb, l0, l1]
        self.cc.set_cdpair_by_ids(from_list, into_list)


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    robot = XArmLite6(enable_cc=True)
    robot.gen_meshmodel(toggle_cdprim=True).attach_to(base)
    base.run()
    random_conf = robot.rand_conf()
    random_conf = np.array([0, 0, np.radians(180), 0, 0, 0])
    manipulator_instance.fk(random_conf)
    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    manipulator_instance.show_cdprim()
    tic = time.time()
    print(manipulator_instance.is_collided())
    toc = time.time()
    print(toc - tic)

    base.run()
