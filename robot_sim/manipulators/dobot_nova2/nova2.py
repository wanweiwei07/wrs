import os
import basis.robot_math as rm
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
            os.path.join(current_file_dir, "meshes", "base_link0.stl"),
            cdprim_type=mcm.mc.CDPType.USER_DEFINED,
            userdef_cdprim_fn=self._base_cdprim)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # first joint and link
        self.jlc.jnts[0].loc_pos = np.array([.0, .0, .2234])
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, -1])
        self.jlc.jnts[0].motion_range = np.array([-2 * np.pi, 2 * np.pi])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j1.stl"))
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # second joint and link
        self.jlc.jnts[1].loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(1.5708, 1.5708, 0)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[1].motion_range = np.array([-np.pi, np.pi])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j2.stl"))
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # third joint and link
        self.jlc.jnts[2].loc_pos = np.array([-.28, .0, .0])
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[2].motion_range = np.array([-2.79, 2.79])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j3.stl"))
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = np.array([-0.22501, .0, 0.1175])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(0, 0, -1.5708)
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 0, -1])
        self.jlc.jnts[3].motion_range = np.array([-2 * np.pi, 2 * np.pi])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j4.stl"))
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = np.array([.0, -0.12, .0])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(1.5708, .0, .0)
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 0, -1])
        self.jlc.jnts[4].motion_range = np.array([-2 * np.pi, 2 * np.pi])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j5.stl"))
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = np.array([0., 0.088004, 0])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(-1.5708, .0, .0)
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[5].motion_range = np.array([-2 * np.pi, 2 * np.pi])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j6.stl"))
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
    # arm.jlc._ik_solver.test_success_rate()
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
