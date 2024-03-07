import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.model_collection as mmc
import modeling.collision_model as mcm
import robot_sim._kinematics.jlchain as rkjlc
import robot_sim.robots.yumi.yumi_single_arm as ysa
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import robot_sim.system.system_interface as ri


class Yumi(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='yumi', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        self.body = rkjlc.JLChain(name="yumi_body", pos=self.pos, rotmat=self.rotmat, n_dof=0)
        self.body.anchor.lnk = mcm.CollisionModel(initor=os.path.join(current_file_dir, "meshes", "body.stl"),
                                                  cdprim_type=mcm.mc.CDPType.USER_DEFINED,
                                                  userdef_cdprim_fn=self._base_combined_cdnp)
        self.body.finalize(ik_solver=None)
        # left arm
        self._loc_lft_arm_pos = np.array([0.05355, 0.07250, 0.41492])
        self._loc_lft_arm_rotmat = rm.rotmat_from_euler(0.9781, -0.5716, 2.3180)
        self.lft_arm = ysa.YumiSglArm(pos=self.pos + self.rotmat @ self._loc_lft_arm_pos,
                                      rotmat=self.rotmat @ self._loc_lft_arm_rotmat,
                                      name='yumi_lft_arm', enable_cc=False)
        self.lft_arm.home_conf = np.radians(np.array([20, -90, 120, 30, 0, 40, 0]))
        # right arm
        self._loc_rgt_arm_pos = np.array([0.05355, 0.07250, 0.41492])
        self._loc_rgt_arm_rotmat = rm.rotmat_from_euler(0.9781, -0.5716, 2.3180)
        self.rgt_arm = ysa.YumiSglArm(pos=self.pos + self.rotmat @ self._loc_rgt_arm_pos,
                                      rotmat=self.rotmat @ self._loc_rgt_arm_rotmat,
                                      name='yumi_lft_arm', enable_cc=False)
        self.rgt_arm.home_conf = np.radians(np.array([-20, -90, -120, 30, .0, 40, 0]))
        # collision detection
        if enable_cc:
            self.setup_cc()

    def _base_combined_cdnp(name, radius):
        pdcnd = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-.2, 0, 0.04),
                                              x=.16 + radius, y=.2 + radius, z=.04 + radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(-.24, 0, 0.24),
                                              x=.12 + radius, y=.125 + radius, z=.24 + radius)
        pdcnd.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(-.07, 0, 0.4),
                                              x=.075 + radius, y=.125 + radius, z=.06 + radius)
        pdcnd.addSolid(collision_primitive_c2)
        collision_primitive_l0 = CollisionBox(Point3(0, 0.145, 0.03),
                                              x=.135 + radius, y=.055 + radius, z=.03 + radius)
        pdcnd.addSolid(collision_primitive_l0)
        collision_primitive_r0 = CollisionBox(Point3(0, -0.145, 0.03),
                                              x=.135 + radius, y=.055 + radius, z=.03 + radius)
        pdcnd.addSolid(collision_primitive_r0)
        cdprim = NodePath("user_defined")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    def setup_cc(self):
        # TODO when pose is changed, oih info goes wrong
        return
        # self.cc.add_cdlnks(self.lft_body, [0, 1, 2, 3, 4, 5, 6, 7])
        # self.cc.add_cdlnks(self.lft_arm, [1, 2, 3, 4, 5, 6])
        # self.cc.add_cdlnks(self.lft_hnd.lft, [0, 1])
        # self.cc.add_cdlnks(self.lft_hnd.rgt, [1])
        # self.cc.add_cdlnks(self.rgt_arm, [1, 2, 3, 4, 5, 6])
        # self.cc.add_cdlnks(self.rgt_hnd.lft, [0, 1])
        # self.cc.add_cdlnks(self.rgt_hnd.rgt, [1])
        # activelist = [self.lft_arm.lnks[1],
        #               self.lft_arm.lnks[2],
        #               self.lft_arm.lnks[3],
        #               self.lft_arm.lnks[4],
        #               self.lft_arm.lnks[5],
        #               self.lft_arm.lnks[6],
        #               self.lft_hnd.lft.lnks[0],
        #               self.lft_hnd.lft.lnks[1],
        #               self.lft_hnd.rgt.lnks[1],
        #               self.rgt_arm.lnks[1],
        #               self.rgt_arm.lnks[2],
        #               self.rgt_arm.lnks[3],
        #               self.rgt_arm.lnks[4],
        #               self.rgt_arm.lnks[5],
        #               self.rgt_arm.lnks[6],
        #               self.rgt_hnd.lft.lnks[0],
        #               self.rgt_hnd.lft.lnks[1],
        #               self.rgt_hnd.rgt.lnks[1]]
        # self.cc.set_active_cdlnks(activelist)
        # fromlist = [self.lft_body.lnks[0],  # table
        #             self.lft_body.lnks[1],  # body
        #             self.lft_arm.lnks[1],
        #             self.rgt_arm.lnks[1]]
        # intolist = [self.lft_arm.lnks[5],
        #             self.lft_arm.lnks[6],
        #             self.lft_hnd.lft.lnks[0],
        #             self.lft_hnd.lft.lnks[1],
        #             self.lft_hnd.rgt.lnks[1],
        #             self.rgt_arm.lnks[5],
        #             self.rgt_arm.lnks[6],
        #             self.rgt_hnd.lft.lnks[0],
        #             self.rgt_hnd.lft.lnks[1],
        #             self.rgt_hnd.rgt.lnks[1]]
        # self.cc.set_cdpair(fromlist, intolist)
        # fromlist = [self.lft_arm.lnks[1],
        #             self.lft_arm.lnks[2],
        #             self.rgt_arm.lnks[1],
        #             self.rgt_arm.lnks[2]]
        # intolist = [self.lft_hnd.lft.lnks[0],
        #             self.lft_hnd.lft.lnks[1],
        #             self.lft_hnd.rgt.lnks[1],
        #             self.rgt_hnd.lft.lnks[0],
        #             self.rgt_hnd.lft.lnks[1],
        #             self.rgt_hnd.rgt.lnks[1]]
        # self.cc.set_cdpair(fromlist, intolist)
        # fromlist = [self.lft_arm.lnks[2],
        #             self.lft_arm.lnks[3],
        #             self.lft_arm.lnks[4],
        #             self.lft_arm.lnks[5],
        #             self.lft_arm.lnks[6],
        #             self.lft_hnd.lft.lnks[0],
        #             self.lft_hnd.lft.lnks[1],
        #             self.lft_hnd.rgt.lnks[1]]
        # intolist = [self.rgt_arm.lnks[2],
        #             self.rgt_arm.lnks[3],
        #             self.rgt_arm.lnks[4],
        #             self.rgt_arm.lnks[5],
        #             self.rgt_arm.lnks[6],
        #             self.rgt_hnd.lft.lnks[0],
        #             self.rgt_hnd.lft.lnks[1],
        #             self.rgt_hnd.rgt.lnks[1]]
        # self.cc.set_cdpair(fromlist, intolist)

    def fix_to(self, pos, rotmat):
        self.body.fix_to(pos, rotmat)
        self.pos = pos
        self.rotmat = rotmat
        self.lft_body.fix_to(self.pos, self.rotmat)
        self.lft_arm.fix_to(pos=self.pos + self.rotmat @ self._loc_lft_arm_pos,
                            rotmat=self.rotmat @ self._loc_lft_arm_rotmat)
        self.rgt_arm.fix_to(pos=self.pos + self.rotmat @ self._loc_rgt_arm_pos,
                            rotmat=self.rotmat @ self._loc_rgt_arm_rotmat)

    def goto_given_conf(self, jnt_values):
        """
        :param jnt_values: nparray 1x14, 0:7lft, 7:14rgt
        :return:
        author: weiwei
        date: 20240307
        """
        if len(jnt_values) != self.lft_arm.manipulator.n_dof + self.rgt_arm.manipulator.n_dof:
            raise ValueError("The given joint values do not match total n_dof")
        self.lft_arm.goto_given_conf(jnt_values=jnt_values[:self.lft_arm.manipulator.n_dof])
        self.rgt_arm.goto_given_conf(jnt_values=jnt_values[self.rgt_arm.manipulator.n_dof:])

    def get_jnt_values(self):
        return self.lft_arm.get_jnt_values() + self.rgt_arm.get_jnt_values()

    def rand_conf(self):
        """
        :return:
        author: weiwei
        date: 20210406
        """
        return self.lft_arm.rand_conf() + self.rgt_arm.rand_conf()

    def are_jnts_in_ranges(self, jnt_values):
        return self.lft_arm.are_jnts_in_ranges(
            jnt_values=jnt_values[:self.lft_arm.manipulator.n_dof]) and self.rgt_arm.are_jnts_in_ranges(
            jnt_values=jnt_values[self.rgt_arm.manipulator.n_dof:])

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name='yumi_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.lft_arm.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                    toggle_jnt_frames=toggle_jnt_frames,
                                    toggle_flange_frame=toggle_flange_frame,
                                    name=name + "_lft_arm").attach_to(m_col)
        self.rtt_arm.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                    toggle_jnt_frames=toggle_jnt_frames,
                                    toggle_flange_frame=toggle_flange_frame,
                                    name=name + "_rgt_arm").attach_to(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='yumi_meshmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.body
        self.lft_arm.gen_meshmodel(rgb=rgb,
                                   alpha=alpha,
                                   toggle_tcp_frame=toggle_tcp_frame,
                                   toggle_jnt_frames=toggle_jnt_frames,
                                   toggle_flange_frame=toggle_flange_frame,
                                   toggle_cdprim=toggle_cdprim,
                                   toggle_cdmesh=toggle_cdmesh,
                                   name=name + "_lft_arm").attach_to(m_col)
        self.rgt_arm.gen_meshmodel(rgb=rgb,
                                   alpha=alpha,
                                   toggle_tcp_frame=toggle_tcp_frame,
                                   toggle_jnt_frames=toggle_jnt_frames,
                                   toggle_flange_frame=toggle_flange_frame,
                                   toggle_cdprim=toggle_cdprim,
                                   toggle_cdmesh=toggle_cdmesh,
                                   name=name + "_rgt_arm").attach_to(m_col)
        return m_col


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import basis

    base = wd.World(cam_pos=[3, 1, 1], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)
    yumi_instance = Yumi(enable_cc=True)
    yumi_meshmodel = yumi_instance.gen_meshmodel()
    yumi_meshmodel.attach_to(base)
    # yumi_instance.show_cdprimit()
    base.run()

    # ik test
    component_name = 'rgt_arm'
    tgt_pos = np.array([.4, -.4, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    tic = time.time()
    jnt_values = yumi_instance.ik(component_name, tgt_pos, tgt_rotmat)
    toc = time.time()
    print(toc - tic)
    yumi_instance.fk(component_name, jnt_values)
    yumi_meshmodel = yumi_instance.gen_meshmodel()
    yumi_meshmodel.attach_to(base)
    yumi_instance.gen_stickmodel().attach_to(base)
    tic = time.time()
    result = yumi_instance.is_collided()
    toc = time.time()
    print(result, toc - tic)

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
    yumi_instance.hold(objcm=objcm_copy, jaw_width=0.03, hnd_name='lft_hnd')
    tgt_pos = np.array([.4, .5, .4])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 3)
    jnt_values = yumi_instance.ik(component_name, tgt_pos, tgt_rotmat)
    yumi_instance.fk(component_name, jnt_values)
    yumi_instance.show_cdprimit()
    yumi_meshmodel = yumi_instance.gen_meshmodel()
    yumi_meshmodel.attach_to(base)

    base.run()
