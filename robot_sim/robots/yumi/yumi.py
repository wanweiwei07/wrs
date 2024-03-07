import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.model_collection as mmc
import modeling.collision_model as mcm
import robot_sim._kinematics.jlchain as rkjlc
import robot_sim.robots.yumi.yumi_single_arm as ysa
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import robot_sim.robots.robot_interface as ri


class Yumi(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='yumi', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        #
        self.body = rkjlc.JLChain(name="yumi_body", pos=self.pos, rotmat=self.rotmat, n_dof=0)
        self.body.anchor.lnk.cmodel = mcm.CollisionModel(initor=os.path.join(current_file_dir, "meshes", "body.stl"),
                                                         cdprim_type=mcm.mc.CDPType.USER_DEFINED,
                                                         userdef_cdprim_fn=self._base_combined_cdnp)
        self.body.anchor.lnk.cmodel.rgba = rm.bc.hug_gray
        self.body.finalize(ik_solver=None)
        # left arm
        self._loc_lft_arm_pos = np.array([0.05355, 0.07250, 0.41492])
        self._loc_lft_arm_rotmat = (rm.rotmat_from_euler(0.9781, -0.5716, 2.3180) @
                                    rm.rotmat_from_euler(0.0, 0.0, -np.pi))
        self.lft_arm = ysa.YumiSglArm(pos=self.pos + self.rotmat @ self._loc_lft_arm_pos,
                                      rotmat=self.rotmat @ self._loc_lft_arm_rotmat,
                                      name='yumi_lft_arm', enable_cc=True)
        self.lft_arm.home_conf = np.radians(np.array([20, -90, 120, 30, 0, 40, 0]))
        self.lft_arm.userdef_is_collided_fn = self._lft_arm_is_collided
        # right arm
        self._loc_rgt_arm_pos = np.array([0.05355, -0.07250, 0.41492])
        self._loc_rgt_arm_rotmat = (rm.rotmat_from_euler(-0.9781, -0.5682, -2.3155) @
                                    rm.rotmat_from_euler(0.0, 0.0, -np.pi))
        self.rgt_arm = ysa.YumiSglArm(pos=self.pos + self.rotmat @ self._loc_rgt_arm_pos,
                                      rotmat=self.rotmat @ self._loc_rgt_arm_rotmat,
                                      name='yumi_rgt_arm', enable_cc=True)
        self.rgt_arm.home_conf = np.radians(np.array([-20, -90, -120, 30, .0, 40, 0]))
        self.rgt_arm.userdef_is_collided_fn = self._rgt_arm_is_collided
        if enable_cc:
            self.setup_cc()
        # go home
        self.goto_home_conf()

    def _base_combined_cdnp(self, name="auto", ex_radius=None):
        pdcnd = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-.2, 0, 0.04),
                                              x=.16 + ex_radius, y=.2 + ex_radius, z=.04 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(-.24, 0, 0.24),
                                              x=.12 + ex_radius, y=.125 + ex_radius, z=.24 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(-.07, 0, 0.4),
                                              x=.075 + ex_radius, y=.125 + ex_radius, z=.06 + ex_radius)
        pdcnd.addSolid(collision_primitive_c2)
        collision_primitive_l0 = CollisionBox(Point3(0, 0.145, 0.03),
                                              x=.135 + ex_radius, y=.055 + ex_radius, z=.03 + ex_radius)
        pdcnd.addSolid(collision_primitive_l0)
        collision_primitive_r0 = CollisionBox(Point3(0, -0.145, 0.03),
                                              x=.135 + ex_radius, y=.055 + ex_radius, z=.03 + ex_radius)
        pdcnd.addSolid(collision_primitive_r0)
        cdprim = NodePath("user_defined")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    def _lft_arm_is_collided(self, cc, obstacle_list=[], other_robot_list=[], toggle_contacts=False):
        cc.is_collided(obstacle_list=obstacle_list + [self.body.anchor.lnk.cmodel],
                       other_robot_list=other_robot_list + [self.rgt_arm],
                       toggle_contacts=toggle_contacts)

    def _rgt_arm_is_collided(self, cc, obstacle_list=[], other_robot_list=[], toggle_contacts=False):
        cc.is_collided(obstacle_list=obstacle_list + [self.body.anchor.lnk.cmodel],
                       other_robot_list=other_robot_list + [self.lft_arm],
                       toggle_contacts=toggle_contacts)

    def setup_cc(self):
        # body
        body_l0 = self.cc.add_cce(self.body.anchor.lnk)
        # left ee
        lft_elb = self.cc.add_cce(self.lft_arm.end_effector.jlc.anchor.lnk)
        lft_el0 = self.cc.add_cce(self.lft_arm.end_effector.jlc.jnts[0].lnk)
        lft_el1 = self.cc.add_cce(self.lft_arm.end_effector.jlc.jnts[1].lnk)
        # left manipulator
        lft_ml0 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[0].lnk)
        lft_ml1 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[1].lnk)
        lft_ml2 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[2].lnk)
        lft_ml3 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[3].lnk)
        lft_ml4 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[4].lnk)
        lft_ml5 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[5].lnk)
        # right ee
        rgt_elb = self.cc.add_cce(self.rgt_arm.end_effector.jlc.anchor.lnk)
        rgt_el0 = self.cc.add_cce(self.rgt_arm.end_effector.jlc.jnts[0].lnk)
        rgt_el1 = self.cc.add_cce(self.rgt_arm.end_effector.jlc.jnts[1].lnk)
        # right manipulator
        rgt_ml0 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[0].lnk)
        rgt_ml1 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[1].lnk)
        rgt_ml2 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[2].lnk)
        rgt_ml3 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[3].lnk)
        rgt_ml4 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[4].lnk)
        rgt_ml5 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[5].lnk)
        # first pairs
        from_list = [body_l0, lft_ml0, lft_ml0]
        into_list = [lft_ml4, lft_ml5, lft_elb, lft_el0, lft_el1, rgt_ml4, rgt_ml5, rgt_elb, rgt_el0, rgt_el1]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs
        from_list = [lft_ml0, lft_ml1, rgt_ml0, rgt_ml1]
        into_list = [lft_elb, lft_el0, lft_el1, rgt_elb, rgt_el0, rgt_el1]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        # third pairs
        from_list = [lft_ml1, lft_ml2, lft_ml3, lft_ml4, lft_ml5, lft_elb, lft_el0, lft_el1]
        into_list = [rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5, rgt_elb, rgt_el0, rgt_el1]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.body.fix_to(self.pos, self.rotmat)
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

    def goto_home_conf(self):
        self.lft_arm.goto_home_conf()
        self.rgt_arm.goto_home_conf()

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

    def is_collided(self, obstacle_list=[], other_robot_list=[], toggle_contacts=False):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        :param obstacle_list:
        :param other_robot_list:
        :param toggle_contacts: debug
        :return: see CollisionChecker is_collided for details
        author: weiwei
        date: 20240307
        """
        collision_info = self.cc.is_collided(obstacle_list=obstacle_list,
                                             other_robot_list=other_robot_list,
                                             toggle_contacts=toggle_contacts)
        return collision_info

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name='yumi_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.body.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                 toggle_flange_frame=toggle_flange_frame,
                                 name=name + "_body").attach_to(m_col)
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
        self.body.gen_meshmodel(rgb=rgb, alpha=alpha, toggle_flange_frame=toggle_flange_frame,
                                toggle_jnt_frames=toggle_jnt_frames, toggle_cdprim=toggle_cdprim,
                                toggle_cdmesh=toggle_cdmesh, name=name + "_body").attach_to(m_col)
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
    robot = Yumi(enable_cc=True)
    robot.gen_meshmodel().attach_to(base)
    # robot.show_cdprim()
    # base.run()

    # ik test
    tgt_pos = np.array([.4, -.4, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    # base.run()

    tic = time.time()
    jnt_values = robot.rgt_arm.ik(tgt_pos, tgt_rotmat)
    toc = time.time()
    print(toc - tic)
    if jnt_values is not None:
        robot.rgt_arm.goto_given_conf(jnt_values=jnt_values)
        robot.gen_meshmodel().attach_to(base)
    tic = time.time()
    result = robot.is_collided()
    toc = time.time()
    print(result, toc - tic)
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
    yumi_instance.hold(objcm=objcm_copy, jaw_width=0.03, hnd_name='lft_hnd')
    tgt_pos = np.array([.4, .5, .4])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 3)
    jnt_values = yumi_instance.ik(component_name, tgt_pos, tgt_rotmat)
    yumi_instance.fk(component_name, jnt_values)
    yumi_instance.show_cdprimit()
    yumi_meshmodel = yumi_instance.gen_meshmodel()
    yumi_meshmodel.attach_to(base)

    base.run()
