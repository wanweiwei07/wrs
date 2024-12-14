import os
import math
import copy
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.robots.dual_arm_robot_interface as dari
import wrs.modeling.model_collection as mmc
import wrs.robot_sim.robots.yumi.yumi_single_arm as ysa
import wrs.robot_sim._kinematics.collision_checker as cc


class Yumi(dari.DualArmRobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='yumi', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # the body anchor
        self._body = rkjlc.rkjl.Anchor(name="yumi_body", pos=self.pos, rotmat=self.rotmat, n_flange=2, n_lnk=9)
        self._body.loc_flange_pose_list[0] = [np.array([0.05355, 0.07250, 0.41492]),
                                              (rm.rotmat_from_euler(0.9781, -0.5716, 2.3180) @
                                               rm.rotmat_from_euler(0.0, 0.0, -np.pi))]
        self._body.loc_flange_pose_list[1] = [np.array([0.05355, -0.07250, 0.41492]),
                                              (rm.rotmat_from_euler(-0.9781, -0.5682, -2.3155) @
                                               rm.rotmat_from_euler(0.0, 0.0, -np.pi))]
        self._body.lnk_list[0].name = "yumi_body_main"
        self._body.lnk_list[0].cmodel = mcm.CollisionModel(initor=os.path.join(current_file_dir, "meshes", "body.stl"),
                                                           cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
                                                           userdef_cdprim_fn=self._base_cdprim)
        self._body.lnk_list[0].cmodel.rgba = rm.const.hug_gray
        # table
        self._body.lnk_list[1].name = "yumi_body_table_top"
        self._body.lnk_list[1].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "yumi_tablenotop.stl"), name="yumi_body_table_top")
        self._body.lnk_list[1].cmodel.rgba = rm.const.steel_gray
        # lft column
        self._body.lnk_list[2].name = "yumi_body_lft_column"
        self._body.lnk_list[2].loc_pos = np.array([-.327, -.24, -1.015])
        self._body.lnk_list[2].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "yumi_column60602100.stl"), name="yumi_body_lft_column")
        self._body.lnk_list[2].cmodel.rgba = rm.const.steel_gray
        # rgt column
        self._body.lnk_list[3].name = "yumi_body_rgt_column"
        self._body.lnk_list[3].loc_pos = np.array([-.327, .24, -1.015])
        self._body.lnk_list[3].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "yumi_column60602100.stl"), name="yumi_body_rgt_column")
        self._body.lnk_list[3].cmodel.rgba = rm.const.steel_gray
        # top back column
        self._body.lnk_list[4].name = "yumi_body_top_back_column"
        self._body.lnk_list[4].loc_pos = np.array([-.327, 0, 1.085])
        self._body.lnk_list[4].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "yumi_column6060540.stl"), name="yumi_body_top_back_column")
        self._body.lnk_list[4].cmodel.rgba = rm.const.steel_gray
        # top lft column
        self._body.lnk_list[5].name = "yumi_body_top_lft_column"
        self._body.lnk_list[5].loc_pos = np.array([-.027, -.24, 1.085])
        self._body.lnk_list[5].loc_rotmat = rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
        self._body.lnk_list[5].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "yumi_column6060540.stl"), name="yumi_body_top_lft_column")
        self._body.lnk_list[5].cmodel.rgba = rm.const.steel_gray
        # top rgt column
        self._body.lnk_list[6].name = "yumi_body_top_rgt_column"
        self._body.lnk_list[6].loc_pos = np.array([-.027, .24, 1.085])
        self._body.lnk_list[6].loc_rotmat = rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
        self._body.lnk_list[6].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "yumi_column6060540.stl"), name="yumi_body_top_rgt_column")
        self._body.lnk_list[6].cmodel.rgba = rm.const.steel_gray
        # top front column
        self._body.lnk_list[7].name = "yumi_body_top_front_column"
        self._body.lnk_list[7].loc_pos = np.array([.273, 0, 1.085])
        self._body.lnk_list[7].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "yumi_column6060540.stl"),
            name="yumi_body_top_front_column")
        self._body.lnk_list[7].cmodel.rgba = rm.const.steel_gray
        # phoxi
        self._body.lnk_list[8].name = "phoxi"
        self._body.lnk_list[8].loc_pos = np.array([.273, 0, 1.085])
        self._body.lnk_list[8].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "phoxi_m.stl"), name="yumi_phoxi")
        self._body.lnk_list[8].cmodel.rgba = rm.const.black
        # left arm
        self._lft_arm = ysa.YumiSglArm(pos=self._body.gl_flange_pose_list[0][0],
                                       rotmat=self._body.gl_flange_pose_list[0][1],
                                       name='yumi_lft_arm', enable_cc=False)
        self._lft_arm.home_conf = np.radians(np.array([20, -90, 120, 30, 0, 40, 0]))
        # right arm
        self._rgt_arm = ysa.YumiSglArm(pos=self._body.gl_flange_pose_list[1][0],
                                       rotmat=self._body.gl_flange_pose_list[1][1],
                                       name='yumi_rgt_arm', enable_cc=False)
        self._rgt_arm.home_conf = np.radians(np.array([-20, -90, -120, 30, .0, 40, 0]))
        if self.cc is not None:
            self.setup_cc()
        # go home
        self.goto_home_conf()
        # set default delegator to left
        self.use_lft()

    @staticmethod
    def _base_cdprim(name="auto", ex_radius=None):
        pdcnd = CollisionNode(name + "_cnode")
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
        cdprim = NodePath(name + "_cdprim")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    def _enable_lft_cc(self):
        self._lft_arm.cc = cc.CollisionChecker("lft_arm_collision_checker")
        # body
        bd = self._lft_arm.cc.add_cce(self._body.lnk_list[0])
        tbl = self._lft_arm.cc.add_cce(self._body.lnk_list[1])
        lc = self._lft_arm.cc.add_cce(self._body.lnk_list[2])
        rc = self._lft_arm.cc.add_cce(self._body.lnk_list[3])
        tbc = self._lft_arm.cc.add_cce(self._body.lnk_list[4])
        tlc = self._lft_arm.cc.add_cce(self._body.lnk_list[5])
        trc = self._lft_arm.cc.add_cce(self._body.lnk_list[6])
        tfc = self._lft_arm.cc.add_cce(self._body.lnk_list[7])
        phx = self._lft_arm.cc.add_cce(self._body.lnk_list[8])
        # left ee
        lft_ee_cces = []
        for id, cdlnk in enumerate(self._lft_arm.end_effector.cdelements):
            lft_ee_cces.append(self._lft_arm.cc.add_cce(cdlnk))
        # left manipulator
        lft_ml0 = self._lft_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[0].lnk)
        lft_ml1 = self._lft_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[1].lnk)
        lft_ml2 = self._lft_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[2].lnk)
        lft_ml3 = self._lft_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[3].lnk)
        lft_ml4 = self._lft_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[4].lnk)
        lft_ml5 = self._lft_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[5].lnk)
        # right ee
        rgt_ee_cces = []
        for id, cdlnk in enumerate(self._rgt_arm.end_effector.cdelements):
            rgt_ee_cces.append(self._lft_arm.cc.add_cce(cdlnk))
        # right manipulator
        rgt_ml0 = self._lft_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[0].lnk)
        rgt_ml1 = self._lft_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[1].lnk)
        rgt_ml2 = self._lft_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[2].lnk)
        rgt_ml3 = self._lft_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[3].lnk)
        rgt_ml4 = self._lft_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[4].lnk)
        rgt_ml5 = self._lft_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[5].lnk)
        # first pairs
        from_list = [lft_ml4, lft_ml5] + lft_ee_cces + [rgt_ml4, rgt_ml5] + rgt_ee_cces
        into_list = [bd, tbl, lc, rc, tbc, tlc, trc, tfc, phx, lft_ml0, rgt_ml0]
        self._lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs
        from_list = lft_ee_cces + rgt_ee_cces
        into_list = [lft_ml0, lft_ml1, rgt_ml0, rgt_ml1]
        self._lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # third pairs
        from_list = [lft_ml1, lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces
        into_list = [rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces
        self._lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self._lft_arm.cc.enable_extcd_by_id_list(
            id_list=[lft_ml1, lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces, type="from")
        self._lft_arm.cc.enable_innercd_by_id_list(
            id_list=[bd, tbl, lc, rc, tbc, tlc, trc, tfc, phx, lft_ml1, lft_ml2, lft_ml3, rgt_ml1, rgt_ml2, rgt_ml3,
                     rgt_ml4] + rgt_ee_cces, type="into")
        self._lft_arm.cc.dynamic_into_list = [tbl]
        self._lft_arm.cc.dynamic_ext_list = lft_ee_cces[1:]

    def _enable_rgt_cc(self):
        self._rgt_arm.cc = cc.CollisionChecker("rgt_arm_collision_checker")
        # body
        bd = self._rgt_arm.cc.add_cce(self._body.lnk_list[0])
        tbl = self._rgt_arm.cc.add_cce(self._body.lnk_list[1])
        lc = self._rgt_arm.cc.add_cce(self._body.lnk_list[2])
        rc = self._rgt_arm.cc.add_cce(self._body.lnk_list[3])
        tbc = self._rgt_arm.cc.add_cce(self._body.lnk_list[4])
        tlc = self._rgt_arm.cc.add_cce(self._body.lnk_list[5])
        trc = self._rgt_arm.cc.add_cce(self._body.lnk_list[6])
        tfc = self._rgt_arm.cc.add_cce(self._body.lnk_list[7])
        phx = self._rgt_arm.cc.add_cce(self._body.lnk_list[8])
        # left ee
        lft_ee_cces = []
        for id, cdlnk in enumerate(self._lft_arm.end_effector.cdelements):
            lft_ee_cces.append(self._rgt_arm.cc.add_cce(cdlnk))
        # left manipulator
        lft_ml0 = self._rgt_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[0].lnk)
        lft_ml1 = self._rgt_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[1].lnk)
        lft_ml2 = self._rgt_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[2].lnk)
        lft_ml3 = self._rgt_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[3].lnk)
        lft_ml4 = self._rgt_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[4].lnk)
        lft_ml5 = self._rgt_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[5].lnk)
        # right ee
        rgt_ee_cces = []
        for id, cdlnk in enumerate(self._rgt_arm.end_effector.cdelements):
            rgt_ee_cces.append(self._rgt_arm.cc.add_cce(cdlnk))
        # right manipulator
        rgt_ml0 = self._rgt_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[0].lnk)
        rgt_ml1 = self._rgt_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[1].lnk)
        rgt_ml2 = self._rgt_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[2].lnk)
        rgt_ml3 = self._rgt_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[3].lnk)
        rgt_ml4 = self._rgt_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[4].lnk)
        rgt_ml5 = self._rgt_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[5].lnk)
        # first pairs
        from_list = [lft_ml4, lft_ml5] + lft_ee_cces + [rgt_ml4, rgt_ml5] + rgt_ee_cces
        into_list = [bd, tbl, lc, rc, tbc, tlc, trc, tfc, phx, lft_ml0, rgt_ml0]
        self._rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs
        from_list = lft_ee_cces + rgt_ee_cces
        into_list = [lft_ml0, lft_ml1, rgt_ml0, rgt_ml1]
        self._rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # third pairs
        from_list = [lft_ml1, lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces
        into_list = [rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces
        self._rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self._rgt_arm.cc.enable_extcd_by_id_list(
            id_list=[rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces, type="from")
        self._rgt_arm.cc.enable_innercd_by_id_list(
            id_list=[bd, tbl, lc, rc, tbc, tlc, trc, tfc, phx, rgt_ml1, rgt_ml2, rgt_ml3, lft_ml1, lft_ml2, lft_ml3,
                     lft_ml4] + lft_ee_cces, type="into")
        self._rgt_arm.cc.dynamic_into_list = [tbl]
        self._rgt_arm.cc.dynamic_ext_list = rgt_ee_cces[1:]


if __name__ == '__main__':
    import time
    from wrs import wd, mgm

    base = wd.World(cam_pos=[3, 1, 1], lookat_pos=[0, 0, 0.5])
    mgm.gen_frame().attach_to(base)
    robot = Yumi(enable_cc=True)
    robot.gen_meshmodel(toggle_cdprim=True).attach_to(base)
    # robot.gen_stickmodel().attach_to(base)
    # robot.show_cdprim()
    base.run()

    # ik test
    tgt_pos = np.array([.6, .0, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    robot.use_rgt()
    jnt_values = robot.ik(tgt_pos, tgt_rotmat)
    robot.goto_given_conf(jnt_values=jnt_values)
    robot.gen_meshmodel().attach_to(base)
    base.run()

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
    robot.show_cdprim()
    # robot.lft_arm.show_cdprim()
    base.run()
