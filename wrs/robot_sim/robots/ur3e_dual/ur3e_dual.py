import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim._kinematics.collision_checker as cc
import wrs.robot_sim.robots.dual_arm_robot_interface as dari
import wrs.robot_sim.robots.ur3e_dual.ur3e_rtqhe as u3ehe


class UR3e_Dual(dari.DualArmRobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='ur3e_dual', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # the body anchor
        self._body = rkjlc.rkjl.Anchor(name=self.name + "_anchor", pos=self.pos, rotmat=self.rotmat, n_flange=2,
                                       n_lnk=9)
        self._body.loc_flange_pose_list[0] = [np.array([.365, .345, 1.33]),
                                              rm.rotmat_from_euler(-np.pi / 2.0, 0, -np.pi / 2.0)]
        self._body.loc_flange_pose_list[1] = [np.array([.365, -.345, 1.33]),
                                              rm.rotmat_from_euler(-np.pi / 2.0, 0, -np.pi / 2.0)]
        self._body.lnk_list[0].name = "ur3e_dual_base_link"
        self._body.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "ur3e_dual_base.stl"), name=self.name + "_body",
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
            userdef_cdprim_fn=self._base_cdprim)
        self._body.lnk_list[0].cmodel.rgba = rm.const.hug_gray
        # left arm
        self._lft_arm = u3ehe.UR3e_RtqHE(pos=self._body.gl_flange_pose_list[0][0],
                                         rotmat=self._body.gl_flange_pose_list[0][1],
                                         ik_solver=None, enable_cc=False)
        self._lft_arm.home_conf = np.array([-np.pi * 2 / 3, -np.pi * 2 / 3, np.pi * 2 / 3, np.pi, -np.pi / 2, 0])
        self._lft_arm.manipulator.jnts[0].motion_range = np.array([-np.pi * 5 / 3, -np.pi / 3])
        self._lft_arm.manipulator.jnts[1].motion_range = np.array([-np.pi, 0])
        self._lft_arm.manipulator.jnts[2].motion_range = np.array([0, np.pi])
        self._lft_arm.manipulator.jnts[3].motion_range = np.array([np.pi / 6, np.pi * 7 / 6])
        self._lft_arm.manipulator.jnts[4].motion_range = np.array([-np.pi, np.pi])
        self._lft_arm.manipulator.jnts[5].motion_range = np.array([-np.pi, np.pi])
        self._lft_arm.manipulator.jlc.finalize(identifier_str=self._lft_arm.name + "_dual_lft")
        # right side
        self._rgt_arm = u3ehe.UR3e_RtqHE(pos=self._body.gl_flange_pose_list[1][0],
                                         rotmat=self._body.gl_flange_pose_list[1][1],
                                         ik_solver=None, enable_cc=False)
        self._rgt_arm.home_conf = np.array([np.pi * 2 / 3, -np.pi / 3, -np.pi * 2 / 3, 0, np.pi / 2, 0])
        self._rgt_arm.manipulator.jnts[0].motion_range = np.array([np.pi / 3, np.pi * 5 / 3])
        self._rgt_arm.manipulator.jnts[1].motion_range = np.array([-np.pi, 0])
        self._rgt_arm.manipulator.jnts[2].motion_range = np.array([-np.pi, 0])
        self._rgt_arm.manipulator.jnts[3].motion_range = np.array([-np.pi * 5 / 6, np.pi / 6])
        self._rgt_arm.manipulator.jnts[4].motion_range = np.array([-np.pi, np.pi])
        self._rgt_arm.manipulator.jnts[5].motion_range = np.array([-np.pi, np.pi])
        self._rgt_arm.manipulator.jlc.finalize(identifier_str=self._rgt_arm.name + "_dual_rgt")
        if self.cc is not None:
            self.setup_cc()
        # go home
        self.goto_home_conf()
        # set default delegator to left
        self.use_lft()

    @staticmethod
    def _base_cdprim(name="ur3e_dual_base", ex_radius=None):
        pdcnd = mcm.CollisionNode(name + "_cnode")
        collision_primitive_c0 = mcm.CollisionBox(mcm.Point3(0.54, 0.0, 0.39),
                                                  x=.54 + ex_radius, y=.6 + ex_radius, z=.39 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = mcm.CollisionBox(mcm.Point3(0.06, 0.0, 0.9),
                                                  x=.06 + ex_radius, y=.375 + ex_radius, z=.9 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        collision_primitive_c2 = mcm.CollisionBox(mcm.Point3(0.18, 0.0, 1.77),
                                                  x=.18 + ex_radius, y=.21 + ex_radius, z=.03 + ex_radius)
        pdcnd.addSolid(collision_primitive_c2)
        collision_primitive_l0 = mcm.CollisionBox(mcm.Point3(0.2425, 0.345, 1.33),
                                                  x=.1225 + ex_radius, y=.06 + ex_radius, z=.06 + ex_radius)
        pdcnd.addSolid(collision_primitive_l0)
        collision_primitive_r0 = mcm.CollisionBox(mcm.Point3(0.2425, -0.345, 1.33),
                                                  x=.1225 + ex_radius, y=.06 + ex_radius, z=.06 + ex_radius)
        pdcnd.addSolid(collision_primitive_r0)
        collision_primitive_l1 = mcm.CollisionBox(mcm.Point3(0.21, 0.405, 1.07),
                                                  x=.03 + ex_radius, y=.06 + ex_radius, z=.29 + ex_radius)
        pdcnd.addSolid(collision_primitive_l1)
        collision_primitive_r1 = mcm.CollisionBox(mcm.Point3(0.21, -0.405, 1.07),
                                                  x=.03 + ex_radius, y=.06 + ex_radius, z=.29 + ex_radius)
        pdcnd.addSolid(collision_primitive_r1)
        cdprim = mcm.NodePath(name + "_cdprim")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    def _enable_lft_cc(self):
        self._lft_arm.cc = cc.CollisionChecker("lft_arm_collision_checker")
        # body
        bd = self._lft_arm.cc.add_cce(self._body.lnk_list[0])
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
        from_list = lft_ee_cces + rgt_ee_cces
        into_list = [bd, lft_ml0, lft_ml1, lft_ml2, rgt_ml0, rgt_ml1, rgt_ml2]
        self._lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs
        from_list = [lft_ml2, rgt_ml2]
        into_list = [bd, lft_ml0, rgt_ml0]
        self._lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # third pairs
        from_list = [lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces
        into_list = [rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces
        self._lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self._lft_arm.cc.enable_extcd_by_id_list(
            id_list=[lft_ml1, lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces, type="from")
        self._lft_arm.cc.enable_innercd_by_id_list(
            id_list=[bd, lft_ml0, lft_ml1, lft_ml2, lft_ml3, rgt_ml0, rgt_ml1, rgt_ml2, rgt_ml3,
                     rgt_ml4, rgt_ml5] + rgt_ee_cces, type="into")
        self._lft_arm.cc.dynamic_into_list = [bd]
        self._lft_arm.cc.dynamic_ext_list = lft_ee_cces[1:]

    def _enable_rgt_cc(self):
        self._rgt_arm.cc = cc.CollisionChecker("rgt_arm_collision_checker")
        # body
        bd = self._rgt_arm.cc.add_cce(self._body.lnk_list[0])
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
        from_list = lft_ee_cces + rgt_ee_cces
        into_list = [bd, lft_ml0, lft_ml1, lft_ml2, rgt_ml0, rgt_ml1, rgt_ml2]
        self._rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs
        from_list = [lft_ml2, rgt_ml2]
        into_list = [bd, lft_ml0, rgt_ml0]
        self._rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # third pairs
        from_list = [lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces
        into_list = [rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces
        self._rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self._rgt_arm.cc.enable_extcd_by_id_list(
            id_list=[rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces, type="from")
        self._rgt_arm.cc.enable_innercd_by_id_list(
            id_list=[bd, rgt_ml0, rgt_ml1, rgt_ml2, rgt_ml3, lft_ml0, lft_ml1, lft_ml2, lft_ml3, lft_ml4,
                     lft_ml5] + lft_ee_cces, type="into")
        self._rgt_arm.cc.dynamic_into_list = [bd]
        self._rgt_arm.cc.dynamic_ext_list = rgt_ee_cces[1:]


if __name__ == '__main__':
    import time
    import wrs.visualization.panda.world as wd
    from tqdm import tqdm

    base = wd.World(cam_pos=[3, 3, 3], lookat_pos=[0, 0, 1])
    mcm.mgm.gen_frame().attach_to(base)
    robot = UR3e_Dual(enable_cc=True)
    robot.gen_meshmodel(alpha=.5).attach_to(base)
    robot.gen_stickmodel().attach_to(base)
    robot.use_rgt()
    # robot.delegator.manipulator.jlc._ik_solver.test_success_rate()
    base.run()

    count = 0
    # ik test
    for i in tqdm(range(100)):
        rand_conf = robot.rand_conf()
        # print(rand_conf, robot.delegator.manipulator.jnt_ranges)
        tgt_pos, tgt_rotmat = robot.fk(jnt_values=rand_conf)
        # tgt_pos = np.array([.8, -.1, .9])
        # tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], np.pi)
        mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat, ax_length=.3).attach_to(base)

        tic = time.time()
        jnt_values = robot.ik(tgt_pos, tgt_rotmat, toggle_dbg=False)
        print(jnt_values)
        toc = time.time()
        if jnt_values is not None:
            count += 1
            robot.goto_given_conf(jnt_values=jnt_values)
            robot.gen_meshmodel().attach_to(base)
            # base.run()
        # tic = time.time()
        # result = robot.is_collided()
        # toc = time.time()
        # print(result, toc - tic)
        # robot.show_cdprim()
        print(count)
    base.run()
