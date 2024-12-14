import os
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim._kinematics.collision_checker as cc
import wrs.robot_sim.robots.dual_arm_robot_interface as dari
import wrs.robot_sim.robots.diana7_dual.diana7_rtq85 as d7r85
import wrs.robot_sim.robots.diana7_dual.diana7_sd as d7sd


class Diana7_Dual(dari.DualArmRobotInterface):
    """
    author: ziqi.xu, revised by weiwei
    date: 20221118, 20241031
    """

    def __init__(self, pos=rm.np.zeros(3), rotmat=rm.np.eye(3), name='diana7_dual', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        current_file_dir = os.path.dirname(__file__)
        # the body anchor
        self._body = rkjlc.rkjl.Anchor(name="diana7_dual_base", pos=self.pos, rotmat=self.rotmat, n_flange=2, n_lnk=1)
        self._body.loc_flange_pose_list[0] = [rm.np.array([.000559, -.615604, 0]), rm.np.eye(3)]
        self._body.loc_flange_pose_list[1] = [rm.np.array([-0.01165, 0.677502, 0]), rm.np.eye(3)]
        self._body.lnk_list[0].name = "diana7_dual_base"
        self._body.lnk_list[0].loc_pos = rm.np.array([-.453, 0, 0])
        self._body.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "base2.stl"), name="diana7_base2",
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED, userdef_cdprim_fn=self._base_cdprim)
        self._body.lnk_list[0].cmodel.rgba = rm.const.steel_gray
        # left arm
        self._lft_arm = d7sd.Diana7_SD(pos=self._body.gl_flange_pose_list[0][0],
                                      rotmat=self._body.gl_flange_pose_list[0][1], enable_cc=False)
        self._lft_arm.home_conf = rm.np.radians([0.0, -30.0, 0.0, 100.0, 0.0, -110.0, 0.0])
        self._lft_arm.manipulator.jlc.finalize(identifier_str=self._lft_arm.name + "_dual_lft")
        # right side
        self._rgt_arm = d7r85.Diana7_Rtq85(pos=self._body.gl_flange_pose_list[1][0],
                                          rotmat=self._body.gl_flange_pose_list[1][1], enable_cc=False)
        self._rgt_arm.home_conf = rm.np.radians([90.0, -30.0, 0.0, 100.0, 0.0, -110.0, 0.0])
        self._rgt_arm.manipulator.jlc.finalize(identifier_str=self._rgt_arm.name + "_dual_rgt")
        if self.cc is not None:
            self.setup_cc()
        # go home
        self.goto_home_conf()
        # set default delegator to left
        self.use_lft()

    @staticmethod
    def _base_cdprim(name="auto", ex_radius=None):
        pdcnd = mcm.CollisionNode(name + "_cnode")
        collision_primitive_c0 = mcm.CollisionBox(mcm.Point3(.55, 0, -0.05),
                                                  x=.55 + ex_radius, y=.85 + ex_radius, z=.05 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_b1 = mcm.CollisionBox(mcm.Point3(.055, -.43, 0.005),
                                                  x=.06 + ex_radius, y=.08 + ex_radius, z=.008 + ex_radius)
        pdcnd.addSolid(collision_primitive_b1)
        collision_primitive_b21 = mcm.CollisionBox(mcm.Point3(.055, .43, 0.005),
                                                   x=.06 + ex_radius, y=.08 + ex_radius, z=.008 + ex_radius)
        pdcnd.addSolid(collision_primitive_b21)
        collision_primitive_c1 = mcm.CollisionBox(mcm.Point3(.0225, -.43, 0.6),
                                                  x=.02 + ex_radius, y=.04 + ex_radius, z=.6 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        collision_primitive_c2 = mcm.CollisionBox(mcm.Point3(.0225, .43, 0.6),
                                                  x=.02 + ex_radius, y=.04 + ex_radius, z=.6 + ex_radius)
        pdcnd.addSolid(collision_primitive_c2)
        collision_primitive_l0 = mcm.CollisionBox(mcm.Point3(0.02, 0, 1.255),
                                                  x=.02 + ex_radius, y=.6 + ex_radius, z=.04 + ex_radius)
        pdcnd.addSolid(collision_primitive_l0)
        collision_primitive_r0 = mcm.CollisionBox(mcm.Point3(0.35, 0, 1.315),
                                                  x=.35 + ex_radius, y=.02 + ex_radius, z=.02 + ex_radius)
        pdcnd.addSolid(collision_primitive_r0)
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
        lft_mlb = self._lft_arm.cc.add_cce(self._lft_arm.manipulator.jlc.anchor.lnk_list[0])
        lft_ml0 = self._lft_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[0].lnk)
        lft_ml1 = self._lft_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[1].lnk)
        lft_ml2 = self._lft_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[2].lnk)
        lft_ml3 = self._lft_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[3].lnk)
        lft_ml4 = self._lft_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[4].lnk)
        lft_ml5 = self._lft_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[5].lnk)
        # lft_ml6 = self._lft_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[6].lnk)
        # right ee
        rgt_ee_cces = []
        for id, cdlnk in enumerate(self._rgt_arm.end_effector.cdelements):
            rgt_ee_cces.append(self._lft_arm.cc.add_cce(cdlnk))
        # right manipulator
        rgt_mlb = self._lft_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.anchor.lnk_list[0])
        rgt_ml0 = self._lft_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[0].lnk)
        rgt_ml1 = self._lft_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[1].lnk)
        rgt_ml2 = self._lft_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[2].lnk)
        rgt_ml3 = self._lft_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[3].lnk)
        rgt_ml4 = self._lft_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[4].lnk)
        rgt_ml5 = self._lft_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[5].lnk)
        # rgt_ml6 = self._lft_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[6].lnk)
        # first pairs (hand vs. others)
        from_list = lft_ee_cces + rgt_ee_cces
        into_list = [bd, lft_mlb, lft_ml0, lft_ml1, lft_ml2, lft_ml3, lft_ml4,
                     rgt_mlb, rgt_ml0, rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4]
        self._lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs (forearm vs. others)
        from_list = [lft_ml4, lft_ml5, rgt_ml4, rgt_ml5]
        into_list = [bd, lft_mlb, lft_ml0, lft_ml1, rgt_mlb, rgt_ml0, rgt_ml1]
        self._lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # third pairs (lft vs. rgt)
        from_list = [lft_ml4, lft_ml5] + lft_ee_cces
        into_list = [rgt_ml4, rgt_ml5] + rgt_ee_cces
        self._lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self._lft_arm.cc.enable_extcd_by_id_list(id_list=[lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces, type="from")
        self._lft_arm.cc.enable_innercd_by_id_list(
            id_list=[bd, lft_mlb, lft_ml0, lft_ml1, lft_ml2, lft_ml4, rgt_ml0, rgt_ml1, rgt_ml2, rgt_ml3,
                     rgt_ml4, rgt_ml5] + rgt_ee_cces, type="into")
        self._lft_arm.cc.dynamic_into_list = [bd]
        self._lft_arm.cc.dynamic_ext_list = lft_ee_cces[1:]

    def _enable_rgt_cc(self):
        self._rgt_arm.cc = cc.CollisionChecker("lft_arm_collision_checker")
        # body
        bd = self._rgt_arm.cc.add_cce(self._body.lnk_list[0])
        # left ee
        lft_ee_cces = []
        for id, cdlnk in enumerate(self._lft_arm.end_effector.cdelements):
            lft_ee_cces.append(self._rgt_arm.cc.add_cce(cdlnk))
        # left manipulator
        lft_mlb = self._rgt_arm.cc.add_cce(self._lft_arm.manipulator.jlc.anchor.lnk_list[0])
        lft_ml0 = self._rgt_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[0].lnk)
        lft_ml1 = self._rgt_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[1].lnk)
        lft_ml2 = self._rgt_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[2].lnk)
        lft_ml3 = self._rgt_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[3].lnk)
        lft_ml4 = self._rgt_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[4].lnk)
        lft_ml5 = self._rgt_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[5].lnk)
        # lft_ml6 = self._rgt_arm.cc.add_cce(self._lft_arm.manipulator.jlc.jnts[6].lnk)
        # right ee
        rgt_ee_cces = []
        for id, cdlnk in enumerate(self._rgt_arm.end_effector.cdelements):
            rgt_ee_cces.append(self._rgt_arm.cc.add_cce(cdlnk))
        # right manipulator
        rgt_mlb = self._rgt_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.anchor.lnk_list[0])
        rgt_ml0 = self._rgt_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[0].lnk)
        rgt_ml1 = self._rgt_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[1].lnk)
        rgt_ml2 = self._rgt_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[2].lnk)
        rgt_ml3 = self._rgt_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[3].lnk)
        rgt_ml4 = self._rgt_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[4].lnk)
        rgt_ml5 = self._rgt_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[5].lnk)
        # rgt_ml6 = self._rgt_arm.cc.add_cce(self._rgt_arm.manipulator.jlc.jnts[6].lnk)
        # first pairs (hand vs. others)
        from_list = lft_ee_cces + rgt_ee_cces
        into_list = [bd, lft_mlb, lft_ml0, lft_ml1, lft_ml2, lft_ml3, lft_ml4,
                     rgt_mlb, rgt_ml0, rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4]
        self._rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs (forearm vs. others)
        from_list = [lft_ml4, lft_ml5, rgt_ml4, rgt_ml5]
        into_list = [bd, lft_mlb, lft_ml0, lft_ml1, rgt_mlb, rgt_ml0, rgt_ml1]
        self._rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # third pairs (lft vs. rgt)
        from_list = [lft_ml4, lft_ml5] + lft_ee_cces
        into_list = [rgt_ml4, rgt_ml5] + rgt_ee_cces
        self._rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self._rgt_arm.cc.enable_extcd_by_id_list(id_list=[rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces, type="from")
        self._rgt_arm.cc.enable_innercd_by_id_list(
            id_list=[bd, rgt_mlb, rgt_ml0, rgt_ml1, rgt_ml2, rgt_ml4, lft_mlb, lft_ml0, lft_ml1, lft_ml2, lft_ml3,
                     lft_ml4, lft_ml5] + lft_ee_cces, type="into")
        self._rgt_arm.cc.dynamic_into_list = [bd]
        self._rgt_arm.cc.dynamic_ext_list = lft_ee_cces[1:]


if __name__ == '__main__':
    from wrs import wd, mgm

    this_dir, this_filename = os.path.split(__file__)

    base = wd.World(cam_pos=[5, 0, 3], lookat_pos=[0, 0, 1])
    mgm.gen_frame().attach_to(base)
    dia_dual = Diana7_Dual()
    dia_dual.gen_meshmodel(toggle_cdprim=True).attach_to(base)

    base.run()
