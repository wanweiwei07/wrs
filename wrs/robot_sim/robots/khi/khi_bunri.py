import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim._kinematics.collision_checker as cc
import wrs.robot_sim.robots.dual_arm_robot_interface as dari
import wrs.robot_sim.robots.khi.khi_or2fg7 as kg
import wrs.robot_sim.robots.khi.khi_orsd as ksd


class KHI_DUAL(dari.DualArmRobotInterface):
    """
    author: weiwei
    date: 20230805 Toyonaka
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='khi_dual', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        lft_arm_homeconf = np.radians(np.array([0, -30, -120, 0, -60, 0]))
        rgt_arm_homeconf = np.radians(np.array([0, -30, -120, 0, -60, 0]))
        # the body anchor
        self._body = rkjlc.rkjl.Anchor(name=self.name + "_anchor", pos=self.pos, rotmat=self.rotmat, n_flange=2,
                                       n_lnk=1)
        self._body.loc_flange_pose_list[0] = [np.array([0, .25, .0]), rm.rotmat_from_euler(0, 0, -np.pi / 2)]
        self._body.loc_flange_pose_list[1] = [np.array([0, -.25, 0]), rm.rotmat_from_euler(0, 0, -np.pi / 2)]
        self._body.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "base_bunri.stl"), name=self.name + "_body",
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
            userdef_cdprim_fn=self._base_cdprim)
        # lft
        self._lft_arm = kg.KHI_OR2FG7(pos=self._body.gl_flange_pose_list[0][0],
                                      rotmat=self._body.gl_flange_pose_list[0][1], enable_cc=False)
        self._lft_arm.home_conf=lft_arm_homeconf
        # self._lft_arm.manipulator.jnts[0].motion_range = rm.degrees(rm.vec(-180, 110))
        # self._lft_arm.manipulator.jnts[1].motion_range = rm.degrees(rm.vec(-40, 90))
        # self._lft_arm.manipulator.jnts[2].motion_range = rm.degrees(rm.vec(-157, -20))
        # self._lft_arm.manipulator.jnts[3].motion_range = rm.degrees(rm.vec(-100, 100))
        # self._lft_arm.manipulator.jnts[4].motion_range = rm.degrees(rm.vec(-125, 0))
        # self._lft_arm.manipulator.jnts[5].motion_range = rm.degrees(rm.vec(-150, 130))
        self._lft_arm.manipulator.jlc.finalize(identifier_str=self._lft_arm.name + "_dual_lft")
        # rgt
        self._rgt_arm = ksd.KHI_ORSD(pos=self._body.gl_flange_pose_list[1][0],
                                     rotmat=self._body.gl_flange_pose_list[1][1], enable_cc=False)
        self._rgt_arm.home_conf=rgt_arm_homeconf
        # self._rgt_arm.manipulator.jnts[0].motion_range = rm.degrees(rm.vec(-110, 180))
        # self._rgt_arm.manipulator.jnts[1].motion_range = rm.degrees(rm.vec(-40, 90))
        # self._rgt_arm.manipulator.jnts[2].motion_range = rm.degrees(rm.vec(-157, -20))
        # self._rgt_arm.manipulator.jnts[3].motion_range = rm.degrees(rm.vec(-40, 150))
        # self._rgt_arm.manipulator.jnts[4].motion_range = rm.degrees(rm.vec(-125, 0))
        # self._rgt_arm.manipulator.jnts[5].motion_range = rm.degrees(rm.vec(-240, 240))
        self._rgt_arm.manipulator.jlc.finalize(identifier_str=self._lft_arm.name + "_dual_rgt")
        # collision detection
        if self.cc is not None:
            self.setup_cc()
        # go home
        self.goto_home_conf()
        # set default delegator to left
        self.use_lft()

    @staticmethod
    def _base_cdprim(name="ur3e_dual_base", ex_radius=None):
        pdcnd = mcm.CollisionNode(name + "_cnode")
        collision_primitive_t = mcm.CollisionBox(mcm.Point3(0.2, 0.0, -0.367),
                                                  x=.6 + ex_radius, y=.9 + ex_radius, z=.385 + ex_radius)
        pdcnd.addSolid(collision_primitive_t)
        collision_primitive_c0 = mcm.CollisionBox(mcm.Point3(0.77, 0.87, .533),
                                                  x=.03 + ex_radius, y=.03 + ex_radius, z=.515 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = mcm.CollisionBox(mcm.Point3(0.77, -0.87, .533),
                                                  x=.03 + ex_radius, y=.03 + ex_radius, z=.515 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        collision_primitive_c2 = mcm.CollisionBox(mcm.Point3(-0.37, 0.87, .533),
                                                  x=.03 + ex_radius, y=.03 + ex_radius, z=.515 + ex_radius)
        pdcnd.addSolid(collision_primitive_c2)
        collision_primitive_c3 = mcm.CollisionBox(mcm.Point3(-0.37, -0.87, .533),
                                                  x=.03 + ex_radius, y=.03 + ex_radius, z=.515 + ex_radius)
        pdcnd.addSolid(collision_primitive_c3)
        collision_primitive_r0 = mcm.CollisionBox(mcm.Point3(0.77, 0, 1.018),
                                                  x=.03 + ex_radius, y=.9 + ex_radius, z=.03 + ex_radius)
        pdcnd.addSolid(collision_primitive_r0)
        collision_primitive_r1 = mcm.CollisionBox(mcm.Point3(-0.37, 0, 1.018),
                                                  x=.03 + ex_radius, y=.9 + ex_radius, z=.03 + ex_radius)
        pdcnd.addSolid(collision_primitive_r1)
        collision_primitive_r2 = mcm.CollisionBox(mcm.Point3(0.2, 0.87, 1.018),
                                                  x=.6 + ex_radius, y=.03 + ex_radius, z=.03 + ex_radius)
        pdcnd.addSolid(collision_primitive_r2)
        collision_primitive_r3 = mcm.CollisionBox(mcm.Point3(0.2, -0.87, 1.018),
                                                  x=.6 + ex_radius, y=.03 + ex_radius, z=.03 + ex_radius)
        pdcnd.addSolid(collision_primitive_r3)
        # pdcnd.addSolid(collision_primitive_r1)
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
        from_list = lft_ee_cces + rgt_ee_cces + [lft_ml5, rgt_ml5]
        into_list = [bd, lft_ml0, lft_ml1, lft_ml2, rgt_ml0, rgt_ml1, rgt_ml2]
        self._lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs
        from_list = [lft_ml2, lft_ml3, rgt_ml2, rgt_ml3]
        into_list = [bd]
        self._lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # third pairs
        from_list = [lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces
        into_list = [rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces
        self._lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self._lft_arm.cc.enable_extcd_by_id_list(
            id_list=[lft_ml1, lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces, type="from")
        self._lft_arm.cc.enable_innercd_by_id_list(
            id_list=[bd, lft_ml0, lft_ml1, lft_ml2, lft_ml3, lft_ml4, rgt_ml0, rgt_ml1, rgt_ml2, rgt_ml3,
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
        from_list = lft_ee_cces + rgt_ee_cces + [lft_ml5, rgt_ml5]
        into_list = [bd, lft_ml0, lft_ml1, lft_ml2, rgt_ml0, rgt_ml1, rgt_ml2]
        self._rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs
        from_list = [lft_ml2, lft_ml3, rgt_ml2, rgt_ml3]
        into_list = [bd]
        self._rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # third pairs
        from_list = [lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces
        into_list = [rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces
        self._rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self._rgt_arm.cc.enable_extcd_by_id_list(
            id_list=[rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces, type="from")
        self._rgt_arm.cc.enable_innercd_by_id_list(
            id_list=[bd, rgt_ml0, rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4, lft_ml0, lft_ml1, lft_ml2, lft_ml3, lft_ml4,
                     lft_ml5] + lft_ee_cces, type="into")
        self._rgt_arm.cc.dynamic_into_list = [bd]
        self._rgt_arm.cc.dynamic_ext_list = rgt_ee_cces


if __name__ == '__main__':
    from wrs import wd, mgm

    base = wd.World(cam_pos=[5, 0, 1.2], lookat_pos=[0, 0, .5])
    mcm.mgm.gen_frame().attach_to(base)
    khibt = KHI_DUAL(enable_cc=True)
    # khibt.lft_arm.goto_given_conf(np.radians([90, 90, 90, 90, 90, 90]))
    khibt.gen_meshmodel(toggle_tcp_frame=True, toggle_cdprim=True).attach_to(base)
    base.run()
    tgt_pos = np.array([.4, 0, 1])
    tgt_rotmat = rm.rotmat_from_euler(0, rm.pi * 2 / 3, -rm.pi *3 / 4)
    # tgt_rotmat = rm.rotmat_from_euler(0, rm.pi * 2 / 3, -rm.pi / 4)
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    jnt_values_mult = khibt.lft_arm.ik(tgt_pos, tgt_rotmat, option="multiple")
    if jnt_values_mult is not None:
        print(jnt_values_mult)
        for jnt_values in jnt_values_mult:
            print(jnt_values)
            khibt.goto_given_conf(jnt_values)
            model = khibt.gen_meshmodel()
            model.attach_to(base)
    base.run()
    khibt.lft_arm.manipulator.jnts[0].motion_range = rm.radians(rm.vec(-180, 110))
    khibt.lft_arm.manipulator.jnts[1].motion_range = rm.radians(rm.vec(-40, 90))
    khibt.lft_arm.manipulator.jnts[2].motion_range = rm.radians(rm.vec(-157, -20))
    khibt.lft_arm.manipulator.jnts[3].motion_range = rm.radians(rm.vec(-100, 100))
    khibt.lft_arm.manipulator.jnts[4].motion_range = rm.radians(rm.vec(-125, 0))
    khibt.lft_arm.manipulator.jnts[5].motion_range = rm.radians(rm.vec(-150, 130))
    khibt.lft_arm.manipulator.jlc.finalize(identifier_str=khibt._lft_arm.name + "_dual_lft")
    jnt_values = khibt.lft_arm.ik(tgt_pos, tgt_rotmat)
    print(jnt_values, khibt.jnt_ranges)
    if jnt_values is not None:
        print("with limits ", rm.degrees(jnt_values))
        khibt.goto_given_conf(jnt_values)
        model = khibt.gen_meshmodel(rgb=rm.const.blue)
        model.attach_to(base)

    base.run()