import os
import numpy as np
from panda3d.core import NodePath, CollisionNode, CollisionBox, Point3
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.robots.dual_arm_robot_interface as dari
import wrs.robot_sim.robots.ur3_dual.ur3_rtq85 as u3rtq85
import wrs.robot_sim._kinematics.collision_checker as cc


class UR3Dual(dari.DualArmRobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='ur3_dual', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # the body anchor
        self._body = rkjlc.rkjl.Anchor(name="ur3_dual_base", pos=self.pos, rotmat=self.rotmat, n_flange=2, n_lnk=3)
        self._body.loc_flange_pose_list[0] = [np.array([.0, .258485281374, 1.61051471863]),
                                              rm.rotmat_from_euler(-3.0 * np.pi / 4.0, 0, 0)]
        self._body.loc_flange_pose_list[1] = [np.array([.0, -.258485281374, 1.61051471863]),
                                              rm.rotmat_from_euler(3.0 * np.pi / 4.0, 0, 0) @
                                              rm.rotmat_from_euler(0, 0, np.pi)]
        self._body.lnk_list[0].name = "ur3_dual_base_link"
        self._body.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "ur3_dual_base.stl"),
            name="ur3_dual_base",
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
            userdef_cdprim_fn=self._base_cdprim)
        self._body.lnk_list[0].cmodel.rgba = rm.const.dim_gray
        self._body.lnk_list[1].name = "ur3_dual_frame_link"
        self._body.lnk_list[1].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "ur3_dual_frame.stl"),
            name="ur3_dual_frame",
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
            userdef_cdprim_fn=self._base_cdprim)
        self._body.lnk_list[1].cmodel.rgba = rm.const.tab20_list[14]
        self._body.lnk_list[2].name = "ur3_dual_table_link"
        self._body.lnk_list[2].cmodel = mcm.gen_box(xyz_lengths=np.array([.8, 1.83, .024]), rgb=rm.const.tab20_list[3])
        self._body.lnk_list[2].loc_pos = np.array([.45, .0, 1.082])
        # left arm
        self._lft_arm = u3rtq85.UR3_Rtq85(pos=self._body.gl_flange_pose_list[0][0],
                                          rotmat=self._body.gl_flange_pose_list[0][1],
                                          enable_cc=False)
        self._lft_arm.home_conf = np.array(
            [np.pi / 12.0, -np.pi * 1.0 / 3.0, -np.pi * 2.0 / 3.0, -np.pi, -np.pi * 2.0 / 3.0, 0])
        self._lft_arm.manipulator.jnts[0].motion_range = np.array([-np.pi * 5 / 3, -np.pi / 3])
        self._lft_arm.manipulator.jnts[1].motion_range = np.array([-np.pi, 0])
        self._lft_arm.manipulator.jnts[2].motion_range = np.array([0, np.pi])
        self._lft_arm.manipulator.jnts[3].motion_range = np.array([np.pi / 6, np.pi * 7 / 6])
        self._lft_arm.manipulator.jnts[4].motion_range = np.array([-np.pi, np.pi])
        self._lft_arm.manipulator.jnts[5].motion_range = np.array([-np.pi, np.pi])
        self._lft_arm.manipulator.jlc.finalize(identifier_str=self._lft_arm.name + "_dual_lft")
        # right side
        self._rgt_arm = u3rtq85.UR3_Rtq85(pos=self._body.gl_flange_pose_list[1][0],
                                          rotmat=self._body.gl_flange_pose_list[1][1],
                                          enable_cc=False)
        self._rgt_arm.home_conf = np.array(
            [-np.pi / 12, -np.pi * 2.0 / 3.0, np.pi * 2.0 / 3.0, 0, np.pi * 2.0 / 3.0, np.pi])
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
    def _base_cdprim(name="auto", ex_radius=None):
        pdcnd = CollisionNode(name + "_cnode")
        collision_primitive_c0 = CollisionBox(Point3(0.18, 0.0, 0.105),
                                              x=.61 + ex_radius, y=.41 + ex_radius, z=.105 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.0, 0.0, 0.4445),
                                              x=.321 + ex_radius, y=.321 + ex_radius, z=.2345 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(0.0, 0.0, 0.8895),
                                              x=.05 + ex_radius, y=.05 + ex_radius, z=.6795 + ex_radius)
        pdcnd.addSolid(collision_primitive_c2)
        collision_primitive_c3 = CollisionBox(Point3(0.0, 0.0, 1.619),
                                              x=.1 + ex_radius, y=.275 + ex_radius, z=.05 + ex_radius)
        pdcnd.addSolid(collision_primitive_c3)
        collision_primitive_l0 = CollisionBox(Point3(0.0, 0.300, 1.669),
                                              x=.1 + ex_radius, y=.029 + ex_radius, z=.021 + ex_radius)
        pdcnd.addSolid(collision_primitive_l0)
        collision_primitive_r0 = CollisionBox(Point3(0.0, -0.300, 1.669),
                                              x=.1 + ex_radius, y=.029 + ex_radius, z=.021 + ex_radius)
        pdcnd.addSolid(collision_primitive_r0)
        cdprim = NodePath(name + "_cdprim")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    def _enable_lft_cc(self):
        self._lft_arm.cc = cc.CollisionChecker("lft_arm_collision_checker")
        # body
        bd = self._lft_arm.cc.add_cce(self._body.lnk_list[0])
        tbl = self._lft_arm.cc.add_cce(self._body.lnk_list[2])
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
        into_list = [tbl, bd, lft_ml0, lft_ml1, lft_ml2, rgt_ml0, rgt_ml1, rgt_ml2]
        self._lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs
        from_list = [lft_ml2, rgt_ml2]
        into_list = [tbl, bd, lft_ml0, rgt_ml0]
        self._lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # third pairs
        from_list = [lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces
        into_list = [rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces
        self._lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self._lft_arm.cc.enable_extcd_by_id_list(
            id_list=[lft_ml1, lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces, type="from")
        self._lft_arm.cc.enable_innercd_by_id_list(
            id_list=[tbl, bd, lft_ml0, lft_ml1, lft_ml2, lft_ml3, rgt_ml0, rgt_ml1, rgt_ml2, rgt_ml3,
                     rgt_ml4, rgt_ml5] + rgt_ee_cces, type="into")
        self._lft_arm.cc.dynamic_into_list = [tbl]
        self._lft_arm.cc.dynamic_ext_list = lft_ee_cces[1:]

    def _enable_rgt_cc(self):
        self._rgt_arm.cc = cc.CollisionChecker("rgt_arm_collision_checker")
        # body
        bd = self._rgt_arm.cc.add_cce(self._body.lnk_list[0])
        tbl = self._rgt_arm.cc.add_cce(self._body.lnk_list[2])
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
        into_list = [tbl, bd, lft_ml0, lft_ml1, lft_ml2, rgt_ml0, rgt_ml1, rgt_ml2]
        self._rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs
        from_list = [lft_ml2, rgt_ml2]
        into_list = [tbl, bd, lft_ml0, rgt_ml0]
        self._rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # third pairs
        from_list = [lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces
        into_list = [rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces
        self._rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self._rgt_arm.cc.enable_extcd_by_id_list(
            id_list=[rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces, type="from")
        self._rgt_arm.cc.enable_innercd_by_id_list(
            id_list=[tbl, bd, rgt_ml0, rgt_ml1, rgt_ml2, rgt_ml3, lft_ml0, lft_ml1, lft_ml2, lft_ml3, lft_ml4,
                     lft_ml5] + lft_ee_cces, type="into")
        self._rgt_arm.cc.dynamic_into_list = [tbl]
        self._rgt_arm.cc.dynamic_ext_list = rgt_ee_cces[1:]


if __name__ == '__main__':
    from wrs import wd, mgm

    base = wd.World(cam_pos=[3, 1, 3], lookat_pos=[0, 0, 1])
    mcm.mgm.gen_frame().attach_to(base)
    u3d = UR3Dual()
    u3d.gen_meshmodel(toggle_jnt_frames=True).attach_to(base)
    base.run()
    u3d.use_rgt()
    u3d.change_ee_values(ee_values=u3d.end_effector.jaw_range[1])
    tgt_pos = rm.vec(.4, -.3, 1.15)
    tgt_rotmat =rm.rotmat_from_euler(rm.pi,0,0)
    # mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    jnt_values = u3d.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    if jnt_values is not None:
        u3d.goto_given_conf(jnt_values=jnt_values)
        u3d_meshmodel = u3d.gen_meshmodel()
        u3d_meshmodel.attach_to(base)
    base.run()
