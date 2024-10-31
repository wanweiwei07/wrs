import os
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim._kinematics.collision_checker as cc
import wrs.robot_sim.robots.robot_interface as ri
import wrs.robot_sim.robots.diana7_dual.diana7_rtq85 as d7r85
import wrs.robot_sim.robots.diana7_dual.diana7_sd as d7sd


class Diana7_Dual(ri.RobotInterface):
    """
    author: ziqi.xu, revised by weiwei
    date: 20221118, 20241031
    """

    def __init__(self, pos=rm.np.zeros(3), rotmat=rm.np.eye(3), name='diana7_dual', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        current_file_dir = os.path.dirname(__file__)
        # the body anchor
        self.body = rkjlc.rkjl.Anchor(name="diana7_dual_base", pos=self.pos, rotmat=self.rotmat, n_flange=2, n_lnk=1)
        self.body.loc_flange_pose_list[0] = [rm.np.array([.000559, -.615604, 0]), rm.np.eye(3)]
        self.body.loc_flange_pose_list[1] = [rm.np.array([-0.01165, 0.677502, 0]), rm.np.eye(3)]
        self.body.lnk_list[0].name = "diana7_dual_base"
        self.body.lnk_list[0].loc_pos = rm.np.array([-.453, 0, 0])
        self.body.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "base2.stl"), name="diana7_base2",
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED, userdef_cdprim_fn=self._base_cdprim)
        self.body.lnk_list[0].cmodel.rgba = rm.const.steel_gray
        # left arm
        self.lft_arm = d7sd.Diana7_SD(pos=self.body.gl_flange_pose_list[0][0],
                                      rotmat=self.body.gl_flange_pose_list[0][1], enable_cc=False)
        self.lft_arm.home_conf = rm.np.radians([0.0, -30.0, 0.0, 100.0, 0.0, -110.0, 0.0])
        self.lft_arm.manipulator.jlc.finalize(identifier_str=self.lft_arm.name + "_dual_lft")
        # right side
        self.rgt_arm = d7r85.Diana7_Rtq85(pos=self.body.gl_flange_pose_list[1][0],
                                          rotmat=self.body.gl_flange_pose_list[1][1], enable_cc=False)
        self.rgt_arm.home_conf = rm.np.radians([90.0, -30.0, 0.0, 100.0, 0.0, -110.0, 0.0])
        self.rgt_arm.manipulator.jlc.finalize(identifier_str=self.rgt_arm.name + "_dual_rgt")
        if self.cc is not None:
            self.setup_cc()
        # go home
        self.goto_home_conf()

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

    @property
    def n_dof(self):
        if self.delegator is None:
            return self.lft_arm.n_dof + self.rgt_arm.n_dof
        else:
            return self.delegator.n_dof

    def _enable_lft_cc(self):
        self.lft_arm.cc = cc.CollisionChecker("lft_arm_collision_checker")
        # body
        bd = self.lft_arm.cc.add_cce(self.body.lnk_list[0])
        # left ee
        lft_ee_cces = []
        for id, cdlnk in enumerate(self.lft_arm.end_effector.cdelements):
            lft_ee_cces.append(self.lft_arm.cc.add_cce(cdlnk))
        # left manipulator
        lft_mlb = self.lft_arm.cc.add_cce(self.lft_arm.manipulator.jlc.anchor.lnk_list[0])
        lft_ml0 = self.lft_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[0].lnk)
        lft_ml1 = self.lft_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[1].lnk)
        lft_ml2 = self.lft_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[2].lnk)
        lft_ml3 = self.lft_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[3].lnk)
        lft_ml4 = self.lft_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[4].lnk)
        lft_ml5 = self.lft_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[5].lnk)
        # lft_ml6 = self.lft_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[6].lnk)
        # right ee
        rgt_ee_cces = []
        for id, cdlnk in enumerate(self.rgt_arm.end_effector.cdelements):
            rgt_ee_cces.append(self.lft_arm.cc.add_cce(cdlnk))
        # right manipulator
        rgt_mlb = self.lft_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.anchor.lnk_list[0])
        rgt_ml0 = self.lft_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[0].lnk)
        rgt_ml1 = self.lft_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[1].lnk)
        rgt_ml2 = self.lft_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[2].lnk)
        rgt_ml3 = self.lft_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[3].lnk)
        rgt_ml4 = self.lft_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[4].lnk)
        rgt_ml5 = self.lft_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[5].lnk)
        # rgt_ml6 = self.lft_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[6].lnk)
        # first pairs (hand vs. others)
        from_list = lft_ee_cces + rgt_ee_cces
        into_list = [bd, lft_mlb, lft_ml0, lft_ml1, lft_ml2, lft_ml3, lft_ml4,
                     rgt_mlb, rgt_ml0, rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4]
        self.lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs (forearm vs. others)
        from_list = [lft_ml4, lft_ml5, rgt_ml4, rgt_ml5]
        into_list = [bd, lft_mlb, lft_ml0, lft_ml1, rgt_mlb, rgt_ml0, rgt_ml1]
        self.lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # third pairs (lft vs. rgt)
        from_list = [lft_ml4, lft_ml5] + lft_ee_cces
        into_list = [rgt_ml4, rgt_ml5] + rgt_ee_cces
        self.lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self.lft_arm.cc.enable_extcd_by_id_list(id_list=[lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces, type="from")
        self.lft_arm.cc.enable_innercd_by_id_list(
            id_list=[bd, lft_mlb, lft_ml0, lft_ml1, lft_ml2, lft_ml4, rgt_ml0, rgt_ml1, rgt_ml2, rgt_ml3,
                     rgt_ml4, rgt_ml5] + rgt_ee_cces, type="into")
        self.lft_arm.cc.dynamic_into_list = [bd]
        self.lft_arm.cc.dynamic_ext_list = lft_ee_cces[1:]

    def _enable_rgt_cc(self):
        self.rgt_arm.cc = cc.CollisionChecker("lft_arm_collision_checker")
        # body
        bd = self.rgt_arm.cc.add_cce(self.body.lnk_list[0])
        # left ee
        lft_ee_cces = []
        for id, cdlnk in enumerate(self.lft_arm.end_effector.cdelements):
            lft_ee_cces.append(self.rgt_arm.cc.add_cce(cdlnk))
        # left manipulator
        lft_mlb = self.rgt_arm.cc.add_cce(self.lft_arm.manipulator.jlc.anchor.lnk_list[0])
        lft_ml0 = self.rgt_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[0].lnk)
        lft_ml1 = self.rgt_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[1].lnk)
        lft_ml2 = self.rgt_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[2].lnk)
        lft_ml3 = self.rgt_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[3].lnk)
        lft_ml4 = self.rgt_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[4].lnk)
        lft_ml5 = self.rgt_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[5].lnk)
        # lft_ml6 = self.rgt_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[6].lnk)
        # right ee
        rgt_ee_cces = []
        for id, cdlnk in enumerate(self.rgt_arm.end_effector.cdelements):
            rgt_ee_cces.append(self.rgt_arm.cc.add_cce(cdlnk))
        # right manipulator
        rgt_mlb = self.rgt_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.anchor.lnk_list[0])
        rgt_ml0 = self.rgt_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[0].lnk)
        rgt_ml1 = self.rgt_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[1].lnk)
        rgt_ml2 = self.rgt_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[2].lnk)
        rgt_ml3 = self.rgt_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[3].lnk)
        rgt_ml4 = self.rgt_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[4].lnk)
        rgt_ml5 = self.rgt_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[5].lnk)
        # rgt_ml6 = self.rgt_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[6].lnk)
        # first pairs (hand vs. others)
        from_list = lft_ee_cces + rgt_ee_cces
        into_list = [bd, lft_mlb, lft_ml0, lft_ml1, lft_ml2, lft_ml3, lft_ml4,
                     rgt_mlb, rgt_ml0, rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4]
        self.rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs (forearm vs. others)
        from_list = [lft_ml4, lft_ml5, rgt_ml4, rgt_ml5]
        into_list = [bd, lft_mlb, lft_ml0, lft_ml1, rgt_mlb, rgt_ml0, rgt_ml1]
        self.rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # third pairs (lft vs. rgt)
        from_list = [lft_ml4, lft_ml5] + lft_ee_cces
        into_list = [rgt_ml4, rgt_ml5] + rgt_ee_cces
        self.rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self.rgt_arm.cc.enable_extcd_by_id_list(id_list=[rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces, type="from")
        self.rgt_arm.cc.enable_innercd_by_id_list(
            id_list=[bd, rgt_mlb, rgt_ml0, rgt_ml1, rgt_ml2, rgt_ml4, lft_mlb, lft_ml0, lft_ml1, lft_ml2, lft_ml3,
                     lft_ml4, lft_ml5] + lft_ee_cces, type="into")
        self.rgt_arm.cc.dynamic_into_list = [bd]
        self.rgt_arm.cc.dynamic_ext_list = lft_ee_cces[1:]

    def setup_cc(self):
        """
        author: weiwei
        date: 20240309
        """
        # dual arm
        self._enable_lft_cc()
        self._enable_rgt_cc()
        if self.delegator is not None:
            self.cc = self.delegator.cc
        else:
            self.cc = self.lft_arm.cc  # set left to default

    def use_both(self):
        self.delegator = None
        self.cc = self.lft_arm.cc

    def use_lft(self):
        self.delegator = self.lft_arm
        self.cc = self.delegator.cc

    def use_rgt(self):
        self.delegator = self.rgt_arm
        self.cc = self.delegator.cc

    def backup_state(self):
        if self.delegator is None:
            self.rgt_arm.backup_state()
            self.lft_arm.backup_state()
        else:
            self.delegator.backup_state()

    def restore_state(self):
        if self.delegator is None:
            self.rgt_arm.restore_state()
            self.lft_arm.restore_state()
        else:
            self.delegator.restore_state()

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.body.pos = self.pos
        self.body.rotmat = self.rotmat
        self.lft_arm.fix_to(pos=self.body.gl_flange_pose_list[0][0],
                            rotmat=self.body.gl_flange_pose_list[0][1])
        self.rgt_arm.fix_to(pos=self.body.gl_flange_pose_list[1][0],
                            rotmat=self.body.gl_flange_pose_list[1][1])

    def fk(self, jnt_values, toggle_jacobian=False):
        if self.delegator is None:
            raise AttributeError("FK is not available in multi-arm mode.")
        else:
            return self.delegator.fk(jnt_values=jnt_values, toggle_jacobian=toggle_jacobian)

    def ik(self, tgt_pos, tgt_rotmat, seed_jnt_values=None, obstacle_list=None, toggle_dbg=False):
        if self.delegator is None:
            raise AttributeError("IK is not available in multi-arm mode.")
        else:
            candidates = self.delegator.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=seed_jnt_values,
                                           option="multiple", toggle_dbg=toggle_dbg)
            if candidates is None:
                return None
            result = None
            self.delegator.backup_state()
            for jnt_values in candidates:
                self.delegator.goto_given_conf(jnt_values=jnt_values)
                if self.is_collided(obstacle_list=obstacle_list, toggle_contacts=False):
                    continue
                else:
                    result = jnt_values
                    break
            self.delegator.restore_state()
            return result

    def goto_given_conf(self, jnt_values, ee_values=None):
        """
        :param jnt_values: nparray 1x14, 0:7lft, 7:14rgt
        :return:
        author: weiwei
        date: 20240307
        """
        if self.delegator is None:
            if len(jnt_values) != self.lft_arm.manipulator.n_dof + self.rgt_arm.manipulator.n_dof:
                raise ValueError("The given joint values do not match total n_dof")
            self.lft_arm.goto_given_conf(jnt_values=jnt_values[:self.lft_arm.manipulator.n_dof])
            self.rgt_arm.goto_given_conf(jnt_values=jnt_values[self.rgt_arm.manipulator.n_dof:])  # TODO
        else:
            self.delegator.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)

    def goto_home_conf(self):
        if self.delegator is None:
            self.lft_arm.goto_home_conf()
            self.rgt_arm.goto_home_conf()
        else:
            self.delegator.goto_home_conf()

    def get_jnt_values(self):
        if self.delegator is None:
            return np.concatenate((self.lft_arm.get_jnt_values(), self.rgt_arm.get_jnt_values()))
        else:
            return self.delegator.get_jnt_values()

    def rand_conf(self):
        """
        :return:
        author: weiwei
        date: 20210406
        """
        if self.delegator is None:
            return np.concatenate((self.lft_arm.rand_conf(), self.rgt_arm.rand_conf()))
        else:
            return self.delegator.rand_conf()

    def are_jnts_in_ranges(self, jnt_values):
        if self.delegator is None:
            return self.lft_arm.are_jnts_in_ranges(
                jnt_values=jnt_values[:self.lft_arm.manipulator.n_dof]) and self.rgt_arm.are_jnts_in_ranges(
                jnt_values=jnt_values[self.rgt_arm.manipulator.n_dof:])
        else:
            return self.delegator.are_jnts_in_ranges(jnt_values=jnt_values)

    def get_jaw_width(self):
        return self.get_ee_values()

    def change_jaw_width(self, jaw_width):
        self.change_ee_values(ee_values=jaw_width)

    def is_collided(self, obstacle_list=None, other_robot_list=None, toggle_contacts=False, toggle_dbg=False):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        :param obstacle_list:
        :param other_robot_list:
        :param toggle_contacts: debug
        :param toggle_dbg: debug
        :return: see CollisionChecker is_collided for details
        author: weiwei
        date: 20240307
        """
        collision_info = self.cc.is_collided(obstacle_list=obstacle_list,
                                             other_robot_list=other_robot_list,
                                             toggle_contacts=toggle_contacts,
                                             toggle_dbg=toggle_dbg)
        return collision_info

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False):
        name = self.name + "_stickmodel"
        m_col = mmc.ModelCollection(name=name)
        self.body.gen_stickmodel(toggle_root_frame=toggle_jnt_frames,
                                 toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        self.lft_arm.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                    toggle_jnt_frames=toggle_jnt_frames,
                                    toggle_flange_frame=toggle_flange_frame,
                                    name=name + "_lft_arm").attach_to(m_col)
        self.rgt_arm.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
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
                      toggle_cdmesh=False):
        name = self.name + "_meshmodel"
        m_col = mmc.ModelCollection(name=name)
        self.body.gen_meshmodel(rgb=rgb, alpha=alpha, toggle_flange_frame=toggle_flange_frame,
                                toggle_root_frame=toggle_jnt_frames, toggle_cdprim=toggle_cdprim,
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
    from wrs import wd, mgm

    this_dir, this_filename = os.path.split(__file__)

    base = wd.World(cam_pos=[5, 0, 3], lookat_pos=[0, 0, 1])
    mgm.gen_frame().attach_to(base)
    dia_dual = Diana7_Dual()
    dia_dual.gen_meshmodel(toggle_cdprim=True).attach_to(base)

    base.run()
