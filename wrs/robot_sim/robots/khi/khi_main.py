import os
import math
import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim.robots.robot_interface as ri
import wrs.modeling.model_collection as mmc
import wrs.robot_sim.robots.khi.khi_or2fg7 as kg
import wrs.robot_sim.robots.khi.khi_orsd as ksd
import wrs.robot_sim._kinematics.jlchain as rkjlc


class KHI_DUAL(ri.RobotInterface):
    """
    author: weiwei
    date: 20230805 Toyonaka
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='khi_dual', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        lft_arm_homeconf = np.radians(np.array([0, 0, 0, 0, 0, 0]))
        rgt_arm_homeconf = np.radians(np.array([0, 0, 0, 0, 0, 0]))
        # the body anchor
        self.body = rkjlc.rkjl.Anchor(name="khi_dual_base", pos=self.pos, rotmat=self.rotmat, n_flange=2, n_lnk=1)
        self.body.loc_flange_pose_list[0] = [np.array([0, .4, .726]), rm.rotmat_from_euler(0, 0, -np.pi / 2)]
        self.body.loc_flange_pose_list[1] = [np.array([0, -.4, .726]), rm.rotmat_from_euler(0, 0, -np.pi / 2)]
        self.body.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "base_table.stl"))
        # lft
        self.lft_arm = kg.KHI_OR2FG7(pos=self.body.gl_flange_pose_list[0][0],
                                     rotmat=self.body.gl_flange_pose_list[0][1], enable_cc=False)
        self.lft_arm.fk(jnt_values=lft_arm_homeconf)
        # rgt
        self.rgt_arm = ksd.KHI_ORSD(pos=self.body.gl_flange_pose_list[1][0],
                                    rotmat=self.body.gl_flange_pose_list[1][1], enable_cc=False)
        self.rgt_arm.fk(jnt_values=rgt_arm_homeconf)
        # collision detection
        if self.cc is not None:
            self.setup_cc()
        # go home
        self.goto_home_conf()

    @property
    def n_dof(self):
        if self.delegator is None:
            return self.lft_arm.n_dof + self.rgt_arm.n_dof
        else:
            return self.delegator.n_dof

    def setup_cc(self):
        """
        author: weiwei
        date: 20240524
        """
        # dual arm
        # base
        bd = self.cc.add_cce(self.body.lnk_list[0], toggle_extcd=False)
        # left ee
        lft_elb = self.cc.add_cce(self.lft_arm.end_effector.jlc.anchor.lnk_list[0])
        lft_el0 = self.cc.add_cce(self.lft_arm.end_effector.jlc.jnts[0].lnk)
        lft_el1 = self.cc.add_cce(self.lft_arm.end_effector.jlc.jnts[1].lnk)
        # left manipulator
        lft_ml0 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[0].lnk, toggle_extcd=False)
        lft_ml1 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[1].lnk)
        lft_ml2 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[2].lnk)
        lft_ml3 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[3].lnk)
        lft_ml4 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[4].lnk)
        lft_ml5 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[5].lnk)
        # right ee
        rgt_elb = self.cc.add_cce(self.rgt_arm.end_effector.jlc.anchor.lnk_list[0])
        # right manipulator
        rgt_ml0 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[0].lnk, toggle_extcd=False)
        rgt_ml1 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[1].lnk)
        rgt_ml2 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[2].lnk)
        rgt_ml3 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[3].lnk)
        rgt_ml4 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[4].lnk)
        rgt_ml5 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[5].lnk)
        # first pairs
        from_list = [lft_ml4, lft_ml5, lft_elb, lft_el0, lft_el1, rgt_ml4, rgt_ml5, rgt_elb]
        into_list = [bd, lft_ml0, rgt_ml0]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs
        from_list = [lft_ml0, lft_ml1, rgt_ml0, rgt_ml1]
        into_list = [lft_elb, lft_el0, lft_el1, rgt_elb]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        # third pairs
        from_list = [lft_ml1, lft_ml2, lft_ml3, lft_ml4, lft_ml5, lft_elb, lft_el0, lft_el1]
        into_list = [rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5, rgt_elb]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        # point low-level cc to the high-level one
        self.lft_arm.cc = self.cc
        self.rgt_arm.cc = self.cc

    def use_both(self):
        self.delegator = None

    def use_lft(self):
        self.delegator = self.lft_arm

    def use_rgt(self):
        self.delegator = self.rgt_arm

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
            self.rgt_arm.goto_given_conf(jnt_values=jnt_values[self.rgt_arm.manipulator.n_dof:]) # TODO
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

    def is_collided(self, obstacle_list=None, other_robot_list=None, toggle_contacts=False):
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
                      toggle_cdmesh=False,
                      name='yumi_meshmodel'):
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
    import time
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[5, 3, 4], lookat_pos=[0, 0, 1])
    mcm.mgm.gen_frame().attach_to(base)
    khibt = KHI_DUAL(enable_cc=True)
    khibt = khibt.gen_meshmodel()
    khibt.attach_to(base)
    base.run()

    # tgt_pos = np.array([.4, 0, .2])
    # tgt_rotmat = rm.rotmat_from_euler(0, math.pi * 2 / 3, -math.pi / 4)
    # ik test
    component_name = 'lft_arm_waist'
    tgt_pos = np.array([-.3, .45, .55])
    tgt_rotmat = rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
    # tgt_rotmat = np.eye(3)
    mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    tic = time.time()
    jnt_values = nxt_instance.ik(component_name, tgt_pos, tgt_rotmat, toggle_dbg=True)
    toc = time.time()
    print(toc - tic)
    nxt_instance.fk(component_name, jnt_values)
    nxt_meshmodel = nxt_instance.gen_mesh_model()
    nxt_meshmodel.attach_to(base)
    nxt_instance.gen_stickmodel().attach_to(base)
    # tic = time.time()
    # result = nxt_instance.is_collided()
    # toc = time.time()
    # print(result, toc - tic)
    base.run()

    # hold test
    component_name = 'lft_arm'
    obj_pos = np.array([-.1, .3, .3])
    obj_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    objfile = os.path.join(basis.__path__[0], 'objects', 'tubebig.stl')
    objcm = mcm.CollisionModel(objfile, cdprim_type='cylinder')
    objcm.set_pos(obj_pos)
    objcm.set_rotmat(obj_rotmat)
    objcm.attach_to(base)
    objcm_copy = objcm.copy()
    nxt_instance.hold(objcm=objcm_copy, jaw_width=0.03, hnd_name='lft_hnd')
    tgt_pos = np.array([.4, .5, .4])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 3)
    jnt_values = nxt_instance.ik(component_name, tgt_pos, tgt_rotmat)
    nxt_instance.fk(component_name, jnt_values)
    # nxt_instance.show_cdprimit()
    nxt_meshmodel = nxt_instance.gen_mesh_model()
    nxt_meshmodel.attach_to(base)

    base.run()
