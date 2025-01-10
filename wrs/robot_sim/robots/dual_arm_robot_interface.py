import numpy as np
import wrs.modeling.model_collection as mmc
import wrs.robot_sim.robots.robot_interface as ri


class DualArmRobotInterface(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='robot_interface', enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        self._body = None
        self._lft_arm = None
        self._rgt_arm = None

    @property
    def delegator(self):
        return self._delegator

    @property
    def lft_arm(self):
        return self._lft_arm

    @property
    def rgt_arm(self):
        return self._rgt_arm

    @property
    def n_dof(self):
        if self._delegator is None:
            return self._lft_arm.n_dof + self._rgt_arm.n_dof
        else:
            return self._delegator.n_dof

    def _enable_lft_cc(self):
        raise NotImplementedError

    def _enable_rgt_cc(self):
        raise NotImplementedError

    def setup_cc(self):
        """
        author: weiwei
        date: 20240309
        """
        # dual arm
        self._enable_lft_cc()
        self._enable_rgt_cc()
        if self._delegator is not None:
            self.cc = self._delegator.cc
        else:
            self.cc = self._lft_arm.cc  # set left to default

    def use_both(self):
        self._delegator = None
        self.cc = self._lft_arm.cc

    def use_lft(self):
        self._delegator = self._lft_arm
        self.cc = self._delegator.cc

    def use_rgt(self):
        self._delegator = self._rgt_arm
        self.cc = self._delegator.cc

    def backup_state(self):
        if self._delegator is None:
            self._rgt_arm.backup_state()
            self._lft_arm.backup_state()
        else:
            self._delegator.backup_state()

    def restore_state(self):
        if self._delegator is None:
            self._rgt_arm.restore_state()
            self._lft_arm.restore_state()
        else:
            self._delegator.restore_state()

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self._body.pos = self.pos
        self._body.rotmat = self.rotmat
        self._lft_arm.fix_to(pos=self._body.gl_flange_pose_list[0][0],
                            rotmat=self._body.gl_flange_pose_list[0][1])
        self._rgt_arm.fix_to(pos=self._body.gl_flange_pose_list[1][0],
                            rotmat=self._body.gl_flange_pose_list[1][1])

    def fk(self, jnt_values, toggle_jacobian=False):
        if self._delegator is None:
            raise AttributeError("FK is not available in multi-arm mode.")
        else:
            return self._delegator.fk(jnt_values=jnt_values, toggle_jacobian=toggle_jacobian)

    def ik(self, tgt_pos, tgt_rotmat, seed_jnt_values=None, obstacle_list=None, toggle_dbg=False):
        if self._delegator is None:
            raise AttributeError("IK is not available in multi-arm mode.")
        else:
            candidates = self._delegator.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=seed_jnt_values,
                                           option="multiple", toggle_dbg=toggle_dbg)
            if candidates is None:
                return None
            result = None
            self._delegator.backup_state()
            for jnt_values in candidates:
                self._delegator.goto_given_conf(jnt_values=jnt_values)
                if self.is_collided(obstacle_list=obstacle_list, toggle_contacts=False):
                    continue
                else:
                    result = jnt_values
                    break
            self._delegator.restore_state()
            return result

    def goto_given_conf(self, jnt_values, ee_values=None):
        """
        :param jnt_values: nparray 1x14, 0:7lft, 7:14rgt
        :return:
        author: weiwei
        date: 20240307
        """
        if self._delegator is None:
            if len(jnt_values) != self._lft_arm.manipulator.n_dof + self._rgt_arm.manipulator.n_dof:
                raise ValueError("The given joint values do not match total n_dof")
            self._lft_arm.goto_given_conf(jnt_values=jnt_values[:self._lft_arm.manipulator.n_dof])
            self._rgt_arm.goto_given_conf(jnt_values=jnt_values[self._rgt_arm.manipulator.n_dof:])  # TODO
        else:
            self._delegator.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)

    def goto_home_conf(self):
        if self._delegator is None:
            self._lft_arm.goto_home_conf()
            self._rgt_arm.goto_home_conf()
        else:
            self._delegator.goto_home_conf()

    def get_jnt_values(self):
        if self._delegator is None:
            return np.concatenate((self._lft_arm.get_jnt_values(), self._rgt_arm.get_jnt_values()))
        else:
            return self._delegator.get_jnt_values()

    def rand_conf(self):
        """
        :return:
        author: weiwei
        date: 20210406
        """
        if self._delegator is None:
            return np.concatenate((self._lft_arm.rand_conf(), self._rgt_arm.rand_conf()))
        else:
            return self._delegator.rand_conf()

    def are_jnts_in_ranges(self, jnt_values):
        if self._delegator is None:
            return self._lft_arm.are_jnts_in_ranges(
                jnt_values=jnt_values[:self._lft_arm.manipulator.n_dof]) and self._rgt_arm.are_jnts_in_ranges(
                jnt_values=jnt_values[self._rgt_arm.manipulator.n_dof:])
        else:
            return self._delegator.are_jnts_in_ranges(jnt_values=jnt_values)

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

    def toggle_off_eecd(self):
        if self._delegator is None:
            self._lft_arm.toggle_off_eecd()
            self._rgt_arm.toggle_off_eecd()
        else:
            self._delegator.toggle_off_eecd()

    def toggle_on_eecd(self):
        if self._delegator is None:
            self._lft_arm.toggle_on_eecd()
            self._rgt_arm.toggle_on_eecd()
        else:
            self._delegator.toggle_on_eecd()

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False):
        m_col = mmc.ModelCollection(name=self.name + "_stickmodel")
        self._body.gen_stickmodel(name=self.name + "_body_stickmodel",
                                 toggle_root_frame=toggle_jnt_frames,
                                 toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        self._lft_arm.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                    toggle_jnt_frames=toggle_jnt_frames,
                                    toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        self._rgt_arm.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                    toggle_jnt_frames=toggle_jnt_frames,
                                    toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=1,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False):
        m_col = mmc.ModelCollection(name=self.name + "_meshmodel")
        self._body.gen_meshmodel(rgb=rgb, alpha=alpha, toggle_flange_frame=toggle_flange_frame,
                                toggle_root_frame=toggle_jnt_frames, toggle_cdprim=toggle_cdprim,
                                toggle_cdmesh=toggle_cdmesh, name=self.name + "_body_meshmodel").attach_to(m_col)
        self._lft_arm.gen_meshmodel(rgb=rgb,
                                   alpha=alpha,
                                   toggle_tcp_frame=toggle_tcp_frame,
                                   toggle_jnt_frames=toggle_jnt_frames,
                                   toggle_flange_frame=toggle_flange_frame,
                                   toggle_cdprim=toggle_cdprim,
                                   toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        self._rgt_arm.gen_meshmodel(rgb=rgb,
                                   alpha=alpha,
                                   toggle_tcp_frame=toggle_tcp_frame,
                                   toggle_jnt_frames=toggle_jnt_frames,
                                   toggle_flange_frame=toggle_flange_frame,
                                   toggle_cdprim=toggle_cdprim,
                                   toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        return m_col