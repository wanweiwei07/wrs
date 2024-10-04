import math
import numpy as np
import wrs.modeling.model_collection as mmc
import wrs.robot_sim.robots.xarm7_xg_shuidi.xarm7_xg as x7g
from wrs import robot_sim as sd, robot_sim as ri, modeling as gm


class XArm7XGShuidi(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="xarm7_shuidi", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        self.agv = sd.Shuidi(pos=pos, rotmat=rotmat, name=name + "_agv", enable_cc=False)
        self.arm = x7g.XArm7XG(pos=self.agv.gl_flange_pose_list[0][0], rotmat=self.agv.gl_flange_pose_list[0][1],
                               name=name + '_arm', enable_cc=False)
        if self.cc is not None:
            self.setup_cc()

    @property
    def n_dof(self):
        if self.delegator is None:
            return self.agv.n_dof + self.arm.n_dof
        else:
            return self.delegator.n_dof

    def setup_cc(self):
        # agv
        ab = self.cc.add_cce(self.agv.jlc.jnts[2].lnk)
        af = self.cc.add_cce(self.agv.anchor.lnk_list[1])
        # ee
        eb = self.cc.add_cce(self.arm.end_effector.palm.lnk_list[0])
        el0 = self.cc.add_cce(self.arm.end_effector.lft_outer_jlc.jnts[0].lnk)
        el1 = self.cc.add_cce(self.arm.end_effector.lft_outer_jlc.jnts[1].lnk)
        er0 = self.cc.add_cce(self.arm.end_effector.rgt_outer_jlc.jnts[0].lnk)
        er1 = self.cc.add_cce(self.arm.end_effector.rgt_outer_jlc.jnts[1].lnk)
        # manipulator
        mlb = self.cc.add_cce(self.arm.manipulator.jlc.anchor.lnk_list[0])
        ml0 = self.cc.add_cce(self.arm.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.arm.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.arm.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.arm.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.arm.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.arm.manipulator.jlc.jnts[5].lnk)
        ml6 = self.cc.add_cce(self.arm.manipulator.jlc.jnts[6].lnk)
        from_list = [ml4, ml5, ml6, eb, el0, el1, er0, er1]
        into_list = [ab, af, mlb, ml0, ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        # TODO oiee?

    def use_agv(self):
        self.delegator = self.agv

    def use_arm(self):
        self.delegator = self.arm

    def use_all(self):
        self.delegator = None

    def backup_state(self):
        if self.delegator is None:
            self.agv.backup_state()
            self.arm.backup_state()
        else:
            self.delegator.backup_state()

    def restore_state(self):
        if self.delegator is None:
            self.agv.restore_state()
            self.arm.fix_to(pos=self.agv.gl_flange_pose_list[0][0], rotmat=self.agv.gl_flange_pose_list[0][1],
                            jnt_values=self.arm.manipulator.jnt_values_bk.pop())
        else:
            self.delegator.restore_state()

    def fk(self, jnt_values, toggle_jacobian=False):
        if self.delegator is self.arm:
            return self.delegator.fk(jnt_values=jnt_values, toggle_jacobian=toggle_jacobian)
        else:
            raise AttributeError("FK is only available to the manipulator.")

    def goto_given_conf(self, jnt_values, ee_values=None):
        if self.delegator is None:
            if len(jnt_values) != self.agv.n_dof + self.arm.n_dof:
                raise ValueError("The given joint values do not match total n_dof")
            self.agv.goto_given_conf(jnt_values=jnt_values[:self.agv.n_dof])
            self.arm.fix_to(pos=self.agv.gl_flange_pose_list[0][0],
                            rotmat=self.agv.gl_flange_pose_list[0][1],
                            jnt_values=jnt_values[self.agv.n_dof:])  # TODO
        elif self.delegator is self.arm:
            self.delegator.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)
        else:
            self.delegator.goto_given_conf(jnt_values=jnt_values)

    def goto_home_conf(self):
        if self.delegator is None or self.delegator is self.agv:
            self.agv.goto_home_conf()
            self.arm.fix_to(pos=self.agv.gl_flange_pose_list[0][0], rotmat=self.agv.gl_flange_pose_list[0][1])
        else:
            self.delegator.goto_home_conf()

    def get_jnt_values(self):
        if self.delegator is None:
            return np.concatenate([self.agv.get_jnt_values(), self.arm.get_jnt_values()])
        else:
            return self.delegator.get_jnt_values()

    def rand_conf(self):
        if self.delegator is None:
            return np.concatenate((self.agv.rand_conf(), self.arm.rand_conf()))
        else:
            return self.delegator.rand_conf()

    def are_jnts_in_ranges(self, jnt_values):
        if self.delegator is None:
            return (self.agv.are_jnts_in_ranges(jnt_values=jnt_values[:self.agv.n_dof]) and
                    self.arm.are_jnts_in_ranges(jnt_values=jnt_values[self.arm.n_dof:]))
        else:
            return self.delegator.are_jnts_in_ranges(jnt_values=jnt_values)

    def get_ee_values(self):
        return self.arm.get_ee_values()

    def change_ee_values(self, ee_values):
        self.arm.change_ee_values(ee_values=ee_values)

    def get_jaw_width(self):
        return self.get_ee_values()

    def change_jaw_width(self, jaw_width):
        self.change_ee_values(ee_values=jaw_width)

    def is_collided(self, obstacle_list=None, other_robot_list=None, toggle_contacts=False):
        collision_info = self.cc.is_collided(obstacle_list=obstacle_list,
                                             other_robot_list=other_robot_list,
                                             toggle_contacts=toggle_contacts)
        return collision_info

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name='xarm7_xg_shuidi_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.agv.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                toggle_flange_frame=toggle_flange_frame,
                                name=name + "_agv").attach_to(m_col)
        self.arm.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                toggle_jnt_frames=toggle_jnt_frames,
                                toggle_flange_frame=toggle_flange_frame,
                                name=name + "_arm").attach_to(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='xarm7_xg_shuidi_meshmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.agv.gen_meshmodel(rgb=rgb,
                               alpha=alpha,
                               toggle_flange_frame=toggle_flange_frame,
                               toggle_jnt_frames=toggle_jnt_frames,
                               toggle_cdprim=toggle_cdprim,
                               toggle_cdmesh=toggle_cdmesh,
                               name=name + "_agv").attach_to(m_col)
        self.arm.gen_meshmodel(rgb=rgb,
                               alpha=alpha,
                               toggle_tcp_frame=toggle_tcp_frame,
                               toggle_jnt_frames=toggle_jnt_frames,
                               toggle_flange_frame=toggle_flange_frame,
                               toggle_cdprim=toggle_cdprim,
                               toggle_cdmesh=toggle_cdmesh,
                               name=name + "_arm").attach_to(m_col)
        return m_col


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1.5, 0, 3], lookat_pos=[0, 0, .5])

    gm.gen_frame().attach_to(base)
    robot = XArm7XGShuidi(enable_cc=True)
    robot.change_jaw_width(jaw_width=.08)
    robot.gen_meshmodel(toggle_cdprim=False).attach_to(base)
    robot.goto_given_conf(jnt_values=np.array([1, 1, 0, 0, -math.pi/6, 0, math.pi/6, 0, 0, 0]))
    robot.gen_meshmodel(toggle_cdprim=True).attach_to(base)
    base.run()
