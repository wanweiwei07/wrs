import math
import numpy as np
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.robots.robot_interface as ri
import wrs.robot_sim._kinematics.model_generator as rkmg


class XYBot(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='XYBot', enable_cc=False):
        """
        2D collision detection when enable_cc is False
        :param pos:
        :param rotmat:
        :param name:
        :param enable_cc:
        """
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        self.jlc = rkjlc.JLChain(n_dof=2, name='XYBot')
        self.jlc.home = np.zeros(2)
        self.jlc.jnts[0].change_type(type=rkjlc.const.JntType.PRISMATIC)
        self.jlc.jnts[0].loc_motion_ax = np.array([1, 0, 0])
        self.jlc.jnts[0].loc_pos = np.zeros(3)
        self.jlc.jnts[0].motion_range = np.array([-.2, 1.5])
        self.jlc.jnts[1].change_type(type=rkjlc.const.JntType.PRISMATIC)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[1].loc_pos = np.zeros(3)
        self.jlc.jnts[1].motion_range = np.array([-.2, 1.5])
        self.jlc.jnts[1].lnk.cmodel = mcm.gen_box(xyz_lengths=np.array([.2, .2, .2]), rgb=np.array([0, .5, .7]), alpha=1)
        self.jlc.finalize()
        if self.cc is not None:
            self.setup_cc()
        self.jnt_values_bk = []

    def setup_cc(self):
        body = self.cc.add_cce(self.jlc.jnts[1].lnk)

    def backup_state(self):
        self.jnt_values_bk.append(self.jlc.get_jnt_values())

    def restore_state(self):
        self.jlc.goto_given_conf(jnt_values=self.jnt_values_bk.pop())

    def goto_given_conf(self, jnt_values=np.zeros(2)):
        self.jlc.goto_given_conf(jnt_values=jnt_values)

    def rand_conf(self):
        return self.jlc.rand_conf()

    def get_jnt_values(self):
        return self.jlc.get_jnt_values()

    def are_jnts_in_ranges(self, jnt_values):
        return self.jlc.are_jnts_in_ranges(jnt_values)

    def is_collided(self, obstacle_list=[], other_robot_list=[], toggle_contacts=False):
        if self.cc is None:
            for (pos, diameter) in obstacle_list:
                dist = np.linalg.norm(np.asarray(pos) - self.get_jnt_values())
                if dist <= diameter / 2.0:
                    return (True, []) if toggle_contacts else True  # collision
            return (False, []) if toggle_contacts else False  # safe
        else:
            return super().is_collided(obstacle_list=obstacle_list, other_robot_list=other_robot_list,
                                       toggle_contacts=toggle_contacts)

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='single_arm_robot_interface_meshmodel'):
        return rkmg.gen_jlc_mesh(jlc=self.jlc,
                                 rgb=rgb,
                                 alpha=alpha)


class XYWBot(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='TwoWheelCarBot', enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        self.jlc = jl.JLChain(n_dof=3, name='XYWBot')
        self.jlc.home = np.zeros(3)
        self.jlc.jnts[0].change_type(type=rkc.JntType.PRISMATIC)
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 0])
        self.jlc.jnts[0].loc_pos = np.zeros(3)
        self.jlc.jnts[0].motion_range = np.array([-2.0, 15.0])
        self.jlc.jnts[1].change_type(type=rkc.JntType.PRISMATIC)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[1].loc_pos = np.zeros(3)
        self.jlc.jnts[1].motion_range = np.array([-2.0, 15.0])
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[2].loc_pos = np.zeros(3)
        self.jlc.jnts[2].motion_range = [-math.pi, math.pi]
        self.jlc.finalize()
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        self.cc.add_cce(self.jlc.jnts[1].lnk)

    def goto_given_conf(self, jnt_values=np.zeros(3)):
        self.jlc.goto_given_conf(jnt_values=jnt_values)

    def rand_conf(self):
        return self.jlc.rand_conf()

    def get_jnt_values(self):
        return self.jlc.get_jnt_values()

    def are_jnts_in_ranges(self, jnt_values):
        return self.jlc.are_jnts_in_ranges(jnt_values)

    def is_collided(self, obstacle_list=[], other_robot_list=[], toggle_contacts=False):
        if self.cc is None:
            for (pos, diameter) in obstacle_list:
                dist = np.linalg.norm(np.asarray(pos) - self.get_jnt_values())
                if dist <= diameter / 2.0:
                    return True  # collision
            return False  # safe
        else:
            return super().is_collided(obstacle_list=obstacle_list, other_robot_list=other_robot_list,
                                       toggle_contacts=toggle_contacts)

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='single_arm_robot_interface_meshmodel'):
        return rkmg.gen_jlc_mesh(jlc=self.jlc,
                                 rgb=rgb,
                                 alpha=alpha)


if __name__ == "__main__":
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mcm.mgm.gen_frame().attach_to(base)
    robot = XYBot(enable_cc=True)
    robot.gen_meshmodel().attach_to(base)
    base.run()
