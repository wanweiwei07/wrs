import math
import numpy as np
import robot_sim._kinematics.jlchain as jl
import robot_sim.robots.robot_interface as ri
import robot_sim._kinematics.constant as rkc


class XYBot(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='XYBot'):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=False)
        self.jlc = jl.JLChain(n_dof=2, name='XYBot')
        self.jlc.home = np.zeros(2)
        self.jlc.jnts[0].change_type(type=rkc.JntType.PRISMATIC)
        self.jlc.jnts[0].loc_motion_ax = np.array([1, 0, 0])
        self.jlc.jnts[0].loc_pos = np.zeros(3)
        self.jlc.jnts[0].motion_range = np.array([-2.0, 15.0])
        self.jlc.jnts[1].change_type(type=rkc.JntType.PRISMATIC)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[1].loc_pos = np.zeros(3)
        self.jlc.jnts[1].motion_range = np.array([-2.0, 15.0])
        self.jlc.finalize()

    def goto_given_conf(self, jnt_values=np.zeros(2)):
        self.jlc.go_given_conf(jnt_values=jnt_values)

    def rand_conf(self):
        return self.jlc.rand_conf()

    def get_jnt_values(self):
        return self.jlc.get_jnt_values()

    def are_jnts_in_ranges(self, jnt_values):
        return self.jlc.are_jnts_in_ranges(jnt_values)

    def is_collided(self, obstacle_list=[]):
        for (pos, diameter) in obstacle_list:
            dist = np.linalg.norm(np.asarray(pos) - self.get_jnt_values())
            if dist <= diameter / 2.0:
                return True  # collision
        return False  # safe


class XYWBot(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='TwoWheelCarBot'):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
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

    def goto_given_conf(self, jnt_values=np.zeros(3)):
        self.jlc.go_given_conf(jnt_values=jnt_values)

    def rand_conf(self):
        return self.jlc.rand_conf()

    def get_jnt_values(self):
        return self.jlc.get_jnt_values()

    def are_jnts_in_ranges(self, jnt_values):
        return self.jlc.are_jnts_in_ranges(jnt_values)

    def is_collided(self, obstacle_list=[]):
        for (pos, diameter) in obstacle_list:
            dist = np.linalg.norm(np.asarray(pos) - self.get_jnt_values()[:2])
            if dist <= diameter / 2.0:
                return True  # collision
        return False  # safe
