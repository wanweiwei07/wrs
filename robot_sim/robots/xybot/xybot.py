import math
import numpy as np
import robot_sim._kinematics.jlchain as jl
import robot_sim.robots.robot_interface as ri

class XYBot(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='XYBot'):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        self.jlc = jl.JLChain(homeconf=np.zeros(2), name='XYBot')
        self.jlc.jnts[1]['type'] = 'prismatic'
        self.jlc.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.jlc.jnts[1]['loc_pos'] = np.zeros(3)
        self.jlc.jnts[1]['motion_rng'] = [-2.0, 15.0]
        self.jlc.jnts[2]['type'] = 'prismatic'
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.jnts[2]['loc_pos'] = np.zeros(3)
        self.jlc.jnts[2]['motion_rng'] = [-2.0, 15.0]
        self.jlc.reinitialize()

    def fk(self, component_name='all', jnt_values=np.zeros(2)):
        if component_name != 'all':
            raise ValueError("Only support hnd_name == 'all'!")
        self.jlc.fk(jnt_values)

    def rand_conf(self, component_name='all'):
        if component_name != 'all':
            raise ValueError("Only support hnd_name == 'all'!")
        return self.jlc.rand_conf()

    def get_jntvalues(self, component_name='all'):
        if component_name != 'all':
            raise ValueError("Only support hnd_name == 'all'!")
        return self.jlc.get_jnt_values()

    def is_jnt_values_in_ranges(self, component_name, jnt_values):
        if component_name != 'all':
            raise ValueError("Only support hnd_name == 'all'!")
        return self.jlc.is_jnt_values_in_ranges(jnt_values)

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        for (obpos, size) in obstacle_list:
            dist = np.linalg.norm(np.asarray(obpos) - self.get_jntvalues())
            if dist <= size / 2.0:
                return True  # collision
        return False  # safe


class XYTBot(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='TwoWheelCarBot'):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        self.jlc = jl.JLChain(homeconf=np.zeros(3), name='XYBot')
        self.jlc.jnts[1]['type'] = 'prismatic'
        self.jlc.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.jlc.jnts[1]['loc_pos'] = np.zeros(3)
        self.jlc.jnts[1]['motion_rng'] = [-2.0, 15.0]
        self.jlc.jnts[2]['type'] = 'prismatic'
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.jnts[2]['loc_pos'] = np.zeros(3)
        self.jlc.jnts[2]['motion_rng'] = [-2.0, 15.0]
        self.jlc.jnts[3]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[3]['loc_pos'] = np.zeros(3)
        self.jlc.jnts[3]['motion_rng'] = [-math.pi, math.pi]
        self.jlc.reinitialize()

    def fk(self, component_name='all', jnt_values=np.zeros(3)):
        if component_name != 'all':
            raise ValueError("Only support hnd_name == 'all'!")
        self.jlc.fk(jnt_values)

    def rand_conf(self, component_name='all'):
        if component_name != 'all':
            raise ValueError("Only support hnd_name == 'all'!")
        return self.jlc.rand_conf()

    def get_jntvalues(self, component_name='all'):
        if component_name != 'all':
            raise ValueError("Only support hnd_name == 'all'!")
        return self.jlc.get_jnt_values()

    def is_jnt_values_in_ranges(self, component_name, jnt_values):
        if component_name != 'all':
            raise ValueError("Only support hnd_name == 'all'!")
        return self.jlc.is_jnt_values_in_ranges(jnt_values)

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        for (obpos, size) in obstacle_list:
            dist = np.linalg.norm(np.asarray(obpos) - self.get_jntvalues()[:2])
            if dist <= size / 2.0:
                return True  # collision
        return False  # safe