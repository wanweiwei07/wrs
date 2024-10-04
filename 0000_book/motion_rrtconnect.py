import numpy as np
import matplotlib.pyplot as plt
from wrs import robot_sim as jl, robot_sim as ri, motion as rrtc


class XYBot(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='XYBot'):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        self.jlc = jl.JLChain(home_conf=np.zeros(2), name='XYBot')
        self.jlc.jnts[1]['end_type'] = 'prismatic'
        self.jlc.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.jlc.jnts[1]['loc_pos'] = np.zeros(3)
        self.jlc.jnts[1]['motion_range'] = [-2.0, 15.0]
        self.jlc.jnts[2]['end_type'] = 'prismatic'
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.jnts[2]['loc_pos'] = np.zeros(3)
        self.jlc.jnts[2]['motion_range'] = [-2.0, 15.0]
        self.jlc.finalize()

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

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        for (obpos, size) in obstacle_list:
            dist = np.linalg.norm(np.asarray(obpos) - self.get_jntvalues())
            if dist <= size / 2.0:
                return True  # collision
        return False  # safe


if __name__ == '__main__':
    # ====Search Path with RRT====
    obstacle_list = [
        ((5, 5), 3), ((3, 6), 3), ((3, 8), 3), ((3, 10), 3),
        ((7, 5), 3), ((9, 5), 3), ((10, 5), 3), ((10, 0), 3),
        ((10, -2), 3), ((10, -4), 3), ((0, 12), 3), ((-1, 10), 3), ((-2, 8), 3)
    ]  # [x,y,size]
    # Set Initial parameters
    robot = XYBot()
    rrtc_instance = rrtc.RRTConnect(robot)
    path = rrtc_instance.plan(component_name='all',
                              start_conf=np.array([0, 0]),
                              goal_conf=np.array([5, 10]),
                              obstacle_list=obstacle_list,
                              ext_dist=1,
                              max_time=300,
                              animation=True)
    plt.plot([conf[0] for conf in path], [conf[1] for conf in path], '-k')
    plt.show()
