import numpy as np
import robot_sim._kinematics.jlchain as jl
import robot_sim.robots.robot_interface as ri
import rrt
import rrt_star as rrts


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

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        for (obpos, size) in obstacle_list:
            dist = np.linalg.norm(np.asarray(obpos) - self.get_jntvalues())
            if dist <= size / 2.0:
                return True  # collision
        return False  # safe


# ====Search Path with RRT====
obstacle_list = [
    ((5, 5), 3),
    ((3, 6), 3),
    ((3, 8), 3),
    ((3, 10), 3),
    ((7, 5), 3),
    ((9, 5), 3),
    ((10, 5), 3),
    ((10, 0), 3),
    ((10, -2), 3),
    ((10, -4), 3),
    ((15, 5), 3),
    ((15, 7), 3),
    ((15, 9), 3),
    ((15, 11), 3),
    ((0, 12), 3),
    ((-1, 10), 3),
    ((-2, 8), 3)
]  # [x,y,size]
# Set Initial parameters
robot = XYBot()
rrt_s = rrt.RRT(robot)
rrts_s = rrts.RRT(robot)
import time

start_conf = np.array([15, 0])
goal_conf = np.array([5, -2.5])
start_conf = np.array([0, 0])
goal_conf = np.array([5, 10])

total_t = 0
for i in range(200):
    tic = time.time()
    path = rrt_s.plan(start_conf=start_conf, goal_conf=goal_conf,
                       obstacle_list=obstacle_list,
                       ext_dist=1, rand_rate=70, max_time=1000,
                       max_iter=100000,
                       component_name='all', smoothing_iterations=0,
                       animation=False)
    toc = time.time()
    total_t = total_t + toc - tic
print("rrt costs:", total_t)

total_t = 0
for i in range(200):
    tic = time.time()
    path = rrts_s.plan(start_state=start_conf, goal_conf=goal_conf,
                       obstacle_list=obstacle_list,
                       ext_dist=1, rand_rate=70, max_time=1000,
                       max_iter=100000,
                       component_name='all', smoothing_iterations=0,
                       animation=False)
    toc = time.time()
    total_t = total_t + toc - tic
print("rrts costs:", total_t)
