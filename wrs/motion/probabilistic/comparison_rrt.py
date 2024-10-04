import numpy as np
import rrt
import rrt_star as rrts
from wrs import robot_sim as xyb

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
robot = xyb.XYBot()
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
                       max_n_iter=100000,
                       component_name='all', smoothing_n_iter=0,
                       animation=False)
    toc = time.time()
    total_t = total_t + toc - tic
print("rrts costs:", total_t)
