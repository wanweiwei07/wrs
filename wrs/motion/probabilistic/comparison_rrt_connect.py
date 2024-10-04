import numpy as np
from wrs import robot_sim as xyb
import _rrt_connect_wrsnew as rrtc_wrn
import rrt_connect as rrtc
import _rrt_connect_intuitive as rrtc_iv
import _rrt_connect_wrsold as rrtc_wro

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
rrtc_s = rrtc.RRTConnect(robot)
rrtc_iv_s = rrtc_iv.RRTConnect(robot)
rrtc_wro_s = rrtc_wro.RRTConnect(robot)
rrtc_wrn_s = rrtc_wrn.RRTConnect(robot)
import time

start_conf = np.array([15, 0])
goal_conf = np.array([5, -2.5])
start_conf = np.array([0, 0])
goal_conf = np.array([5, 10])

total_t = 0
for i in range(500):
    tic = time.time()
    path = rrtc_s.plan(start_conf=start_conf, goal_conf=goal_conf,
                       obstacle_list=obstacle_list,
                       ext_dist=1, rand_rate=70, max_time=300,
                       component_name='all', smoothing_iterations=0,
                       animation=False)
    toc = time.time()
    total_t = total_t + toc - tic
print("rrtc costs:", total_t)

total_t = 0
for i in range(500):
    tic = time.time()
    path = rrtc_wrn_s.plan(start_conf=start_conf, goal_conf=goal_conf,
                           obstacle_list=obstacle_list,
                           ext_dist=1, rand_rate=70, max_time=300,
                           component_name='all', smoothing_iterations=0,
                           animation=False)
    toc = time.time()
    total_t = total_t + toc - tic
print("rrtc wrsnew costs:", total_t)

total_t = 0
for i in range(500):
    tic = time.time()
    path = rrtc_iv_s.plan(start_conf=start_conf, goal_conf=goal_conf,
                          obstacle_list=obstacle_list,
                          ext_dist=1, rand_rate=70, max_time=300,
                          component_name='all',
                          animation=False)
    toc = time.time()
    total_t = total_t + toc - tic
print("rrtc intuitive costs:", total_t)

total_t = 0
for i in range(500):
    tic = time.time()
    path = rrtc_wro_s.plan(start_conf=start_conf, goal_conf=goal_conf,
                          obstacle_list=obstacle_list,
                          ext_dist=1, rand_rate=70, max_time=300,
                          component_name='all',
                          animation=False)
    toc = time.time()
    total_t = total_t + toc - tic
print("rrtc wrsold costs:", total_t)
