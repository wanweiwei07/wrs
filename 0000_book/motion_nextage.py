import math
import numpy as np
from wrs import basis as rm, robot_sim as nxt, motion as rrtc, modeling as gm, modeling as cm
import wrs.visualization.panda.world as wd

base = wd.World(cam_pos=[4, -1, 2], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
object_box = cm.gen_box(xyz_lengths=[.15, .15, .15])
object_box.set_pos(np.array([.4, .3, .4]))
object_box.set_rgba([.5, .7, .3, 1])
object_box.attach_to(base)
# robot_s
component_name = 'lft_arm'
robot_s = nxt.Nextage()

# start_pos = np.array([.4, 0, .2])
# start_rotmat = rm.rotmat_from_euler(0, math.pi * 2 / 3, -math.pi / 4)
start_pos = np.array([.4, .1, .1])
start_rotmat = rm.rotmat_from_axangle([0,1,0], -math.pi/2)
start_conf = robot_s.ik(component_name, start_pos, start_rotmat)
# goal_pos = np.array([.3, .5, .7])
# goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi)
goal_pos = np.array([.3, .5, .6])
goal_rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi/2).dot(rm.rotmat_from_axangle([0, 1, 0], math.pi))
goal_conf = robot_s.ik(component_name, goal_pos, goal_rotmat)

print(start_conf, goal_conf)

rrtc_planner = rrtc.RRTConnect(robot_s)
path = rrtc_planner.plan(component_name=component_name,
                         start_conf=start_conf,
                         goal_conf=goal_conf,
                         obstacle_list=[object_box],
                         ext_dist=.1,
                         smoothing_iterations=150,
                         max_time=300)
# print(path)
# import matplotlib.pyplot as plt
# import motion.trajectory.polynomial_wrsold as trajp
# control_frequency = .005
# interval_time = 1
# traj_gen = trajp.TrajPoly(method="quintic")
# interpolated_confs, interpolated_spds, interpolated_accs = \
#     traj_gen.piecewise_interpolation(path, control_frequency=control_frequency, time_interval=interval_time)
#
# fig, axs = plt.subplots(3, figsize=(21,7))
# fig.tight_layout(pad=.7)
# x = np.linspace(0, interval_time*(len(path) - 1), (len(path) - 1) * math.floor(interval_time / control_frequency))
# axs[0].plot(x, interpolated_confs)
# axs[0].plot(range(0, interval_time * (len(path)), interval_time), path, '--o')
# axs[1].plot(x, interpolated_spds)
# axs[2].plot(x, interpolated_accs)
# plt.show()

for pose in path[1:-2]:
    print(pose)
    robot_s.fk(component_name, pose)
    robot_s.gen_stickmodel().attach_to(base)
for pose in [path[0], path[-1]]:
    print(pose)
    robot_s.fk(component_name, pose)
    robot_meshmodel = robot_s.gen_meshmodel()
    robot_meshmodel.attach_to(base)
base.run()
