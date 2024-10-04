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
component_name = 'lft_arm_waist'
robot_s = nxt.Nextage()

start_pos = np.array([.4, .1, .1])
start_rotmat = rm.rotmat_from_axangle([0,1,0], -math.pi/2)
gm.gen_frame(pos=start_pos,rotmat=start_rotmat).attach_to(base)
robot_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
# base.run()
start_conf = robot_s.ik(component_name, start_pos, start_rotmat)
goal_pos = np.array([.3, .5, .6])
goal_rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi/2).dot(rm.rotmat_from_axangle([0, 1, 0], math.pi))
gm.gen_frame(pos=goal_pos,rotmat=goal_rotmat).attach_to(base)
# base.run()
goal_conf = robot_s.ik(component_name, goal_pos, goal_rotmat)
rrtc_planner = rrtc.RRTConnect(robot_s)
path = rrtc_planner.plan(component_name=component_name,
                         start_conf=start_conf,
                         goal_conf=goal_conf,
                         obstacle_list=[object_box],
                         ext_dist=.05,
                         smoothing_iterations=150,
                         max_time=300)
print(path)
for pose in path[1:-2]:
    print(pose)
    robot_s.fk(component_name, pose)
    robot_s.gen_stickmodel().attach_to(base)
for pose in [path[0]]:
    print(pose)
    robot_s.fk(component_name, pose)
    robot_meshmodel = robot_s.gen_meshmodel(rgba=[1,0,0,.3])
    robot_meshmodel.attach_to(base)
for pose in [path[-1]]:
    print(pose)
    robot_s.fk(component_name, pose)
    robot_meshmodel = robot_s.gen_meshmodel(rgba=[0,1,0,.3])
    robot_meshmodel.attach_to(base)
    # robot_s.gen_stickmodel().attach_to(base)

# robot_attached_list = []
# counter = [0]
# def update(robot_s, path, robot_attached_list, counter, task):
#     if counter[0] >= len(path):
#         counter[0] = 0
#     if len(robot_attached_list) != 0:
#         for robot_attached in robot_attached_list:
#             robot_attached.detach()
#         robot_attached_list.clear()
#     pose = path[counter[0]]
#     robot_s.fk(hnd_name, pose)
#     robot_meshmodel = robot_s.gen_meshmodel()
#     robot_meshmodel.attach_to(base)
#     robot_attached_list.append(robot_meshmodel)
#     counter[0]+=1
#     return task.again
#
# taskMgr.doMethodLater(0.01, update, "update",
#                       extraArgs=[robot_s, path, robot_attached_list, counter],
#                       appendTask=True)

base.run()
