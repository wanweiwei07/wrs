import math
import numpy as np
import wrs.visualization.panda.world as wd
from wrs import basis as rm, robot_sim as ur3d, motion as rrtc, modeling as gm, modeling as cm

base = wd.World(cam_pos=[2, 1, 3], lookat_pos=[0, 0, 1.1])
gm.gen_frame().attach_to(base)
obstacle = cm.CollisionModel("./objects/Part_Studio_1.stl")
obstacle.set_pos(np.array([.4, -.5, 1.2]))
obstacle.set_rgba([.5, .7, .3, 1])
obstacle.attach_to(base)

obstacle2 = cm.CollisionModel("./objects/Part_Studio_2.stl")
obstacle2.set_pos(np.array([.4, -.5, 1.3]))
obstacle2.set_rgba([.5, .7, .3, 1])
obstacle2.attach_to(base)

robot_s = ur3d.UR3Dual()
rrtc_planner = rrtc.RRTConnect(robot_s)
component_name = 'rgt_arm'
#スタート位置
start_pos = np.array([.2, -.55, 1.2])
start_rotmat = rm.rotmat_from_euler(ai=math.pi / 2, aj=math.pi, ak=0, axes="szxz")
start_angle = robot_s.ik(component_name=component_name, tgt_pos=start_pos, tgt_rotmat=start_rotmat)
#start_angle = np.array([0, -math.pi / 2, math.pi / 6, 0, math.pi / 2, -math.pi])

print(start_angle)

#終端
end_pos = np.array([.4, -.5, 1.2])
end_rotmat = rm.rotmat_from_euler(ai=math.pi / 2, aj=math.pi / 2, ak=0, axes="szxz")
end_angle = robot_s.ik(component_name=component_name, tgt_pos=end_pos, tgt_rotmat=end_rotmat)
#end_angle = np.array([0, math.pi / 6, math.pi / 3, -math.pi / 8, -math.pi / 5, math.pi / 10])

print(end_angle)
path = rrtc_planner.plan(component_name=component_name,
                         start_conf=start_angle,
                         goal_conf=end_angle,
                         obstacle_list=[obstacle, obstacle2],
                         ext_dist=.2,
                         max_time=300)
for pose in path:
    robot_s.fk(component_name, pose)
    robot_meshmodel = robot_s.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    robot_s.gen_stickmodel().attach_to(base)

base.run()