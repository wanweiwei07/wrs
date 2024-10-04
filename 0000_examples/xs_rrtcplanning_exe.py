import math
import numpy as np
import wrs.visualization.panda.world as wd
from wrs import robot_sim as xss, motion as rrtc, modeling as gm, modeling as cm
import wrs.robot_con.xarm_shuidi.xarm_shuidi_x as xsx

base = wd.World(cam_pos=[3, 1, 2], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
object = cm.CollisionModel("./objects/bunnysim.stl")
object.set_pos(np.array([.85, 0, .37]))
object.set_rgba([.5,.7,.3,1])
object.attach_to(base)
# robot_s
component_name='arm'
robot_s = xss.XArmShuidi()
robot_s.fk(component_name, np.array([0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0]))
# robot_x
robot_x = xsx.XArmShuidiX(ip="10.2.0.203")
init_jnt_angles = robot_x.arm_get_jnt_values()
print(init_jnt_angles)
rrtc_planner = rrtc.RRTConnect(robot_s)
path = rrtc_planner.plan(start_conf=init_jnt_angles,
                         # end_conf=np.array([math.pi/3, math.pi * 1 / 3, 0, math.pi/2, 0, math.pi / 6, 0]),
                         goal_conf = robot_s.manipulator_dict['arm'].home_conf,
                         obstacle_list=[object],
                         ext_dist= .1,
                         max_time=300,
                         component_name=component_name)
robot_x.arm_move_jspace_path(path)

# print(path)
for pose in path:
    # print(pose)
    robot_s.fk(component_name, pose)
    robot_meshmodel = robot_s.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    # robot_meshmodel.show_cdprimit()
    robot_s.gen_stickmodel().attach_to(base)

base.run()
