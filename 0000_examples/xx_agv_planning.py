import math
import numpy as np
import wrs.visualization.panda.world as wd
from wrs import robot_sim as xss, modeling as gm, modeling as cm
import wrs.motion.probabilistic.rrt_differential_wheel_connect as rrtdwc

base = wd.World(cam_pos=[10, 1, 5], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
box_homo = np.eye(4)
box_homo[:3,3]=np.array([0,0,.5])
object_box = cm.gen_box(np.array([.3,.3,1]), homomat=box_homo, rgba=[.57, .57, .5, 1])
object_box.set_pos(np.array([1.9,-1,0]))
object_box.attach_to(base)
# object2
object_box2 = object_box.copy()
object_box2.set_pos(np.array([1.9,-.5,0]))
object_box2.attach_to(base)
# object3
object_box3 = object_box.copy()
object_box3.set_pos(np.array([.9,-.5,0]))
object_box3.attach_to(base)
# object4
object_box4 = object_box.copy()
object_box4.set_pos(np.array([.9,-1,0]))
object_box4.attach_to(base)
# robot_s
component_name='agv'
robot_instance = xss.XArmShuidi()
rrtc_planner = rrtdwc.RRTDWConnect(robot_instance)
path = rrtc_planner.plan(start_conf=np.array([0,0,0]),
                         goal_conf=np.array([2,-2,math.radians(190)]),
                         obstacle_list=[object_box, object_box2, object_box3, object_box4],
                         ext_dist= .1,
                         max_time=300,
                         component_name=component_name)
# print(path)
for pose in path:
    # print(pose)
    robot_instance.fk(component_name, pose)
    robot_meshmodel = robot_instance.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    # robot_meshmodel.show_cdprimit()
    robot_instance.gen_stickmodel().attach_to(base)

base.run()
