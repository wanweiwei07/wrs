import math
import numpy as np
import wrs.visualization.panda.world as wd
from wrs import robot_sim as xss, motion as rrtc, modeling as gm, modeling as cm

base = wd.World(cam_pos=[3, 1, 2], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
object = cm.CollisionModel("./objects/bunnysim.stl")
object.set_pos(np.array([.85, 0, .57]))
object.set_rgba([.5,.7,.5,1])
object.attach_to(base)
# robot_s
component_name='arm'
rbt_s = xss.XArmShuidi()
rbt_s.fk(component_name, np.array([0, math.pi * 1 / 2, 0, math.pi*.9, 0, -math.pi / 6, 0]))
#     rbt_s.gen_meshmodel().attach_to(base)
#     rbt_s.show_cdprimit()
#     base.run()
rrtc_planner = rrtc.RRTConnect(rbt_s)
path = rrtc_planner.plan(start_conf=rbt_s.get_jnt_values(),
                         goal_conf=np.array([math.pi/3, math.pi * 1 / 3, 0, math.pi/2, 0, math.pi / 6, 0]),
                         obstacle_list=[object],
                         ext_dist= .1,
                         max_time=300,
                         component_name=component_name)
# print(path)
for pose in path:
    # print(pose)
    rbt_s.fk(component_name, pose)
    robot_meshmodel = rbt_s.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    # robot_meshmodel.show_cdprimit()
    rbt_s.gen_stickmodel().attach_to(base)

base.run()