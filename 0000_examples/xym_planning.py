import time
import math
import numpy as np
from basis import robot_math as rm
import visualization.panda.world as wd
import modeling.geometricmodel as gm
import modeling.collisionmodel as cm
import robotsim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xav
import motion.probabilistic.rrt_connect as rrtc

base = wd.World(campos=[3, 1, 2], lookatpos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
object = cm.CollisionModel("./objects/bunnysim.stl")
object.set_pos(np.array([.85, 0, .37]))
object.set_rgba([.5,.7,.3,1])
object.attach_to(base)
# robot
component_name='arm'
robot_instance = xav.XArm7YunjiMobile()
robot_instance.fk(component_name, np.array([0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0]))
rrtc_planner = rrtc.RRTConnect(robot_instance)
path = rrtc_planner.plan(start_conf=np.array([0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0]),
                         goal_conf=np.array([math.pi/3, math.pi * 1 / 3, 0, math.pi/2, 0, math.pi / 6, 0]),
                         obstacle_list=[object],
                         ext_dist=.1,
                         rand_rate=70,
                         maxtime=300,
                         component_name=component_name)
print(path)
for pose in path:
    print(pose)
    robot_instance.fk(component_name, pose)
    robot_meshmodel = robot_instance.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    # robot_meshmodel.show_cdprimit()
    robot_instance.gen_stickmodel().attach_to(base)
# hold
# robot_instance.hold(object, jawwidth=.05)
# robot_instance.fk(np.array([0, 0, 0, math.pi/6, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, math.pi/6]))
# robot_meshmodel = robot_instance.gen_meshmodel()
# robot_meshmodel.attach_to(base)
# robot_instance.show_cdprimit()
# tic = time.time()
# result = robot_instance.is_collided() # problematic
# toc = time.time()
# print(result, toc - tic)
# base.run()
# release
# robot_instance.release(object, jawwidth=.082)
# robot_instance.fk(np.array([0, 0, 0, math.pi/3, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, math.pi/6]))
# robot_meshmodel = robot_instance.gen_meshmodel()
# robot_meshmodel.attach_to(base)
# robot_meshmodel.show_cdprimit()
# tic = time.time()
# result = robot_instance.is_collided()
# toc = time.time()
# print(result, toc - tic)

#copy
# robot_instance2 = robot_instance.copy()
# robot_instance2.move_to(pos=np.array([.5,0,0]), rotmat=rm.rotmat_from_axangle([0,0,1], math.pi/6))
# objcm_list = robot_instance2.get_hold_objlist()
# robot_instance2.release(objcm_list[-1], jawwidth=.082)
# robot_meshmodel = robot_instance2.gen_meshmodel()
# robot_meshmodel.attach_to(base)
# robot_instance2.show_cdprimit()

base.run()
