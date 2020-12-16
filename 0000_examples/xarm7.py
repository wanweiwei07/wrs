import time
import math
import numpy as np
import visualization.panda.world as wd
import modeling.geometricmodel as gm
import modeling.collisionmodel as cm
import robotsim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xav

base = wd.World(campos=[3, 1, 2], lookatpos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
object = cm.CollisionModel("./objects/bunnysim.stl")
object.set_pos(np.array([.85, 0, .37]))
object.set_color([.5,.7,.3,1])
object.attach_to(base)
# robot
robot_instance = xav.XArm7YunjiMobile()
robot_instance.fk(np.array([0, 0, 0, 0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0, 0.082]))
robot_meshmodel = robot_instance.gen_meshmodel()
robot_meshmodel.attach_to(base)
robot_meshmodel.show_cdprimit()
robot_instance.gen_stickmodel().attach_to(base)
# hold
robot_instance.hold(object, jawwidth=.05)
robot_instance.fk(np.array([0, 0, 0, math.pi/6, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, math.pi/6]))
robot_instance.gen_meshmodel().attach_to(base)
tic = time.time()
result = robot_instance.is_selfcollided()
toc = time.time()
print(result, toc - tic)
# base.run()
# release
robot_instance.release(object, jawwidth=.082)
robot_instance.fk(np.array([0, 0, 0, math.pi/3, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, math.pi/6]))
robot_instance.gen_meshmodel().attach_to(base)
tic = time.time()
result = robot_instance.is_selfcollided()
toc = time.time()
print(result, toc - tic)
base.run()
