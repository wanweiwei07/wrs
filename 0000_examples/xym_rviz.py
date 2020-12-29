import os
import time
import math
import basis
import numpy as np
from basis import robotmath as rm
import visualization.panda.world as wd
import modeling.geometricmodel as gm
import modeling.collisionmodel as cm
import robotsim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xav
import motion.probabilistic.rrt_connect as rrtc
import visualization.panda.rpc.rviz_client as rv_client

rvc = rv_client.RVizClient(host="192.168.1.111:182001")

# remote code
rvc.run_code("import os")
rvc.run_code("import math")
rvc.run_code("import basis")
rvc.run_code("import numpy as np")
rvc.run_code("import modeling.geometricmodel as gm")
rvc.run_code("import modeling.collisionmodel as cm")
rvc.run_code("import robotsim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xav")

create_attach_global_frame = \
'''
global_frame = gm.gen_frame()
global_frame.attach_to(base)
'''
rvc.run_code(create_attach_global_frame)

create_attach_object = \
'''
objpath = os.path.join(basis.__path__[0], 'objects', '%s')
object = cm.CollisionModel(objpath)
object.set_pos(%s)
object.set_color(%s)
object.attach_to(base)
''' % ('bunnysim.stl', 'np.array([.85, 0, .17])', '[.5,.7,.3,1]')
rvc.run_code(create_attach_object)

create_attach_robot = \
'''
jlc_name='arm'
robot_instance = xav.XArm7YunjiMobile()
robot_instance.fk(np.array([0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0]), jlc_name=jlc_name)
robot_meshmodel = robot_instance.gen_meshmodel()
robot_meshmodel.attach_to(base)
'''
rvc.run_code(create_attach_robot)

# local code
objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
object = cm.CollisionModel(objpath)
object.set_pos(np.array([.85, 0, .17]))
object.set_color([.5,.7,.3,1])
jlc_name='arm'
robot_instance = xav.XArm7YunjiMobile()
robot_instance.fk(np.array([0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0]), jlc_name=jlc_name)
rrtc_planner = rrtc.RRTConnect(robot_instance)
path = rrtc_planner.plan(start_conf=np.array([0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0]),
                         goal_conf=np.array([math.pi/3, math.pi * 1 / 3, 0, math.pi/2, 0, math.pi / 6, 0]),
                         obstacle_list=[object],
                         ext_dist=.1,
                         rand_rate=70,
                         maxtime=300,
                         jlc_name=jlc_name)

while True:
    for pose in path:
        update_robot = \
'''
robot_meshmodel.detach()
robot_instance.fk(np.array(%s), jlc_name=jlc_name)
robot_meshmodel = robot_instance.gen_meshmodel()
robot_meshmodel.attach_to(base)
''' % np.array2string(pose, separator=',')
        rvc.run_code(update_robot)
        # robot_meshmodel = robot_instance.gen_meshmodel()
        # robot_meshmodel.attach_to(base)
        # # robot_meshmodel.show_cdprimit()
        # robot_instance.gen_stickmodel().attach_to(base)
        time.sleep(.04)


# print(path)
# for pose in path:
#     print(pose)
#     robot_instance.fk(pose, jlc_name=jlc_name)
#     robot_meshmodel = robot_instance.gen_meshmodel()
#     robot_meshmodel.attach_to(base)
#     # robot_meshmodel.show_cdprimit()
#     robot_instance.gen_stickmodel().attach_to(base)
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
