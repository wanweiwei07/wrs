import math
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.xarm_shuidi.xarm_shuidi as xav
import motion.probabilistic.rrt_connect as rrtc
import robot_con.xarm_shuidi_grpc.xarm.xarm_client as xac

base = wd.World(cam_pos=[3, 1, 2], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
object = cm.CollisionModel("./objects/bunnysim.stl")
object.set_pos(np.array([.85, 0, .37]))
object.set_rgba([.5,.7,.3,1])
object.attach_to(base)
# robot_s
component_name='arm'
robot_s = xav.XArmShuidi()
robot_s.fk(component_name, np.array([0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0]))
# robot_x
robot_x = xac.XArm7(host="192.168.50.77:18300")
init_jnt_angles = robot_x.get_jnt_vlaues()
print(init_jnt_angles)
rrtc_planner = rrtc.RRTConnect(robot_s)
path = rrtc_planner.plan(start_conf=init_jnt_angles,
                         # goal_conf=np.array([math.pi/3, math.pi * 1 / 3, 0, math.pi/2, 0, math.pi / 6, 0]),
                         goal_conf = robot_s.manipulator_dict['arm'].homeconf,
                         obstacle_list=[object],
                         ext_dist= .1,
                         max_time=300,
                         component_name=component_name)
robot_x.move_jspace_path(path, time_interval=.1)

# print(path)
for pose in path:
    # print(pose)
    robot_s.fk(component_name, pose)
    robot_meshmodel = robot_s.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    # robot_meshmodel.show_cdprimit()
    robot_s.gen_stickmodel().attach_to(base)
# hol1
# robot_s.hold(object, jaw_width=.05)
# robot_s.fk(np.array([0, 0, 0, math.pi/6, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, math.pi/6]))
# robot_meshmodel = robot_s.gen_meshmodel()
# robot_meshmodel.attach_to(base)
# robot_s.show_cdprimit()
# tic = time.time()
# result = robot_s.is_collided() # problematic
# toc = time.time()
# print(result, toc - tic)
# base.run()
# release
# robot_s.release(object, jaw_width=.082)
# robot_s.fk(np.array([0, 0, 0, math.pi/3, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, math.pi/6]))
# robot_meshmodel = robot_s.gen_meshmodel()
# robot_meshmodel.attach_to(base)
# robot_meshmodel.show_cdprimit()
# tic = time.time()
# result = robot_s.is_collided()
# toc = time.time()
# print(result, toc - tic)

#copy
# robot_instance2 = robot_s.copy()
# robot_instance2.move_to(pos=np.array([.5,0,0]), rotmat=rm.rotmat_from_axangle([0,0,1], math.pi/6))
# objcm_list = robot_instance2.get_hold_objlist()
# robot_instance2.release(objcm_list[-1], jaw_width=.082)
# robot_meshmodel = robot_instance2.gen_meshmodel()
# robot_meshmodel.attach_to(base)
# robot_instance2.show_cdprimit()

base.run()
