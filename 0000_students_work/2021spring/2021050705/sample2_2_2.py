import math
import numpy as np
import wrs.visualization.panda.world as wd
from wrs import basis as rm, robot_sim as ur3d, motion as rrtc, modeling as gm, modeling as cm

base = wd.World(cam_pos=[2, 1, 3], lookat_pos=[0, 0, 1.1])
gm.gen_frame().attach_to(base)

# object1の生成
object1 = cm.CollisionModel("objects/01cup.stl")
object1.set_pos(np.array([.30, -.50, 1.10]))
object1.set_rgba([0, 1, 0, 1])
object1.attach_to(base)
# object2の生成
object2 = cm.CollisionModel("objects/02cup.stl")
object2.set_pos(np.array([.40, -.60, 1.30]))
object2.set_rgba([0, 0, 1, 1])
object2.attach_to(base)

# robot_s.l/ln;
component_name = 'rgt_arm'
robot_s = ur3d.UR3Dual()

#逆運動学でstart姿勢を生成
start_pos = np.array([.30, -.30, 1.20])
start_rot = rm.rotmat_from_euler(ai=math.pi/2, aj=math.pi, ak=0, axes='szxz')
start_conf = robot_s.ik(component_name=component_name, tgt_pos=start_pos, tgt_rotmat=start_rot)
#start_conf = robot_s.lft_arm.home

#逆運動学でgoal姿勢を生成
goal_pos = np.array([.30, -.70, 1.20])
goal_rot = rm.rotmat_from_euler(ai=math.pi/2, aj=math.pi, ak=0, axes='szxz')
goal_conf = robot_s.ik(component_name=component_name, tgt_pos=goal_pos, tgt_rotmat=goal_rot)
#end_conf = np.array([0, -math.pi / 2, -math.pi/3, -math.pi / 2, math.pi / 6, math.pi / 6])

#軌道の生成
rrtc_planner = rrtc.RRTConnect(robot_s)
path = rrtc_planner.plan(component_name=component_name,
                         start_conf=start_conf,
                         goal_conf=goal_conf,
                         obstacle_list=[object1, object2],
                         ext_dist=.2,
                         max_time=300)
print(path)
for pose in path:
    print(pose)
    robot_s.fk(component_name, pose)
    robot_meshmodel = robot_s.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    robot_s.gen_stickmodel().attach_to(base)

base.run()
