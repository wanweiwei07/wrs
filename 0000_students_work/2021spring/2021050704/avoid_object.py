import math
import numpy as np
import wrs.visualization.panda.world as wd
from wrs import basis as rm, robot_sim as ur3d, motion as rrtc, modeling as gm, modeling as cm

base = wd.World(cam_pos=[2, 1, 3], lookat_pos=[0, 0, 1.1])
gm.gen_frame().attach_to(base)

# オブジェクト
object1 = cm.CollisionModel("./my_objects/LEGO.stl")
object1.set_pos(np.array([.4, -.4, 1.11]))
object1.set_rgba([0, 1, 0, 1])
object1.attach_to(base)

object2 = cm.CollisionModel("./my_objects/hand_spinner.stl")
object2.set_pos(np.array([.5, -.6, 1.2]))
object2.set_rgba([0, 0, 1, 1])
object2.attach_to(base)

# ロボットの定義
component_name = 'rgt_arm'
robot_s = ur3d.UR3Dual()

#スタート姿勢
s_pos = np.array([.4, -.7, 1.11])
s_rotmat = rm.rotmat_from_euler(ai=math.pi/2,aj=math.pi,ak=math.pi,axes='szxz')
s_conf = robot_s.ik(component_name=component_name, tgt_pos=s_pos, tgt_rotmat=s_rotmat)

#ゴール姿勢
g_pos = np.array([.4, -.3, 1.11])
g_rotmat = rm.rotmat_from_euler(ai=math.pi/2,aj=math.pi,ak=0,axes='szxz')
g_conf = robot_s.ik(component_name=component_name, tgt_pos=g_pos, tgt_rotmat=g_rotmat)


# possible right goal np.array([0, -math.pi/4, 0, math.pi/2, math.pi/2, math.pi / 6])
# possible left goal np.array([0, -math.pi / 2, -math.pi/3, -math.pi / 2, math.pi / 6, math.pi / 6])

#ロボットの経路計画
rrtc_planner = rrtc.RRTConnect(robot_s)
path = rrtc_planner.plan(component_name=component_name,
                         start_conf=s_conf,
                         goal_conf=g_conf,
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
