import math
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.ur3_dual.ur3_dual as ur3d
import motion.probabilistic.rrt_connect as rrtc
import robot_con.ur.ur3_dual_x as ur3dx

base = wd.World(cam_pos=[2, 1, 3], lookat_pos=[0, 0, 1.1])
gm.gen_frame().attach_to(base)
# object
object = cm.CollisionModel("./objects/bunnysim.stl")
object.set_pos(np.array([.55, -.3, 1.3]))
object.set_rgba([.5, .7, .3, 1])
object.attach_to(base)
# robot_s
component_name = 'both_arm'
robot_instance = ur3d.UR3Dual()
# init_lft_arm_jnt_values = robot_s.lft_arm.get_jnt_values()
# init_rgt_arm_jnt_values = robot_s.rgt_arm.get_jnt_values()
# full_jnt_values = np.hstack((init_lft_arm_jnt_values, init_rgt_arm_jnt_values))
full_jnt_values = np.hstack((robot_instance.lft_arm.homeconf, robot_instance.rgt_arm.homeconf))
goal_lft_arm_jnt_values = np.array([0, -math.pi / 2, -math.pi/3, -math.pi / 2, math.pi / 6, math.pi / 6])
goal_rgt_arm_jnt_values = np.array([0, -math.pi/4, 0, math.pi/2, math.pi/2, math.pi / 6])

rrtc_planner = rrtc.RRTConnect(robot_instance)
path = rrtc_planner.plan(component_name=component_name,
                         start_conf=full_jnt_values,
                         goal_conf=np.hstack((goal_lft_arm_jnt_values, goal_rgt_arm_jnt_values)),
                         obstacle_list=[object],
                         ext_dist=.2,
                         max_time=300)
print(path)
for pose in path:
    print(pose)
    robot_instance.fk(component_name, pose)
    robot_meshmodel = robot_instance.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    robot_instance.gen_stickmodel().attach_to(base)

ur_dual_x = ur3dx.UR3DualX(lft_robot_ip='10.2.0.50', rgt_robot_ip='10.2.0.51', pc_ip='10.2.0.100')
ur_dual_x.move_jspace_path(path)

base.run()