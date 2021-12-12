import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import math
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.cobotta.cobotta as cbt
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc

base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
gm.gen_frame().attach_to(base)
# ground
ground = cm.gen_box(extent=[5, 5, 1], rgba=[.7, .7, .7, .7])
ground.set_pos(np.array([0, 0, -.51]))
ground.attach_to(base)

robot_s = cbt.Cobotta()
robot_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
seed_jnt_values = None
for z in np.linspace(.1, .6, 5):
    goal_pos = np.array([.25, -.1, z])
    goal_rot = rm.rotmat_from_axangle(np.array([0, 1, 0]), math.pi * 1 / 2)
    gm.gen_frame(goal_pos, goal_rot).attach_to(base)

    jnt_values = robot_s.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rot, seed_jnt_values=seed_jnt_values)
    print(jnt_values)
    if jnt_values is not None:
        robot_s.fk(jnt_values=jnt_values)
        seed_jnt_values = jnt_values
    robot_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
base.run()
