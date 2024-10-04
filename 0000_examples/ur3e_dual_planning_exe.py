import numpy as np
import wrs.visualization.panda.world as wd
import wrs.robot_sim.robots.ur3e_dual.ur3e_dual as u3ed
from wrs import basis as rm, motion as rrtc, modeling as mgm, modeling as mcm
import wrs.robot_con.ur.ur3e_rtqhe_x as u3erhex

class Data(object):
    def __init__(self, mot_data):
        self.counter = 0
        self.mot_data = mot_data


base = wd.World(cam_pos=[3, 2, 3], lookat_pos=[0.5, 0, 1.1])
mgm.gen_frame().attach_to(base)
# robot
robot = u3ed.UR3e_Dual()
robot.use_lft()
# planner
rrtc_planner = rrtc.RRTConnect(robot)
# executor
robotx = u3erhex.UR3ERtqHEX(robot_ip="10.0.2.2", pc_ip="10.0.2.17")

# # recover planner
start_conf = np.asarray(robotx.arm.getj())
# robot.goto_given_conf(jnt_values=start_conf)
# robot.gen_meshmodel().attach_to(base)
# robot.show_cdprim()
# print(robot.is_collided())
# base.run()

# goal_conf = robot.delegator.home_conf
# # planning
# mot_data = rrtc_planner.plan(start_conf=start_conf,
#                              goal_conf=goal_conf,
#                              obstacle_list=[],
#                              ext_dist=.1,
#                              max_time=30,
#                              smoothing_n_iter=100)
# anime_data = Data(mot_data)
#
# def update(anime_data, robotx, task):
#     if anime_data.counter > 0:
#         anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
#     if anime_data.counter >= len(anime_data.mot_data):
#         anime_data.counter = 0
#     mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
#     mesh_model.attach_to(base)
#     mesh_model.show_cdprim()
#     if base.inputmgr.keymap['space']:
#         anime_data.counter += 1
#     if base.inputmgr.keymap['g']:
#         robotx.move_jntspace_path(anime_data.mot_data.jv_list, control_frequency=.005)
#     return task.again
#
#
# taskMgr.doMethodLater(0.01, update, "update",
#                       extraArgs=[anime_data, robotx],
#                       appendTask=True)
#
# base.run()

# obstacle
obstacle = mcm.gen_box(xyz_lengths=[.2, .05, .4])
obstacle.pos = np.array([.8, .2, .98])
obstacle.rgba = np.array([.7, .7, .3, 1])
obstacle.attach_to(base)
# plan
# start_conf = robot.get_jnt_values()
tgt_pos = np.array([.8, .1, 1])
tgt_rotmat = rm.rotmat_from_euler(np.pi, 0, np.pi/2)
mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
jnt_values = robot.ik(tgt_pos, tgt_rotmat, obstacle_list=[obstacle], toggle_dbg=False)
robot.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
if jnt_values is None:
    print("No IK solution found!")
    base.run()
# else:
#     robot.goto_given_conf(jnt_values=jnt_values)
#     robot.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
#     base.run()
goal_conf = jnt_values
robot.goto_given_conf(jnt_values=start_conf)

print(start_conf)
print(jnt_values)
mot_data = rrtc_planner.plan(start_conf=start_conf,
                             goal_conf=goal_conf,
                             obstacle_list=[obstacle],
                             ext_dist=.1,
                             max_time=30,
                             smoothing_n_iter=100)
anime_data = Data(mot_data)


def update(anime_data, task):
    if anime_data.counter > 0:
        anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
    if anime_data.counter >= len(anime_data.mot_data):
        anime_data.counter = 0
    mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
    mesh_model.attach_to(base)
    mesh_model.show_cdprim()
    if base.inputmgr.keymap['space']:
        anime_data.counter += 1
    if base.inputmgr.keymap['g']:
        robotx.move_jntspace_path(anime_data.mot_data.jv_list, control_frequency=.005)
    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()
