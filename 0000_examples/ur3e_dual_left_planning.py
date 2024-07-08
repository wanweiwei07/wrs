import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as mgm
import modeling.collision_model as mcm
import robot_sim.robots.ur3e_dual.ur3e_dual as u3ed
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm


class Data(object):
    def __init__(self, mot_data):
        self.counter = 0
        self.mot_data = mot_data


base = wd.World(cam_pos=[3, 2, 3], lookat_pos=[0.5, 0, 1.1])
mgm.gen_frame().attach_to(base)
# robot
robot = u3ed.UR3e_Dual()
robot.use_lft()
# obstacle
obstacle = mcm.gen_box(xyz_lengths=[.2, .05, .4])
obstacle.pos = np.array([.8, .2, .98])
obstacle.rgba = np.array([.7, .7, .3, 1])
obstacle.attach_to(base)
# planner
rrtc_planner = rrtc.RRTConnect(robot)
# plan
start_conf = robot.get_jnt_values()
tgt_pos = np.array([.8, .1, 1])
tgt_rotmat = rm.rotmat_from_euler(np.pi, 0, 0)
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
    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()
