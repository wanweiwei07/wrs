import numpy as np
import wrs.visualization.panda.world as wd
from wrs import robot_sim as ur3d, motion as rrtc, modeling as mgm, modeling as mcm
import wrs.robot_con.ur.ur3_dual_x as ur3dx


class Data(object):
    def __init__(self):
        self.counter = 0
        self.mot_data = None

base = wd.World(cam_pos=[2, 1, 3], lookat_pos=[0, 0, 1.1])
mgm.gen_frame().attach_to(base)
# object
object = mcm.gen_box(xyz_lengths=np.array([.3, .1, .4]))
object.pos = np.array([.55, -.3, 1.5])
object.rgba = np.array([.5, .7, .3, 1])
object.attach_to(base)
object.show_cdprim()
# rbt
rbt = ur3d.UR3Dual()
# goal
# goal_lft_arm_jnt_values = np.array([0, -math.pi / 2, -math.pi / 3, -math.pi / 2, math.pi / 6, math.pi / 6])
# goal_rgt_arm_jnt_values = np.array([0, -math.pi / 4, 0, math.pi / 2, math.pi / 2, math.pi / 6])
goal_lft_arm_jnt_values = rbt.lft_arm.home_conf
goal_rgt_arm_jnt_values = rbt.rgt_arm.home_conf

goal_jnt_values = np.hstack((goal_lft_arm_jnt_values, goal_rgt_arm_jnt_values))
# rbt x
rbt_x = ur3dx.UR3DualX()
# init
init_lft_arm_jnt_values = rbt_x.lft_arm.get_jnt_values()
init_rgt_arm_jnt_values = rbt_x.rgt_arm.get_jnt_values()
init_jnt_values_x = np.hstack((init_lft_arm_jnt_values, init_rgt_arm_jnt_values))
rbt.goto_given_conf(jnt_values=init_jnt_values_x)
rbt.gen_meshmodel(toggle_cdprim=True).attach_to(base)
# base.run()
print(rbt.is_collided())
init_lft_jaw_width = rbt_x.lft_arm.get_jaw_width()
rbt.lft_arm.change_jaw_width(init_lft_jaw_width)
init_rgt_jaw_width = rbt_x.rgt_arm.get_jaw_width()
rbt.rgt_arm.change_jaw_width(init_rgt_jaw_width)
rbt.gen_meshmodel(toggle_cdprim=True).attach_to(base)

rrtc_planner = rrtc.RRTConnect(rbt)
mot_data = rrtc_planner.plan(start_conf=init_jnt_values_x,
                             goal_conf=goal_jnt_values,
                             obstacle_list=[],
                             ext_dist=.2,
                             max_time=30,
                             smoothing_n_iter=100)


anime_data = Data()
anime_data.mot_data = mot_data

def update(anime_data, rbt_x, task):
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
        rbt_x.move_jspace_path(anime_data.mot_data.jv_list, ctrl_freq=.008)
    return task.again

taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[anime_data, rbt_x],
                      appendTask=True)

base.run()

# ur_dual_x = ur3dx.UR3DualX(lft_robot_ip='10.2.0.50', rgt_robot_ip='10.2.0.51', pc_ip='10.2.0.100')
# ur_dual_x.move_jspace_path(path)

# base.run()
