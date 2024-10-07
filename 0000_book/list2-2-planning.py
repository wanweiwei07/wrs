from wrs import wd, rm, ur3ed, mcm, mgm, rrtc


class Data(object):
    def __init__(self, mot_data):
        self.counter = 0
        self.mot_data = mot_data


base = wd.World(cam_pos=[3, 2, 3], lookat_pos=[0.5, 0, 1.1])
mgm.gen_frame().attach_to(base)
# robot
robot = ur3ed.UR3e_Dual()
robot.use_lft()
# obstacle
obstacle = mcm.gen_box(xyz_lengths=[.2, .05, .4])
obstacle.pos = rm.vec(.8, .2, .98)
obstacle.rgba = rm.vec(.7, .7, .3, 1)
obstacle.attach_to(base)
# planner
planner = rrtc.RRTConnect(robot)
# plan
start_conf = robot.get_jnt_values()
tgt_pos = rm.vec(.8, .4, 1)
tgt_rotmat = rm.rotmat_from_euler(rm.pi, 0, 0)
mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
jnt_values = robot.ik(tgt_pos, tgt_rotmat)
if jnt_values is None:
    print("No IK solution found!")
    robot.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    robot.goto_given_conf(jnt_values=robot.jnt_ranges[:, 1])
    robot.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    base.run()
goal_conf = jnt_values
robot.goto_given_conf(jnt_values=start_conf)

print(start_conf)
print(jnt_values)
mot_data = planner.plan(start_conf=start_conf,
                        goal_conf=goal_conf,
                        obstacle_list=[obstacle],
                        ext_dist=.1,
                        max_time=30,
                        smoothing_n_iter=100)
anime_data = Data(mot_data)

# draw traj
for i in range(len(mot_data) - 1):
    pos, rotmat = mot_data.tcp_list[i]
    pos_nxt, rotmat_nxt = mot_data.tcp_list[i + 1]
    mgm.gen_frame(pos=pos, rotmat=rotmat, ax_length=.03).attach_to(base)
    mgm.gen_stick(pos, pos_nxt, rgb=rm.const.gray).attach_to(base)
    if i == len(mot_data) - 2:
        mgm.gen_frame(pos=pos_nxt, rotmat=rotmat_nxt, ax_length=.03).attach_to(base)


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
