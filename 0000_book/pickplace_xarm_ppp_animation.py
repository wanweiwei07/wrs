from wrs import wd, rm, ppp, rrtc, mgm, mcm, x6wg2, gg

base = wd.World(cam_pos=[2.1, -2.1, 2.1], lookat_pos=[.0, 0, .3])
mgm.gen_frame().attach_to(base)
# ground
ground = mcm.gen_box(xyz_lengths=[5, 5, 1], rgb=rm.vec(.57, .57, .5), alpha=.7)
ground.pos = rm.vec(0, 0, -.51)
ground.attach_to(base)
# object box
object_box = mcm.gen_box(xyz_lengths=[.02, .04, .7], rgb=rm.vec(.7, .5, .3), alpha=.7)
# object_box_gl_pos = np.array([.3, -.4, .35])
# object_box_gl_rotmat = np.eye(3)
object_box_gl_pos = rm.vec(.3, -.2, .01)
object_box_gl_rotmat = rm.rotmat_from_euler(rm.pi/2, rm.pi/2, 0)
obgl_start_homomat = rm.homomat_from_posrot(object_box_gl_pos, object_box_gl_rotmat)
object_box.pos = object_box_gl_pos
object_box.rotmat = object_box_gl_rotmat
mgm.gen_frame().attach_to(object_box)
object_box_copy = object_box.copy()
object_box_copy.attach_to(base)
# object box goal
# object_box_gl_goal_pos = np.array([.6, -.1, .1])
# object_box_gl_goal_rotmat = rm.rotmat_from_euler(0, math.pi / 2, math.pi / 2)
object_box_gl_goal_pos = rm.vec(.25, -.25, .01)
object_box_gl_goal_rotmat = rm.rotmat_from_euler(0, rm.pi / 2, rm.pi / 2)
obgl_goal_homomat = rm.homomat_from_posrot(object_box_gl_goal_pos, object_box_gl_goal_rotmat)
object_box_goal_copy = object_box.copy()
object_box_goal_copy.homomat = obgl_goal_homomat
object_box_goal_copy.attach_to(base)

rbt = x6wg2.XArmLite6WG2()
rbt.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
# base.run()
rrtc = rrtc.RRTConnect(rbt)
ppp = ppp.PickPlacePlanner(rbt)

grasp_collection = gg.GraspCollection.load_from_disk(file_name='wg2_long_box.pickle')
start_conf = rbt.get_jnt_values()
mot_data = ppp.gen_pick_and_place(obj_cmodel=object_box,
                                  grasp_collection=grasp_collection,
                                  pick_approach_distance = .05,
                                  pick_depart_distance = .05,
                                  pick_depart_direction = rm.vec(0,0,1),
                                  # end_jnt_values=start_conf,
                                  goal_pose_list=[(object_box_gl_goal_pos, object_box_gl_goal_rotmat)],
                                  obstacle_list=[ground],
                                  toggle_dbg=False)

class Data(object):
    def __init__(self, mot_data):
        self.counter = 0
        self.mot_data = mot_data


anime_data = Data(mot_data)
print(mot_data.oiee_gl_pose_list)

anime_data.mot_data.mesh_list[0].attach_to(base)
anime_data.mot_data.mesh_list[-1].attach_to(base)
# for mesh_model in anime_data.mot_data.mesh_list[1:-2:2]:
#     mesh_model.alpha=.7
#     mesh_model.attach_to(base)
# base.run()

def update(anime_data, task):
    if anime_data.counter > 0:
        anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
    if anime_data.counter >= len(anime_data.mot_data):
        # for mesh_model in anime_data.mot_data.mesh_list:
        #     mesh_model.detach()
        anime_data.counter = 0
    mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
    mesh_model.attach_to(base)
    if base.inputmgr.keymap['space']:
        anime_data.counter += 1
    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()