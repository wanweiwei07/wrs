from wrs import wd, rm, rrtc, mcm, mgm, gg, x6wg2, ppp

base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
mgm.gen_frame().attach_to(base)
# ground
ground = mcm.gen_box(xyz_lengths=[5, 5, 0.1],
                     pos=rm.np.array([0, 0, -0.1]),
                     rgb=rm.np.array([.7, .7, .7]),
                     alpha=.7)
ground.pos = rm.np.array([0, 0, -.51])
ground.attach_to(base)
# object
tube1 = mcm.CollisionModel("objects/tubebig.stl")
tube1.rgba = rm.np.array([.5, .5, .5, 1])
gl_pos1 = rm.np.array([.3, -.05, .0])
gl_rotmat1 = rm.rotmat_from_euler(0, 0, rm.pi / 2)
tube1.pos = gl_pos1
tube1.rotmat = gl_rotmat1

mgm.gen_frame().attach_to(tube1)
t1_copy = tube1.copy()
t1_copy.attach_to(base)

# object holder goal
tube2 = mcm.CollisionModel("objects/tubebig.stl")
gl_pos2 = rm.np.array([.3, .05, .0])
gl_rotmat2 = rm.rotmat_from_euler(0, 0, 2 * rm.pi / 3)
tube2.pos = gl_pos2
tube2.rotmat = gl_rotmat2

t2_copy = tube2.copy()
t2_copy.rgb = rm.const.tab20_list[0]
t2_copy.alpha = .3
t2_copy.attach_to(base)

robot = x6wg2.XArmLite6WG2()
robot.gen_meshmodel().attach_to(base)
print(robot.end_effector.jaw_width)

rrtc = rrtc.RRTConnect(robot)
ppp = ppp.PickPlacePlanner(robot)

grasp_collection = gg.GraspCollection.load_from_disk(file_name='wrs_gripper2_grasps.pickle')
start_conf = robot.get_jnt_values()
print(grasp_collection)
mot_data = ppp.gen_pick_and_place(obj_cmodel=tube1,
                                  grasp_collection=grasp_collection,
                                  end_jnt_values=start_conf,
                                  goal_pose_list=[(gl_pos2, gl_rotmat2)],
                                  obstacle_list=[ground])


class Data(object):
    def __init__(self, mot_data):
        self.counter = 0
        self.mot_data = mot_data


anime_data = Data(mot_data)


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
