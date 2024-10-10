from wrs import wd, rm, rrtc, mcm, mgm, gg, ppp, rrtc
import wrs.robot_sim.robots.khi.khi_main as khi_dual
import wrs.modeling.constant as const
import wrs.robot_sim.robots.khi as khi
import os
import pickle

mesh_name = "bracketR1.stl"
mesh_path = os.path.join(os.getcwd(), "meshes")
save_path = os.path.join(os.getcwd(), "pickles")

base = wd.World(cam_pos=[5, 4, 5], lookat_pos=[.0, 0, .5])
mgm.gen_frame().attach_to(base)
home_pos = rm.np.array([-0.09, 0, 0.022])

# object
workbench = mcm.CollisionModel(os.path.join(mesh_path, "workbench.stl"), rgb=rm.const.orange_red,
                               cdmesh_type=const.CDMeshType.DEFAULT)
workbench.pos = rm.np.array([-.2, 0, .125])
workbench.rotmat = rm.rotmat_from_euler(ai=rm.np.pi / 2, aj=0, ak=0, order='rxyz')
workbench.attach_to(base)

bracketR1 = mcm.CollisionModel(os.path.join(mesh_path, "bracketR1.stl"), rgb=rm.const.gray,
                               cdprim_type=const.CDPrimType.AABB)
bracketR1.pos = rm.np.array([0, .2, .4])
bracketR1.rotmat = rm.rotmat_from_euler(ai=rm.np.pi / 2, aj=0, ak=0, order='rxyz')

# object holder goal
bracketR1_gl = mcm.CollisionModel(os.path.join(mesh_path, "bracketR1.stl"), alpha=.3)
bracketR1_gl.pos = workbench.pos + home_pos
bracketR1_gl.rotmat = rm.rotmat_from_euler(ai=rm.np.pi / 2, aj=0, ak=0, order='rxyz')

# copy to view original object
cpy_start = bracketR1.copy()
cpy_start.attach_to(base)
cpy_goal = bracketR1_gl.copy()
cpy_goal.attach_to(base)

# robot
robot = khi_dual.KHI_DUAL(pos=rm.np.array([-.59, 0, -.74]))
robot.use_lft()

print(robot.pos)

rrtc = rrtc.RRTConnect(robot)
ppp = ppp.PickPlacePlanner(robot)

# robot.gen_meshmodel().attach_to(base)
# base.run()

grasp_collection = gg.GraspCollection.load_from_disk(
    file_name=os.path.join(save_path, mesh_name.split(".")[0] + ".pickle"))
start_conf = robot.get_jnt_values()
print(grasp_collection)

mot_data = ppp.gen_pick_and_place(obj_cmodel=bracketR1,
                                  grasp_collection=grasp_collection,
                                  end_jnt_values=start_conf,
                                  goal_pose_list=[(bracketR1_gl.pos, bracketR1_gl.rotmat)],
                                  obstacle_list=[workbench])
# obstacle_list=[mcm.gen_box(xyz_lengths=[.1,.01,.25], pos=rm.vec(-.2,0,.13))])

if mot_data is None:
    raise ValueError("mot_data is None. Failed to generate pick and place motion data.")

# save jv_list data as pickle
print("do you wanna save the data? : y: yes other: no")
if input() == "y":
    with open(os.path.join(save_path, mesh_name.split(".")[0] + "_jv" + ".pickle"), 'wb') as f:
        pickle.dump(mot_data.jv_list, f)
    # save ev_list data as pickle
    with open(os.path.join(save_path, mesh_name.split(".")[0] + "_ev" + ".pickle"), 'wb') as f:
        pickle.dump(mot_data.ev_list, f)
    # load mesh_list data
    with open(os.path.join(save_path, mesh_name.split(".")[0] + "_mesh" + ".pickle"), 'wb') as f:
        pickle.dump(mot_data.mesh_list, f)
else:
    print("data not saved")


class Data(object):
    def __init__(self, mot_data):
        self.counter = 0
        self.mot_data = mot_data


anime_data = Data(mot_data)

# check collision between robot and workbench
print("robot collided workbench?", robot.is_collided([workbench]))

# anime_data.mot_data.mesh_list[56].attach_to(base)
# base.run()

# check one time
flag1 = False
flag2 = False


def update(anime_data, task):
    global flag1, flag2
    if anime_data.counter > 0:
        anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
        # print("bracket collided workbench?",bracketR1.is_mcdwith(workbench))
        # print("robot collided workbench?",robot.is_mesh_collided(workbench))
    if anime_data.counter >= len(anime_data.mot_data):
        # for mesh_model in anime_data.mot_data.mesh_list:
        #     mesh_model.detach()
        anime_data.counter = 0
        flag1 = True
        flag2 = True
    mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
    mesh_model.attach_to(base)
    # check pick time
    if anime_data.mot_data.ev_list[anime_data.counter - 1] < anime_data.mot_data.ev_list[
        anime_data.counter] and flag1 == False:
        print("pick time:", anime_data.counter)
    # check place time
    if anime_data.mot_data.ev_list[anime_data.counter - 1] > anime_data.mot_data.ev_list[
        anime_data.counter] and flag2 == False:
        print("place time:", anime_data.counter - 1)
    if base.inputmgr.keymap['space']:
        anime_data.counter += 1
    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()
