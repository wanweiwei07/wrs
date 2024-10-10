from wrs import wd, mgm, mcm, rm, mmd
import wrs.robot_sim.robots.khi.khi_main as khi_dual
import wrs.robot_sim.robots.khi as khi
import os
import pickle

meshname = "workbench.stl"
meshpath = os.path.join(os.getcwd(), "meshes")
savepath = os.path.join(os.getcwd(), "pickles")

base = wd.World(cam_pos=[5, 4, 5], lookat_pos=[.0, 0, .5])
mgm.gen_frame().attach_to(base)
# object
workbench = mcm.CollisionModel(os.path.join(meshpath, meshname))
workbench.rgba = rm.np.array([.5, .5, .5, 1])
gl_pos1 = rm.np.array([-.2, .2, .125])
gl_rotmat1 = rm.rotmat_from_euler(ai=rm.np.pi / 2, aj=0, ak=0, order='rxyz')
workbench.pos = gl_pos1
workbench.rotmat = gl_rotmat1

mgm.gen_frame().attach_to(workbench)
t1_copy = workbench.copy()
t1_copy.attach_to(base)

# object holder goal
workbench2 = mcm.CollisionModel(os.path.join(meshpath, meshname))
gl_pos2 = rm.np.array([-.2, 0, .125])
gl_rotmat2 = rm.rotmat_from_euler(ai=rm.np.pi / 2, aj=0, ak=0, order='rxyz')
workbench2.pos = gl_pos2
workbench2.rotmat = gl_rotmat2

t2_copy = workbench2.copy()
t2_copy.rgb = rm.const.red
t2_copy.alpha = .3
t2_copy.attach_to(base)

# robot
robot = khi_dual.KHI_DUAL(pos=rm.np.array([-.59, 0, -.74]))
mot_data = mmd.MotionData(robot)

# load jv_list data
with open(os.path.join(savepath, meshname.split(".")[0] + "_jv" + ".pickle"), 'rb') as f:
    mot_data._jv_list = pickle.load(f)
# load ev_list data
with open(os.path.join(savepath, meshname.split(".")[0] + "_ev" + ".pickle"), 'rb') as f:
    mot_data._ev_list = pickle.load(f)
# load mesh_list data
with open(os.path.join(savepath, meshname.split(".")[0] + "_mesh" + ".pickle"), 'rb') as f:
    mot_data._mesh_list = pickle.load(f)

print("joint value:", mot_data.jv_list)
print("end_effector value:", mot_data.ev_list)


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
    if base.irm.nputmgr.keymap['space']:
        anime_data.counter += 1
    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()
