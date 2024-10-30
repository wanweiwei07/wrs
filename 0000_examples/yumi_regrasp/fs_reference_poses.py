import os
import time
from wrs import wd, rm, mcm, fsp

mesh_name = "bracketR1"
mesh_path = os.path.join(os.getcwd(), "meshes", mesh_name+".stl")
fsref_pose_path = os.path.join(os.getcwd(), "pickles", mesh_name+"_fsref_pose.pickle")

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
ground = mcm.gen_box(xyz_lengths=rm.vec(.5, .5, .01), pos=rm.vec(0, 0, -0.01))
ground.attach_to(base)
bunny = mcm.CollisionModel(mesh_path)

fs_reference_poses = fsp.FSReferencePoses(obj_cmodel=bunny)
fs_reference_poses.save_to_disk(fsref_pose_path)


class AnimeData(object):
    def __init__(self, poses):
        self.counter = 0
        self.model = poses.obj_cmodel
        self.poses = poses
        self.support_facets = poses.support_surfaces


anime_data = AnimeData(poses=fs_reference_poses)


def update(anime_data, task):
    if anime_data.counter >= len(anime_data.poses):
        anime_data.model.detach()
        anime_data.support_facets[anime_data.counter - 1].detach()
        anime_data.counter = 0
    if base.inputmgr.keymap["space"] is True:
        time.sleep(.1)
        anime_data.model.detach()
        print(anime_data.poses[anime_data.counter])
        anime_data.model.pose = anime_data.poses[anime_data.counter]
        anime_data.model.rgb = rm.const.tab20_list[1]
        anime_data.model.alpha = .3
        anime_data.model.attach_to(base)
        if (anime_data.support_facets is not None):
            if anime_data.counter > 0:
                anime_data.support_facets[anime_data.counter - 1].detach()
            anime_data.support_facets[anime_data.counter].pose = anime_data.poses[anime_data.counter]
            anime_data.support_facets[anime_data.counter].attach_to(base)
        anime_data.counter += 1
    return task.cont


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)
base.run()
