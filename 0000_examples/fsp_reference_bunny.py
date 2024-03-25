import os
import time
import basis.robot_math as rm
import numpy as np
import visualization.panda.world as wd
import modeling.collision_model as mcm
import manipulation.placement as mpl

if __name__ == '__main__':

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    obj_path = os.path.join("objects", "bunnysim.stl")
    ground = mcm.gen_box(xyz_lengths=[.5, .5, .01], pos=np.array([0, 0, -0.01]))
    ground.attach_to(base)
    bunny = mcm.CollisionModel(obj_path)

    reference_fsp_poses = mpl.ReferenceFSPPoses(obj_cmodel=bunny)
    reference_fsp_poses.save_to_disk("reference_fsp_poses_bunny.pickle")


    class AnimeData(object):
        def __init__(self, reference_fsp_poses):
            self.counter = 0
            self.model = reference_fsp_poses.obj_cmodel
            self.reference_fsp_poses = reference_fsp_poses
            self.support_facets = reference_fsp_poses.support_surfaces


    anime_data = AnimeData(reference_fsp_poses=reference_fsp_poses)


    def update(anime_data, task):
        if anime_data.counter >= len(anime_data.reference_fsp_poses):
            anime_data.model.detach()
            anime_data.support_facets[anime_data.counter - 1].detach()
            anime_data.counter = 0
        if base.inputmgr.keymap["space"] is True:
            time.sleep(.1)
            anime_data.model.detach()
            print(anime_data.reference_fsp_poses[anime_data.counter])
            anime_data.model.pose = anime_data.reference_fsp_poses[anime_data.counter]
            anime_data.model.rgb = rm.bc.tab20_list[1]
            anime_data.model.alpha = .3
            anime_data.model.attach_to(base)
            if (anime_data.support_facets is not None):
                if anime_data.counter > 0:
                    anime_data.support_facets[anime_data.counter - 1].detach()
                anime_data.support_facets[anime_data.counter].pose = anime_data.reference_fsp_poses[
                    anime_data.counter]
                anime_data.support_facets[anime_data.counter].attach_to(base)
            anime_data.counter += 1
        return task.cont


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[anime_data],
                          appendTask=True)
    base.run()
