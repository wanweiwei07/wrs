import os
import time
import basis.robot_math as rm
import numpy as np
import visualization.panda.world as wd
import modeling.collision_model as mcm
import manipulation.placement.flat_surface_placement as mpfsp
import bench_mark.ycb as ycb
import manipulation.regrasp as reg


class AnimeData(object):
    def __init__(self, reference_fsp_poses):
        self.counter = 0
        self.model = reference_fsp_poses.obj_cmodel
        self.reference_fsp_poses = reference_fsp_poses
        self.support_facets = reference_fsp_poses.support_surfaces


if __name__ == '__main__':

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    ground = mcm.gen_box(xyz_lengths=[.5, .5, .01], pos=np.array([0, 0, -0.01]))
    anime_data_list = []
    ground.attach_to(base)
    # # if rebuild
    # for file in ycb.all_files:
    #     obj_cmodel = mcm.CollisionModel(file)
    #     reference_fsp_poses = mpfsp.ReferenceFSPPoses(obj_cmodel=obj_cmodel)
    #     reference_fsp_poses.save_to_disk(f"reference_fsp_poses_{os.path.basename(file)}.pickle")
    #     anime_data_list.append(AnimeData(reference_fsp_poses=reference_fsp_poses))

    for file_name in os.listdir(os.getcwd()):
        # Check if the file is a pickle file
        if file_name.endswith('.pickle'):
            reference_fsp_poses = reg.mpfsp.ReferenceFSPPoses.load_from_disk(file_name=file_name)
            file = ycb.all_files[file_name.split('.stl', 1)[0].split('_', 3)[-1]]
            obj_cmodel = mcm.CollisionModel(file)
            anime_data = AnimeData(reference_fsp_poses=reference_fsp_poses)
            anime_data.model = obj_cmodel
            anime_data_list.append(anime_data)

    def update(obj_cnter, anime_data_list, task):
        if obj_cnter[0] >= len(anime_data_list):
            obj_cnter[0] = 0
        anime_data = anime_data_list[obj_cnter[0]]
        if anime_data.counter >= len(anime_data.reference_fsp_poses):
            anime_data.model.detach()
            # anime_data.support_facets[anime_data.counter - 1].detach()
            anime_data.counter = 0
            obj_cnter[0] += 1
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


    obj_cnter = [0]
    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[obj_cnter, anime_data_list],
                          appendTask=True)
    base.run()
