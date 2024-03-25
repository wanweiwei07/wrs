import os
import time
import basis.robot_math as rm
import numpy as np
import visualization.panda.world as wd
import modeling.collision_model as mcm
import manipulation.placement as mpl
import grasping.grasp as g
import robot_sim.robots.xarmlite6_wg.x6wg2 as x6wg2

if __name__ == '__main__':

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    obj_path = os.path.join("objects", "bunnysim.stl")
    ground = mcm.gen_box(xyz_lengths=[5, 5, .01], pos=np.array([0, 0, -0.005]))
    ground.attach_to(base)
    bunny = mcm.CollisionModel(obj_path)
    robot = x6wg2.XArmLite6WG2()

    reference_fsp_poses = mpl.ReferenceFSPPoses.load_from_disk(file_name="reference_fsp_poses_bunny.pickle")
    reference_grasps = g.GraspCollection.load_from_disk(file_name="reference_wg2_bunny_grasps.pickle")
    fspgspots_col = mpl.SpotFSPGsCollection(robot=robot,
                                            obj_cmodel=bunny,
                                            reference_fsp_poses=reference_fsp_poses,
                                            reference_grasp_collection=reference_grasps)
    # fspgspots_col.add_new_spot(spot_pos=np.array([.4,0,0]), spot_rotz=0)
    # fspgspots_col.save_to_disk("spotfspgs_col_x6wg2_bunny.pickle")
    fspgspots_col.load_from_disk("spotfspgs_col_x6wg2_bunny.pickle")
    fspgspots_col.add_new_spot(spot_pos=np.array([.4,.2,0]), spot_rotz=0)
    fspgspots_col.save_to_disk("spotfspgs_col_x6wg2_bunny.pickle")
    mesh_model_list = fspgspots_col.gen_meshmodels()
    for fsregspot in fspgspots_col:
        mcm.mgm.gen_frame(pos=fsregspot.spot_pos,
                          rotmat=rm.rotmat_from_euler(0, 0, fsregspot.spot_rotz)).attach_to(base)


    class Data(object):
        def __init__(self, mesh_model_list):
            self.counter = 0
            self.mesh_model_list = mesh_model_list


    anime_data = Data(mesh_model_list=mesh_model_list)


    def update(anime_data, task):
        if anime_data.counter > 0:
            anime_data.mesh_model_list[anime_data.counter - 1].detach()
        if anime_data.counter >= len(anime_data.mesh_model_list):
            # for mesh_model in anime_data.mot_data.mesh_list:
            #     mesh_model.detach()
            anime_data.counter = 0
        anime_data.mesh_model_list[anime_data.counter].attach_to(base)
        if base.inputmgr.keymap['space']:
            print(anime_data.counter)
            anime_data.counter += 1
            time.sleep(.5)
        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[anime_data],
                          appendTask=True)

    base.run()
