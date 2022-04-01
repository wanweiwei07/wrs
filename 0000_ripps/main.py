import os
import robot_sim.robots.cobotta.cobotta_ripps as cbtr
import modeling.collision_model as cm
import visualization.panda.world as wd
import modeling.geometric_model as gm


if __name__ == '__main__':

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    gm.gen_frame().attach_to(base)

    this_dir, this_filename = os.path.split(__file__)
    file_frame = os.path.join(this_dir, "meshes", "frame_bottom.stl")
    frame_bottom = cm.CollisionModel(file_frame)
    frame_bottom.set_rgba([.55, .55, .55, 1])
    frame_bottom.attach_to(base)

    rbt_s = cbtr.CobottaRIPPS()
    rbt_s.gen_meshmodel().attach_to(base)
    base.run()
