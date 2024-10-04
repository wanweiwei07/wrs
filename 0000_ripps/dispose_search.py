import os
import numpy as np
from wrs import basis as rm, robot_sim as cbtr, robot_sim as cbtg, modeling as cm, modeling as gm
import wrs.visualization.panda.world as wd
import utils

if __name__ == '__main__':
    base = wd.World(cam_pos=[1.7, -1, .7], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)

    this_dir, this_filename = os.path.split(__file__)
    file_frame = os.path.join(this_dir, "meshes", "frame_bottom.stl")
    frame_bottom = cm.CollisionModel(file_frame)
    frame_bottom.set_rgba([.55, .55, .55, 1])
    frame_bottom.attach_to(base)

    table_plate = cm.gen_box(xyz_lengths=[.405, .26, .003])
    table_plate.set_pos([0.07 + 0.2025, .055, .0015])
    table_plate.set_rgba([.87, .87, .87, 1])
    table_plate.attach_to(base)

    file_dispose_box = os.path.join(this_dir, "objects", "tip_rack_cover.stl")
    dispose_box = cm.CollisionModel(file_dispose_box)
    dispose_box.set_rgba([140 / 255, 110 / 255, 170 / 255, 1])
    dispose_box.set_pos(pos=np.array([.14, 0.07, .003]))
    dispose_box.attach_to(base)

    rbt_s = cbtr.CobottaRIPPS()
    component_name = "arm"
    file_tip = os.path.join(this_dir, "objects", "tip.stl")
    tip = cm.CollisionModel(file_tip)
    tip.set_rgba([200 / 255, 180 / 255, 140 / 255, 1])
    pos, rotmat = rbt_s.get_gl_tcp(manipulator_name=component_name)
    tip.set_pose(pos, rm.rotmat_from_axangle(rotmat[:, 0], np.pi).dot(rotmat))
    rbt_s.hold(hnd_name="hnd", objcm=tip)
    ee_s = cbtg.CobottaPipette()

    pos = dispose_box.get_pos() + np.array([0, 0.05, .02])
    z_offset = np.array([0, 0.0, .03])
    rotmat = rm.rotmat_from_axangle([1, 0, 0], -np.pi * 4 / 9).dot(rm.rotmat_from_axangle([0, 1, 0], -np.pi))
    utils.search_reachable_configuration(rbt_s=rbt_s,
                                         ee_s=ee_s,
                                         component_name=component_name,
                                         tgt_pos=pos+z_offset,
                                         cone_axis=rotmat[:3,2],
                                         cone_angle=np.pi/18,
                                         rotation_interval=np.radians(22.5),
                                         obstacle_list=[frame_bottom],
                                         toggle_debug=True)
    base.run()
