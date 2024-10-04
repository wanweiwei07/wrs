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

    file_tip_rack = os.path.join(this_dir, "objects", "tip_rack.stl")
    tip_rack = utils.Rack96(file_tip_rack)
    tip_rack.set_rgba([140 / 255, 110 / 255, 170 / 255, 1])
    tip_rack.set_pose(pos=np.array([.25, 0.0, .003]), rotmat=rm.rotmat_from_axangle([0, 0, 1], np.pi / 2))
    tip_rack.attach_to(base)

    file_tip = os.path.join(this_dir, "objects", "tip.stl")
    tip = cm.CollisionModel(file_tip)
    tip.set_rgba([200 / 255, 180 / 255, 140 / 255, 1])
    for id_x in range(8):
        for id_y in range(12):
            pos, rotmat = tip_rack.get_rack_hole_pose(id_x=id_x, id_y=id_y)
            tip_new = tip.copy()
            tip_new.set_pose(pos, rotmat)
            tip_new.attach_to(base)

    rbt_s = cbtr.CobottaRIPPS()
    ee_s = cbtg.CobottaPipette()

    pos, rotmat = tip_rack.get_rack_hole_pose(id_x=6, id_y=3)
    z_offset = np.array([0,0,.03])
    utils.search_reachable_configuration(rbt_s=rbt_s,
                                         ee_s=ee_s,
                                         component_name="arm",
                                         tgt_pos=pos,
                                         cone_axis=-rotmat[:3, 2],
                                         rotation_interval=np.radians(15),
                                         obstacle_list=[frame_bottom],
                                         toggle_debug=True)
    base.run()
