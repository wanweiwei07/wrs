import wrs.visualization.panda.world as wd
from wrs import basis as rm, modeling as cm
import numpy as np
import utils
import os

base = wd.World(cam_pos=[0,0,.3], lookat_pos=[0, 0, 0])

this_dir, this_filename = os.path.split(__file__)
file_tip_rack = os.path.join(this_dir, "objects", "tip_rack.stl")
tip_rack = utils.Rack96(file_tip_rack)
tip_rack.set_rgba([140 / 255, 110 / 255, 170 / 255, 1])
tip_rack.set_pose(pos=np.array([.0, 0.0, .0]), rotmat=rm.rotmat_from_axangle([0, 0, 1], np.pi / 2))
tip_rack.attach_to(base)

file_tip = os.path.join(this_dir, "objects", "tip.stl")
tip = cm.CollisionModel(file_tip)
tip.set_rgba([200 / 255, 180 / 255, 140 / 255, 1])
tip_cm_list = []
for id_x in range(8):
    for id_y in range(12):
        pos, rotmat = tip_rack.get_rack_hole_pose(id_x=id_x, id_y=id_y)
        tip_new = tip.copy()
        tip_new.set_pose(pos, rotmat)
        # mgm.gen_frame(pos=pos, rotmat=rotmat).attach_to(base)
        tip_new.attach_to(base)
        tip_cm_list.append(tip_new)

base.run()