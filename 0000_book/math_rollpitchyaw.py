import visualization.panda.world as wd
import modeling.geometric_model as gm
import basis.robot_math as rm
import math
import numpy as np

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0], toggle_debug=True)
frame_o = gm.gen_frame(axis_length=.2)
frame_o.attach_to(base)
rotmat = rm.rotmat_from_euler(math.pi/3, 0, 0)
frame_a = gm.gen_dashed_frame(axis_length=.2, rotmat=rotmat, len_solid=.06, len_interval=.01)
frame_a.attach_to(base)
# mgm.gen_circarrow(axis=np.array([1,0,0]),
#                  portion = .9,
#                  center = np.array([.1,0,0]),
#                  major_radius=.03,
#                  major_radius=.003,
#                  rgba=[.3,.3,.3,1]).attach_to(base)
rotmat = rm.rotmat_from_euler(math.pi/3, -math.pi/6, 0)
frame_a = gm.gen_dashed_frame(axis_length=.2, rotmat=rotmat, len_solid=.025, len_interval=.01)
frame_a.attach_to(base)
# mgm.gen_circarrow(axis=np.array([0,-1,0]),
#                  portion = .9,
#                  center = np.array([0,.1,0]),
#                  major_radius=.03,
#                  major_radius=.003,
#                  rgba=[.3,.3,.3,1]).attach_to(base)
rotmat = rm.rotmat_from_euler(math.pi/3, -math.pi/6 , math.pi/3)
frame_a = gm.gen_dashed_frame(axis_length=.2, rotmat=rotmat)
frame_a.attach_to(base)
gm.gen_circarrow(axis=np.array([0,0,1]),
                 portion = .9,
                 center = np.array([0,0,.1]),
                 radius=.03,
                 thickness=.003,
                 rgba=[.3,.3,.3,1]).attach_to(base)

print(rotmat)
base.run()