import visualization.panda.world as wd
import modeling.geometric_model as gm
import basis.robot_math as rm
import math
import numpy as np

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0], toggle_debug=True)
rotmat = rm.rotmat_from_euler(math.pi/3, -math.pi/6 , math.pi/3)
alpha, beta, gamma = rm._euler_from_matrix(rotmat, 'sxyx')

frame_o = gm.gen_frame(length=.2)
frame_o.attach_to(base)
# gm.gen_circarrow(axis=-[1,0,0],
#                  portion = .9,
#                  center = np.array([.1,0,0]),
#                  radius=.03,
#                  thickness=.003,
#                  rgba=[.3,.3,.3,1]).attach_to(base)
rotmat = rm.rotmat_from_euler(alpha, 0, 0, 'sxyx')
frame_a = gm.gen_dashframe(length=.2, rotmat=rotmat, lsolid=.06, lspace=.01)
frame_a.attach_to(base)
# gm.gen_circarrow(axis=[0,1,0],
#                  portion = .9,
#                  center = [0,.1,0],
#                  radius=.03,
#                  thickness=.003,
#                  rgba=[.3,.3,.3,1]).attach_to(base)
rotmat = rm.rotmat_from_euler(alpha, beta, 0, 'sxyx')
frame_a = gm.gen_dashframe(length=.2, rotmat=rotmat, lsolid=.025, lspace=.01)
frame_a.attach_to(base)
rotmat = rm.rotmat_from_euler(alpha, beta, gamma, 'sxyx')
frame_a = gm.gen_dashframe(length=.2, rotmat=rotmat)
frame_a.attach_to(base)
gm.gen_circarrow(axis=[1,0,0],
                 portion = .9,
                 center = [.1,0,0],
                 radius=.03,
                 thickness=.003,
                 rgba=[.3,.3,.3,1]).attach_to(base)

# print(rotmat)
base.run()