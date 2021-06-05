import robot_sim.robots.tbm_changer.tbm_changer as rbt
import modeling.geometric_model as gm
import visualization.panda.world as wd
import numpy as np
import basis.robot_math as rm
import math
import pickle

base = wd.World(cam_pos=[-7, 0, 7], lookat_pos=[2.5, 0, 0], auto_cam_rotate=True)
rbt_s = rbt.TBMChanger()
rbt_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
seed0 = np.zeros(6)
seed0[3] = math.pi / 2
seed1 = np.zeros(6)
seed1[3] = -math.pi / 2
# data = []
# for x in np.linspace(1, 3, 10).tolist():
#     print(x)
#     for y in np.linspace(-1.5, 1.5, 15).tolist():
#         for z in np.linspace(-1.5, 1.5, 15).tolist():
#             tgt_pos = np.array([x, y, z])
#             tgt_rotmat = np.eye(3)
#             jnt_values0 = rbt_s.ik(component_name='arm',
#                                    tgt_pos=tgt_pos,
#                                    tgt_rotmat=tgt_rotmat,
#                                    max_niter=100,
#                                    toggle_debug=False,
#                                    seed_jnt_values=seed0)
#             if jnt_values0 is not None:
#                 jnt_values = jnt_values0
#             else:
#                 jnt_values1 = rbt_s.ik(component_name='arm',
#                                        tgt_pos=tgt_pos,
#                                        tgt_rotmat=tgt_rotmat,
#                                        max_niter=100,
#                                        toggle_debug=False,
#                                        seed_jnt_values=seed1)
#                 if jnt_values1 is not None:
#                     jnt_values = jnt_values1
#                 else:
#                     jnt_values = None
#             if jnt_values is not None:
#                 # gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat, thickness=.02).attach_to(base)
#                 rbt_s.fk(jnt_values=jnt_values)
#                 # rbt_s.gen_meshmodel().attach_to(base)
#                 data.append([tgt_pos, tgt_rotmat, rbt_s.manipulability()])
#             else:
#                 # gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat, thickness=.02).attach_to(base)
#                 data.append([tgt_pos, tgt_rotmat, 0])
# pickle.dump(data, open('manipulability.pickle', 'wb'))
# base.run()

# data = pickle.load(open('manipulability.pickle', 'rb'))
# max_manipulability = 0.0
# for item in data:
#     tgt_pos, tgt_rotmat, manipulability = item
#     if max_manipulability < manipulability:
#         max_manipulability = manipulability
# for item in data:
#     tgt_pos, tgt_rotmat, manipulability = item
#     if manipulability > 0:
#         gm.gen_sphere(pos=tgt_pos, radius=.07, rgba=[1-manipulability / max_manipulability, 0, 0, .87]).attach_to(base)
# base.run()

data = pickle.load(open('manipulability.pickle', 'rb'))
max_manipulability = 0.0
for item in data:
    tgt_pos, tgt_rotmat, manipulability = item
    if max_manipulability < manipulability:
        max_manipulability = manipulability
for item in data:
    tgt_pos, tgt_rotmat, manipulability = item
    if manipulability > 0:
        jnt_values0 = rbt_s.ik(component_name='arm',
                               tgt_pos=tgt_pos,
                               tgt_rotmat=tgt_rotmat,
                               max_niter=100,
                               toggle_debug=False,
                               seed_jnt_values=seed0)
        if jnt_values0 is not None:
            jnt_values = jnt_values0
        else:
            jnt_values1 = rbt_s.ik(component_name='arm',
                                   tgt_pos=tgt_pos,
                                   tgt_rotmat=tgt_rotmat,
                                   max_niter=100,
                                   toggle_debug=False,
                                   seed_jnt_values=seed1)
            if jnt_values1 is not None:
                jnt_values = jnt_values1
            else:
                jnt_values = None
        if jnt_values is not None:
            rbt_s.fk(jnt_values=jnt_values)
            rbt_s.gen_meshmodel().attach_to(base)
base.run()
