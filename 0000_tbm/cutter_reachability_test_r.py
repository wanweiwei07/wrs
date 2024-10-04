import wrs.visualization.panda.world as wd
import numpy as np
from wrs import basis as rm, robot_sim as rbt, robot_sim as mcn, modeling as gm
import math
import pickle
import opt_ik

base = wd.World(cam_pos=[-15, 2.624 - 0.275, 15], lookat_pos=[-1.726 - 0.35, 2.624 - 0.275, 5.323],
                auto_cam_rotate=False)
mcn_s = mcn.TBM()
# mcn_s.fk(np.array([-math.pi/24]))
# mcn_s.gen_meshmodel().attach_to(base)
rbt_s = rbt.TBMChangerR(pos=np.array([-1.726 - 0.35, 2.624 - 0.275, 5.323]))
# rbt_s.gen_meshmodel(toggle_flange_frame=True).attach_to(base)
ik_s = opt_ik.OptIK(rbt_s, component_name='arm', obstacle_list=[])
# base.run()
for step in [1, 2, 3, 4]:
    print(step)
    mcn_s.fk(np.array([math.pi / 18 * step]))
    for cutter in mcn_s.cutters['1.5'][4:]:
        tgt_pos = cutter.pos
        tgt_rotmat = rm.rotmat_from_euler(math.pi / 2, 0, 0).dot(cutter.rotmat)
        gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat, axis_radius=.05, axis_length=1).attach_to(base)
        seed0 = np.zeros(6)
        seed0[2] = math.pi / 2
        seed0[3] = math.pi / 2
        seed0[4] = -math.pi / 2
        seed1 = np.zeros(6)
        seed1[2] = -math.pi / 2
        seed1[3] = -math.pi / 2
        seed1[4] = -math.pi / 2
        for tmb_x in [0, -.2]:
            rbt_s.fix_to(pos=np.array([-1.726-0.35+tmb_x, 2.624-0.275, 5.323]), rotmat=np.eye(3))
            try:
                jnt_values, _ = ik_s.solve(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values= rbt_s.arm.home_conf)
            except:
                continue
            # try:
            #     jnt_values, _ = ik_s.solve(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values= seed0)
            # except:
            #     jnt_values = None
            # if jnt_values is None:
            #     try:
            #         jnt_values, _ = ik_s.solve(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values= seed1)
            #     except:
            #         jnt_values = None
            # jnt_values = rbt_s.ik(component_name='arm',
            #                       tgt_pos=tgt_pos,
            #                       tgt_rotmat=tgt_rotmat,
            #                       max_n_iter=500,
            #                       toggle_dbg=False,
            #                       seed_jnt_values=seed0)
            # if jnt_values is None:
            #     jnt_values = rbt_s.ik(component_name='arm',
            #                           tgt_pos=tgt_pos,
            #                           tgt_rotmat=tgt_rotmat,
            #                           max_n_iter=500,
            #                           toggle_dbg=False,
            #                           seed_jnt_values=seed1)
            if jnt_values is None:
                continue
            else:
                rbt_s.fk(component_name="arm", jnt_values=jnt_values)
                rbt_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
                mcn_s.gen_meshmodel().attach_to(base)
                break
    # rbt_s.fix_to(pos=np.array([-1.726 - 0.35, 2.624 - 0.275, 5.323]), rotmat=np.eye(3))
base.run()

# solvable = []
# for angle in np.linspace(0,math.pi*2, 24):
#     print(angle)
#     mcn_s.fk(np.array([angle]))
#     for k in mcn_s.cutters.keys():
#         print(k)
#         for i, cutter in enumerate(mcn_s.cutters[k]):
#             tgt_pos = cutter.pos
#             tgt_rotmat = rm.rotmat_from_euler(math.pi/2, 0, 0).dot(cutter.rotmat)
#             mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat, major_radius=.05, axis_length=1).attach_to(base)
#             seed0 = np.zeros(6)
#             seed0[2] = math.pi / 2
#             seed0[3] = math.pi / 2
#             seed0[4] = -math.pi / 2
#             seed1 = np.zeros(6)
#             seed1[2] = -math.pi / 2
#             seed1[3] = -math.pi / 2
#             seed1[4] = -math.pi / 2
#             jnt_values = rbt_s.ik(component_name='arm',
#                                   tgt_pos=tgt_pos,
#                                   tgt_rotmat=tgt_rotmat,
#                                   max_n_iter=1000,
#                                   toggle_dbg=False,
#                                   seed_jnt_values=seed0)
#             if jnt_values is None:
#                 jnt_values = rbt_s.ik(component_name='arm',
#                                       tgt_pos=tgt_pos,
#                                       tgt_rotmat=tgt_rotmat,
#                                       max_n_iter=1000,
#                                       toggle_dbg=False,
#                                       seed_jnt_values=seed1)
#             if jnt_values is None:
#                 continue
#             # rbt_s.fk(component_name="arm", jnt_values=jnt_values)
#             # rbt_s.gen_meshmodel(toggle_flange_frame=True).attach_to(base)
#             # mcn_s.gen_meshmodel().attach_to(base)
#             solvable.append([mcn_s.cutter_pos_dict[k][i], mcn_s.cutter_rotmat_dict[k][i], jnt_values])
# pickle.dump(solvable, open('manipulability.pickle', 'wb'))
# base.run()

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
#                                    max_n_iter=100,
#                                    toggle_dbg=False,
#                                    seed_jnt_values=seed0)
#             if jnt_values0 is not None:
#                 jnt_values = jnt_values0
#             else:
#                 jnt_values1 = rbt_s.ik(component_name='arm',
#                                        tgt_pos=tgt_pos,
#                                        tgt_rotmat=tgt_rotmat,
#                                        max_n_iter=100,
#                                        toggle_dbg=False,
#                                        seed_jnt_values=seed1)
#                 if jnt_values1 is not None:
#                     jnt_values = jnt_values1
#                 else:
#                     jnt_values = None
#             if jnt_values is not None:
#                 # mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat, major_radius=.02).attach_to(base)
#                 rbt_s.fk(jnt_values=jnt_values)
#                 # rbt_s.gen_meshmodel().attach_to(base)
#                 data.append([tgt_pos, tgt_rotmat, rbt_s.manipulability()])
#             else:
#                 # mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat, major_radius=.02).attach_to(base)
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
#         mgm.gen_sphere(pos=tgt_pos, major_radius=.07, rgba=[1-manipulability / max_manipulability, 0, 0, .87]).attach_to(base)
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
