import visualization.panda.world as wd
import modeling.geometric_model as mgm
import basis.robot_math as rm
import math
import numpy as np


def draw_rot_arrows(rotmat, axis, column_id, rgb, portion):
    mgm.gen_circarrow(axis=axis,
                      starting_vector=rotmat[:, column_id],
                      portion=portion,
                      center=np.array([0, 0, 0]),
                      major_radius=.2,
                      minor_radius=.0015,
                      rgb=rgb,
                      alpha=1,
                      n_sec_major=64).attach_to(base)
    center = rotmat[:, column_id].dot(axis) * axis * .2
    radius = np.linalg.norm(rotmat[:, column_id] * .2 - center)
    mgm.gen_dashed_torus(axis=axis,
                         center=center,
                         major_radius=radius,
                         minor_radius=.0015,
                         rgb=rgb,
                         alpha=.06,
                         n_sec_major=64,
                         portion=1).attach_to(base)


option = 'rz'
base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0], toggle_debug=True)
rotmat = rm.rotmat_from_euler(math.pi / 3, -math.pi / 6, math.pi / 3)
alpha, beta, gamma = rm.rotmat_to_euler(rotmat, 'rzxz')

if option == 'rz':
    mgm.gen_frame(ax_length=.2).attach_to(base)
    rotmat_oa = rm.rotmat_from_euler(alpha, 0, 0, 'rzxz')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_oa, len_solid=.06, len_interval=.01).attach_to(base)
    mgm.gen_circarrow(axis=np.array([0, 0, 1]),
                      portion=.9,
                      center=np.array([0, 0, 1]) * .1,
                      major_radius=.03,
                      minor_radius=.0015,
                      rgb=np.array([.3, .3, .3]),
                      alpha=1).attach_to(base)
    draw_rot_arrows(rotmat=np.eye(3), axis=np.array([0, 0, 1]) * np.sign(alpha), column_id=0, rgb=rm.bc.green,
                    portion=abs(alpha / (2 * np.pi)))
    draw_rot_arrows(rotmat=np.eye(3), axis=np.array([0, 0, 1]) * np.sign(alpha), column_id=1, rgb=rm.bc.blue,
                    portion=abs(alpha / (2 * np.pi)))
    base.run()
if option == 'rzx':
    rotmat_oa = rm.rotmat_from_euler(alpha, 0, 0, 'rzxz')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_oa, len_solid=.06, len_interval=.01).attach_to(base)
    rotmat_ab = rm.rotmat_from_euler(alpha, beta, 0, 'rzxz')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_ab, len_solid=.025, len_interval=.01).attach_to(base)
    mgm.gen_circarrow(axis=rotmat_oa[:3, 0],
                      portion=.9,
                      center=rotmat_oa[:3, 0] * .1,
                      major_radius=.03,
                      minor_radius=.0015,
                      rgb=np.array([.3, .3, .3]),
                      alpha=1).attach_to(base)
    draw_rot_arrows(rotmat=rotmat_oa, axis=rotmat_oa[:3, 0] * np.sign(beta), column_id=1, rgb=rm.bc.green,
                    portion=abs(beta / (2 * np.pi)))
    draw_rot_arrows(rotmat=rotmat_oa, axis=rotmat_oa[:3, 0] * np.sign(beta), column_id=2, rgb=rm.bc.blue,
                    portion=abs(beta / (2 * np.pi)))
    base.run()
    base.run()
if option == 'rzxz':
    rotmat_ab = rm.rotmat_from_euler(alpha, beta, 0, 'rzxz')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_ab, len_solid=.025, len_interval=.01).attach_to(base)
    rotmat_bc = rm.rotmat_from_euler(alpha, beta, gamma, 'rzxz')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_bc).attach_to(base)
    mgm.gen_circarrow(axis=rotmat_ab[:3, 2],
                      portion=.9,
                      center=rotmat_ab[:3, 2] * .1,
                      major_radius=.03,
                      minor_radius=.0015,
                      rgb=np.array([.3, .3, .3]),
                      alpha=1).attach_to(base)
    draw_rot_arrows(rotmat=rotmat_ab, axis=rotmat_ab[:3, 2] * np.sign(gamma), column_id=0, rgb=rm.bc.red,
                    portion=abs(gamma / (2 * np.pi)))
    draw_rot_arrows(rotmat=rotmat_ab, axis=rotmat_ab[:3, 2] * np.sign(gamma), column_id=1, rgb=rm.bc.green,
                    portion=abs(gamma / (2 * np.pi)))
    base.run()
if option == 'final':
    mgm.gen_frame(ax_length=.2).attach_to(base)
    rotmat_oa = rm.rotmat_from_euler(alpha, 0, 0, 'rzxz')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_oa, len_solid=.06, len_interval=.01, alpha=.1).attach_to(base)
    rotmat_ab = rm.rotmat_from_euler(alpha, beta, 0, 'rzxz')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_ab, len_solid=.025, len_interval=.01, alpha=.1).attach_to(base)
    rotmat_bc = rm.rotmat_from_euler(alpha, beta, gamma, 'rzxz')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_bc).attach_to(base)
    print(np.degrees([alpha, beta, gamma]))
    base.run()
