import wrs.visualization.panda.world as wd
from wrs import basis as rm, modeling as mgm
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


option = 'x'
base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0], toggle_debug=True)
if option == 'x':
    mgm.gen_frame(ax_length=.2).attach_to(base)
    rotmat_oa = rm.rotmat_from_euler(np.pi / 3, 0, 0)
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_oa, len_solid=.06, len_interval=.01).attach_to(base)
    mgm.gen_circarrow(axis=np.array([1, 0, 0]),
                      portion=.9,
                      center=np.array([.1, 0, 0]),
                      major_radius=.03,
                      minor_radius=.0015,
                      rgb=np.array([.3, .3, .3]),
                      alpha=1).attach_to(base)
    draw_rot_arrows(rotmat=np.eye(3), axis=np.array([1, 0, 0]), column_id=1, rgb=rm.bc.green, portion=1 / 6)
    draw_rot_arrows(rotmat=np.eye(3), axis=np.array([1, 0, 0]), column_id=2, rgb=rm.bc.blue, portion=1 / 6)
    base.run()
if option == 'xy':
    mgm.gen_frame(ax_length=.2, alpha=np.array([.1, .1, .1])).attach_to(base)
    rotmat_oa = rm.rotmat_from_euler(np.pi / 3, 0, 0)
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_oa, len_solid=.06, len_interval=.01).attach_to(base)
    rotmat_ab = rm.rotmat_from_euler(np.pi / 3, -np.pi / 6, 0)
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_ab, len_solid=.025, len_interval=.01).attach_to(base)
    mgm.gen_circarrow(axis=np.array([0, -1, 0]),
                      portion=.9,
                      center=np.array([0, .1, 0]),
                      major_radius=.03,
                      minor_radius=.0015,
                      rgb=np.array([.3, .3, .3]),
                      alpha=1).attach_to(base)
    draw_rot_arrows(rotmat=rotmat_oa, axis=np.array([0, -1, 0]), column_id=0, rgb=rm.bc.red, portion=1 / 12)
    draw_rot_arrows(rotmat=rotmat_oa, axis=np.array([0, -1, 0]), column_id=1, rgb=rm.bc.green, portion=1 / 12)
    draw_rot_arrows(rotmat=rotmat_oa, axis=np.array([0, -1, 0]), column_id=2, rgb=rm.bc.blue, portion=1 / 12)
    base.run()
if option == 'xyz':
    mgm.gen_frame(ax_length=.2, alpha=np.array([.1, .1, .1])).attach_to(base)
    rotmat_ab = rm.rotmat_from_euler(np.pi / 3, -np.pi / 6, 0)
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_ab, len_solid=.025, len_interval=.01).attach_to(base)
    rotmat_bc = rm.rotmat_from_euler(np.pi / 3, -np.pi / 6, np.pi / 3)
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_bc).attach_to(base)
    mgm.gen_circarrow(axis=np.array([0, 0, 1]),
                      portion=.9,
                      center=np.array([0, 0, .1]),
                      major_radius=.03,
                      minor_radius=.0015,
                      rgb=np.array([.3, .3, .3]),
                      alpha=1).attach_to(base)
    draw_rot_arrows(rotmat=rotmat_ab, axis=np.array([0, 0, 1]), column_id=0, rgb=rm.bc.red, portion=1 / 6)
    draw_rot_arrows(rotmat=rotmat_ab, axis=np.array([0, 0, 1]), column_id=1, rgb=rm.bc.green, portion=1 / 6)
    draw_rot_arrows(rotmat=rotmat_ab, axis=np.array([0, 0, 1]), column_id=2, rgb=rm.bc.blue, portion=1 / 6)
    base.run()
if option == 'final':
    frame_o = mgm.gen_frame(ax_length=.2, alpha=np.array([1, 1, 1]))
    frame_o.attach_to(base)
    rotmat = rm.rotmat_from_euler(np.pi / 3, 0, 0)
    frame_a = mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat, len_solid=.06, len_interval=.01, alpha=.1)
    frame_a.attach_to(base)
    rotmat = rm.rotmat_from_euler(np.pi / 3, -np.pi / 6, 0)
    frame_a = mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat, len_solid=.06, len_interval=.01, alpha=.1)
    frame_a.attach_to(base)
    rotmat = rm.rotmat_from_euler(np.pi / 3, -np.pi / 6, np.pi / 3)
    frame_a = mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat)
    frame_a.attach_to(base)
    base.run()
