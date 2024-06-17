import visualization.panda.world as wd
import modeling.geometric_model as mgm
import basis.robot_math as rm
import numpy as np

option = 'xyz'
base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0], toggle_debug=True)
rotmat = rm.rotmat_from_euler(np.pi / 3, -np.pi / 6, np.pi / 3)
alpha, beta, gamma = rm._euler_from_matrix(rotmat, 'sxyx')
frame_o = mgm.gen_frame(ax_length=.2, alpha=np.array([.1, .1, .1]))
frame_o.attach_to(base)

if option == 'x':
    mgm.gen_circarrow(axis=np.array([-1, 0, 0]),
                      portion=.9,
                      center=np.array([.1, 0, 0]),
                      major_radius=.03,
                      minor_radius=.0015,
                      rgba=[.3, .3, .3, 1]).attach_to(base)
    rotmat = rm.rotmat_from_euler(alpha, 0, 0, 'sxyx')
    frame_a = mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat, len_solid=.06, len_interval=.01)
    frame_a.attach_to(base)
    base.run()
if option == 'xy':
    mgm.gen_circarrow(axis=np.array([0,1,0]),
                     portion = .9,
                     center = [0,.1,0],
                     major_radius=.03,
                     minor_radius=.0015,
                     rgba=[.3,.3,.3,1]).attach_to(base)
    rotmat = rm.rotmat_from_euler(alpha, 0, 0, 'sxyx')
    frame_a = mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat, len_solid=.06, len_interval=.01)
    frame_a.attach_to(base)
    rotmat = rm.rotmat_from_euler(alpha, beta, 0, 'sxyx')
    frame_a = mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat, len_solid=.025, len_interval=.01)
    frame_a.attach_to(base)
    base.run()
if option == 'xyz':
    rotmat = rm.rotmat_from_euler(alpha, beta, 0, 'sxyx')
    frame_a = mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat, len_solid=.025, len_interval=.01)
    frame_a.attach_to(base)
    rotmat = rm.rotmat_from_euler(alpha, beta, gamma, 'sxyx')
    frame_a = mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat)
    frame_a.attach_to(base)
    mgm.gen_circarrow(axis=np.array([1, 0, 0]),
                     portion=.9,
                     center=[.1, 0, 0],
                     major_radius=.03,
                     minor_radius=.0015,
                     rgba=[.3, .3, .3, 1]).attach_to(base)
    base.run()
