from wrs import wd, rm, mgm

def draw_rot_arrows(rotmat, axis, column_id, rgb, portion):
    mgm.gen_circarrow(axis=axis,
                      starting_vector=rotmat[:, column_id],
                      portion=portion,
                      center=rm.np.array([0, 0, 0]),
                      major_radius=.2,
                      minor_radius=.0015,
                      rgb=rgb,
                      alpha=1,
                      n_sec_major=64).attach_to(base)
    center = rotmat[:, column_id].dot(axis) * axis * .2
    radius = rm.np.linalg.norm(rotmat[:, column_id] * .2 - center)
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
rotmat = rm.rotmat_from_euler(rm.pi / 3, -rm.pi / 6, rm.pi / 3)
alpha, beta, gamma = rm.rotmat_to_euler(rotmat, 'rzxz')

if option == 'rz':
    mgm.gen_frame(ax_length=.2).attach_to(base)
    rotmat_oa = rm.rotmat_from_euler(alpha, 0, 0, 'rzxz')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_oa, len_solid=.06, len_interval=.01).attach_to(base)
    mgm.gen_circarrow(axis=rm.const.z_ax,
                      portion=.9,
                      center=rm.const.z_ax * .1,
                      major_radius=.03,
                      minor_radius=.0015,
                      rgb=rm.np.array([.3, .3, .3]),
                      alpha=1).attach_to(base)
    draw_rot_arrows(rotmat=rm.eye(3), axis=rm.const.z_ax * rm.sign(alpha), column_id=0, rgb=rm.const.green,
                    portion=abs(alpha / (2 * rm.pi)))
    draw_rot_arrows(rotmat=rm.eye(3), axis=rm.const.z_ax * rm.sign(alpha), column_id=1, rgb=rm.const.blue,
                    portion=abs(alpha / (2 * rm.pi)))
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
                      rgb=rm.np.array([.3, .3, .3]),
                      alpha=1).attach_to(base)
    draw_rot_arrows(rotmat=rotmat_oa, axis=rotmat_oa[:3, 0] * rm.sign(beta), column_id=1, rgb=rm.const.green,
                    portion=abs(beta / (2 * rm.pi)))
    draw_rot_arrows(rotmat=rotmat_oa, axis=rotmat_oa[:3, 0] * rm.sign(beta), column_id=2, rgb=rm.const.blue,
                    portion=abs(beta / (2 * rm.pi)))
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
                      rgb=rm.np.array([.3, .3, .3]),
                      alpha=1).attach_to(base)
    draw_rot_arrows(rotmat=rotmat_ab, axis=rotmat_ab[:3, 2] * rm.sign(gamma), column_id=0, rgb=rm.const.red,
                    portion=abs(gamma / (2 * rm.pi)))
    draw_rot_arrows(rotmat=rotmat_ab, axis=rotmat_ab[:3, 2] * rm.sign(gamma), column_id=1, rgb=rm.const.green,
                    portion=abs(gamma / (2 * rm.pi)))
    base.run()
if option == 'final':
    mgm.gen_frame(ax_length=.2).attach_to(base)
    rotmat_oa = rm.rotmat_from_euler(alpha, 0, 0, 'rzxz')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_oa, len_solid=.06, len_interval=.01, alpha=.1).attach_to(base)
    rotmat_ab = rm.rotmat_from_euler(alpha, beta, 0, 'rzxz')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_ab, len_solid=.025, len_interval=.01, alpha=.1).attach_to(base)
    rotmat_bc = rm.rotmat_from_euler(alpha, beta, gamma, 'rzxz')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_bc).attach_to(base)
    print(rm.np.degrees([alpha, beta, gamma]))
    base.run()
