from wrs import wd, rm, mgm


def draw_rot_arrows(rotmat, axis, column_id, rgb, portion):
    mgm.gen_circarrow(axis=axis,
                      starting_vector=rotmat[:, column_id],
                      portion=portion,
                      center=rm.np.zeros(3),
                      major_radius=.2,
                      minor_radius=.0015,
                      rgb=rgb,
                      alpha=1,
                      n_sec_major=64).attach_to(base)
    center = rotmat[:, column_id].dot(axis) * axis * .2
    radius = rm.norm(rotmat[:, column_id] * .2 - center)
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
rotmat = rm.rotmat_from_euler(rm.pi / 3, -rm.pi / 6, rm.pi / 3)
alpha, beta, gamma = rm.rotmat_to_euler(rotmat, 'sxyx')

if option == 'x':
    mgm.gen_frame(ax_length=.2).attach_to(base)
    rotmat_oa = rm.rotmat_from_euler(alpha, 0, 0, 'sxyx')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_oa, len_solid=.06, len_interval=.01).attach_to(base)
    mgm.gen_circarrow(axis=-rm.const.x_ax,
                      portion=.9,
                      center=rm.vec(.1, 0, 0),
                      major_radius=.03,
                      minor_radius=.0015,
                      rgb=rm.vec(.3, .3, .3),
                      alpha=1).attach_to(base)
    draw_rot_arrows(rotmat=rm.np.eye(3), axis=rm.const.x_ax * rm.sign(alpha), column_id=1, rgb=rm.const.green,
                    portion=abs(alpha / (2 * rm.pi)))
    draw_rot_arrows(rotmat=rm.np.eye(3), axis=rm.const.x_ax * rm.sign(alpha), column_id=2, rgb=rm.const.blue,
                    portion=abs(alpha / (2 * rm.pi)))
    base.run()
if option == 'xy':
    mgm.gen_frame(ax_length=.2, alpha=rm.vec(.1, .1, .1)).attach_to(base)
    mgm.gen_circarrow(axis=rm.const.y_ax,
                      portion=.9,
                      center=[0, .1, 0],
                      major_radius=.03,
                      minor_radius=.0015,
                      rgb=rm.vec(.3, .3, .3),
                      alpha=1).attach_to(base)
    rotmat_oa = rm.rotmat_from_euler(alpha, 0, 0, 'sxyx')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_oa, len_solid=.06, len_interval=.01).attach_to(base)
    rotmat_ab = rm.rotmat_from_euler(alpha, beta, 0, 'sxyx')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_ab, len_solid=.025, len_interval=.01).attach_to(base)
    draw_rot_arrows(rotmat=rotmat_oa, axis=rm.const.y_ax * rm.sign(beta), column_id=0, rgb=rm.const.red,
                    portion=abs(beta / (2 * rm.pi)))
    draw_rot_arrows(rotmat=rotmat_oa, axis=rm.const.y_ax * rm.sign(beta), column_id=1, rgb=rm.const.green,
                    portion=abs(beta / (2 * rm.pi)))
    draw_rot_arrows(rotmat=rotmat_oa, axis=rm.const.y_ax * rm.sign(beta), column_id=2, rgb=rm.const.blue,
                    portion=abs(beta / (2 * rm.pi)))
    base.run()
if option == 'xyx':
    mgm.gen_frame(ax_length=.2, alpha=rm.vec(.1, .1, .1)).attach_to(base)
    mgm.gen_circarrow(axis=rm.const.x_ax,
                      portion=.9,
                      center=[.1, 0, 0],
                      major_radius=.03,
                      minor_radius=.0015,
                      rgb=rm.vec(.3, .3, .3),
                      alpha=1).attach_to(base)
    rotmat_ab = rm.rotmat_from_euler(alpha, beta, 0, 'sxyx')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_ab, len_solid=.025, len_interval=.01).attach_to(base)
    rotmat_bc = rm.rotmat_from_euler(alpha, beta, gamma, 'sxyx')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_bc).attach_to(base)
    draw_rot_arrows(rotmat=rotmat_ab, axis=rm.const.x_ax * rm.sign(gamma), column_id=0, rgb=rm.const.red,
                    portion=abs(gamma / (2 * rm.pi)))
    draw_rot_arrows(rotmat=rotmat_ab, axis=rm.const.x_ax * rm.sign(gamma), column_id=1, rgb=rm.const.green,
                    portion=abs(gamma / (2 * rm.pi)))
    draw_rot_arrows(rotmat=rotmat_ab, axis=rm.const.x_ax * rm.sign(gamma), column_id=2, rgb=rm.const.blue,
                    portion=abs(gamma / (2 * rm.pi)))
    base.run()
if option == 'final':
    mgm.gen_frame(ax_length=.2, alpha=1).attach_to(base)
    rotmat_oa = rm.rotmat_from_euler(alpha, 0, 0, 'sxyx')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_oa, len_solid=.06, len_interval=.01, alpha=.1).attach_to(base)
    rotmat_ab = rm.rotmat_from_euler(alpha, beta, 0, 'sxyx')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_ab, len_solid=.025, len_interval=.01, alpha=.1).attach_to(base)
    rotmat_bc = rm.rotmat_from_euler(alpha, beta, gamma, 'sxyx')
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_bc).attach_to(base)
    print(rm.degrees([alpha, beta, gamma]))
    base.run()
