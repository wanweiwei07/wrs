from wrs import wd, rm, mgm

alpha = 1
base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0], toggle_debug=True)
# sigma_o
rotmat_o = rm.np.eye(3)
mgm.gen_frame(rotmat=rotmat_o, ax_length=.2, alpha=alpha).attach_to(base)
# simga_a
rotmat_a = rm.rotmat_from_euler(rm.pi / 3, -rm.pi / 6, rm.pi / 3)
mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_a, alpha=alpha).attach_to(base)
# sigma_o_torus
mgm.gen_torus(axis=rotmat_o[:, 2], portion=1, major_radius=.2, minor_radius=.0015, rgb=rm.np.array([1, 1, 0]), alpha=alpha,
              n_sec_major=64).attach_to(base)
# sigma_a_torus
mgm.gen_dashed_torus(axis=rotmat_a[:, 2], portion=1, major_radius=.2, minor_radius=.0015, rgb=rm.np.array([1, 1, 0]),
                     alpha=alpha, len_interval=.007, len_solid=.01, n_sec_major=64).attach_to(base)
# sphere
# mgm.gen_sphere(radius=.2, rgb=rm.np.array([.67, .67, .67]), alpha=.8, ico_level=7).attach_to(base)
# # c_0, c_1
# cross_vec = rm.unit_vector(rm.np.cross(rm.np.array([0, 0, 1]), rotmat[:3, 2]))
# mgm.gen_arrow(epos=cross_vec * .2, rgb=rm.np.array([1, 1, 0]), alpha=1).attach_to(base)
# mgm.gen_sphere(radius=.005, pos=cross_vec * .2, rgb=rm.np.array([0, 0, 0]), alpha=1).attach_to(base)
# nxt_vec_uvw = rotmat.dot(cross_vec)
# mgm.gen_dashed_arrow(epos=nxt_vec_uvw * .2, rgb=rm.np.array([1, 1, 0]), alpha=1).attach_to(base)
# mgm.gen_sphere(radius=.005, pos=nxt_vec_uvw * .2, rgb=rm.np.array([0, 0, 0]), alpha=1).attach_to(base)
# # c_2, c_0
# cross_vec = rm.unit_vector(rm.np.cross(rm.np.array([0, 0, 1]), rotmat[:3, 2]))
# pre_vec_xyz = rotmat.T.dot(cross_vec)
# mgm.gen_arrow(epos=pre_vec_xyz*.2, rgb=rm.np.array([1, 1, 0]), alpha=1).attach_to(base)
# mgm.gen_sphere(radius=.005, pos=pre_vec_xyz * .2, rgb=rm.np.array([0, 0, 0]), alpha=1).attach_to(base)
# mgm.gen_dashed_arrow(epos=cross_vec * .2, rgb=rm.np.array([1, 1, 0]), alpha=1, len_solid=.01, len_interval=.007).attach_to(base)
# mgm.gen_sphere(radius=.005, pos=cross_vec * .2, rgb=rm.np.array([0, 0, 0]), alpha=1).attach_to(base)
# # c_0, c_1, c_2
# cross_vec = rm.unit_vector(rm.np.cross(rm.np.array([0, 0, 1]), rotmat[:3, 2]))
# mgm.gen_arrow(epos=cross_vec * .2, rgb=rm.np.array([1, 1, 0]), alpha=alpha).attach_to(base)
# mgm.gen_sphere(radius=.005, pos=cross_vec * .2, rgb=rm.np.array([0, 0, 0]), alpha=alpha).attach_to(base)
# nxt_vec_uvw = rotmat.dot(cross_vec)
# mgm.gen_arrow(epos=nxt_vec_uvw * .2, rgb=rm.np.array([1, 1, 0]), alpha=alpha).attach_to(base)
# mgm.gen_sphere(radius=.005, pos=nxt_vec_uvw * .2, rgb=rm.np.array([0, 0, 0]), alpha=alpha).attach_to(base)
# pre_vec_xyz = rotmat.T.dot(cross_vec)
# mgm.gen_arrow(epos=pre_vec_xyz * .2, rgb=rm.np.array([1, 1, 0]), alpha=alpha).attach_to(base)
# mgm.gen_sphere(radius=.005, pos=pre_vec_xyz * .2, rgb=rm.np.array([0, 0, 0]), alpha=alpha).attach_to(base)
#
# # torus traj
ax, angle = rm.axangle_between_rotmat(rm.np.eye(3), rotmat_a)
mgm.gen_arrow(epos=ax * .4, rgb=rm.np.array([0, 0, 0]), alpha=1).attach_to(base)
for step_angle in rm.np.linspace(0, angle, 10).tolist():
    rotmat = rm.rotmat_from_axangle(ax, step_angle)
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_a).attach_to(base)
    mgm.gen_dashed_torus(axis=rotmat[:3, 2], portion=1, major_radius=.2, minor_radius=.0015, rgb=rm.np.array([1, 1, 0]),
                     alpha=1, len_interval=.007, len_solid=.01, n_sec_major=64).attach_to(base)
mgm.gen_sphere(radius=.2, rgb=rm.np.array([.67, .67, .67]), alpha=.8, ico_level=5).attach_to(base)
# # torus through c_0,1,2
# major_radius, _ = rm.unit_vector(cross_vec * .2 - cross_vec.dot(ax) * ax * .2, toggle_length=True)
# mgm.gen_circarrow(ax, portion=1, center=cross_vec.dot(ax) * ax * .2, major_radius=major_radius, n_sec_major=64,
#                   minor_radius=.0015, rgb=rm.np.array([0, 0, 0]), alpha=1).attach_to(base)

# # all rotations
# vec = rm.np.array([1, 0, 0])
# portion =1
# # portion = angle/(2*rm.pi)
# major_radius, _ = rm.unit_vector(vec * .2 - vec.dot(ax) * ax * .2, toggle_length=True)
# mgm.gen_circarrow(ax, starting_vector=vec - vec.dot(ax) * ax, portion=portion, center=vec.dot(ax) * ax * .2,
#                   major_radius=major_radius, n_sec_major=64, minor_radius=.0015, rgb=rm.const.red,
#                   alpha=alpha).attach_to(base)
# # mgm.gen_dashed_torus(axis=ax,
# #                      center=vec.dot(ax) * ax * .2,
# #                      major_radius=major_radius,
# #                      minor_radius=.0015,
# #                      rgb=rm.const.red,
# #                      alpha=.06,
# #                      n_sec_major=64,
# #                      portion=1).attach_to(base)
# vec = rm.np.array([0, 1, 0])
# major_radius, _ = rm.unit_vector(vec * .2 - vec.dot(ax) * ax * .2, toggle_length=True)
# mgm.gen_circarrow(ax, starting_vector=vec - vec.dot(ax) * ax, portion=portion, center=vec.dot(ax) * ax * .2,
#                   major_radius=major_radius, n_sec_major=64, minor_radius=.0015, rgb=rm.const.green,
#                   alpha=alpha).attach_to(base)
# # mgm.gen_dashed_torus(axis=ax,
# #                      center=vec.dot(ax) * ax * .2,
# #                      major_radius=major_radius,
# #                      minor_radius=.0015,
# #                      rgb=rm.const.green,
# #                      alpha=.06,
# #                      n_sec_major=64,
# #                      portion=1).attach_to(base)
# vec = rm.np.array([0, 0, 1])
# major_radius, _ = rm.unit_vector(vec * .2 - vec.dot(ax) * ax * .2, toggle_length=True)
# mgm.gen_circarrow(ax, starting_vector=vec - vec.dot(ax) * ax, portion=portion, center=vec.dot(ax) * ax * .2,
#                   major_radius=major_radius, n_sec_major=64, minor_radius=.0015, rgb=rm.const.blue,
#                   alpha=alpha).attach_to(base)
# # mgm.gen_dashed_torus(axis=ax,
# #                      center=vec.dot(ax) * ax * .2,
# #                      major_radius=major_radius,
# #                      minor_radius=.0015,
# #                      rgb=rm.const.blue,
# #                      alpha=.06,
# #                      n_sec_major=64,
# #                      portion=1).attach_to(base)

base.run()
