from wrs import wd, rm, mgm

base = wd.World(cam_pos=rm.np.array([1, 1, 1]), lookat_pos=rm.np.array([0, 0, 0]), toggle_debug=True)
frame_o = mgm.gen_frame(ax_length=.2)
frame_o.attach_to(base)
# rotmat = rm.rotmat_from_axangle([1,1,1],rm.pi/4)
rotmat_a = rm.rotmat_from_euler(rm.pi / 3, -rm.pi / 6, rm.pi / 3)
# frame_a = mgm.gen_mycframe(ax_length=.2, rotmat=rotmat)
frame_a = mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_a)
frame_a.attach_to(base)

# point in a
pos_a = rm.np.array([.15, .07, .05])

# pos_start = rotmat_a.dot(pos_a)
# pos_end = rotmat_a.dot(rm.np.array([pos_a[0], pos_a[1], 0]))
# # mgm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgb=rm.np.zeros(3), alpha=.3], len_solid=.005, len_interval=.005).attach_to(base)
# mgm.gen_stick(pos_start, pos_end, major_radius=.001, rgb=rm.np.zeros(3), alpha=.3]).attach_to(base)
# pos_start = rotmat_a.dot(rm.np.array([pos_a[0], pos_a[1], 0]))
# pos_end = rotmat_a.dot(rm.np.array([pos_a[0], 0, 0]))
# mgm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgb=rm.np.zeros(3), alpha=.3], len_solid=.005, len_interval=.005).attach_to(base)
#
# pos_start = rotmat_a.dot(pos_a)
# pos_end = rotmat_a.dot(rm.np.array([pos_a[0], 0, pos_a[2]]))
# # mgm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgb=rm.np.zeros(3), alpha=.3], len_solid=.005, len_interval=.005).attach_to(base)
# mgm.gen_stick(pos_start, pos_end, major_radius=.001, rgb=rm.np.zeros(3), alpha=.3]).attach_to(base)
# pos_start = rotmat_a.dot(rm.np.array([pos_a[0], 0, pos_a[2]]))
# pos_end = rotmat_a.dot(rm.np.array([pos_a[0], 0, 0]))
# mgm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgb=rm.np.zeros(3), alpha=.3], len_solid=.005, len_interval=.005).attach_to(base)
#
# # pos_start = rotmat_a.dot(pos_a)
# # pos_end = rotmat_a.dot(rm.np.array([pos_a[0], pos_a[1], 0]))
# # mgm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgb=rm.np.zeros(3), alpha=.3], len_solid=.005, len_interval=.005).attach_to(base)
# pos_start = rotmat_a.dot(rm.np.array([pos_a[0], pos_a[1], 0]))
# pos_end = rotmat_a.dot(rm.np.array([0, pos_a[1], 0]))
# mgm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgb=rm.np.zeros(3), alpha=.3], len_solid=.005, len_interval=.005).attach_to(base)
#
# pos_start = rotmat_a.dot(pos_a)
# pos_end = rotmat_a.dot(rm.np.array([0, pos_a[1], pos_a[2]]))
# # mgm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgb=rm.np.zeros(3), alpha=.3], len_solid=.005, len_interval=.005).attach_to(base)
# mgm.gen_stick(pos_start, pos_end, major_radius=.001, rgb=rm.np.zeros(3), alpha=.3]).attach_to(base)
# pos_start = rotmat_a.dot(rm.np.array([0, pos_a[1], pos_a[2]]))
# pos_end = rotmat_a.dot(rm.np.array([0, pos_a[1], 0]))
# mgm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgb=rm.np.zeros(3), alpha=.3], len_solid=.005, len_interval=.005).attach_to(base)
#
# # pos_start = rotmat_a.dot(pos_a)
# # pos_end = rotmat_a.dot(rm.np.array([pos_a[0], 0, pos_a[2]]))
# # mgm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgb=rm.np.zeros(3), alpha=.3], len_solid=.005, len_interval=.005).attach_to(base)
# pos_start = rotmat_a.dot(rm.np.array([pos_a[0], 0, pos_a[2]]))
# pos_end = rotmat_a.dot(rm.np.array([0, 0, pos_a[2]]))
# mgm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgb=rm.np.zeros(3), alpha=.3], len_solid=.005, len_interval=.005).attach_to(base)
# # pos_start = rotmat_a.dot(pos_a)
# # pos_end = rotmat_a.dot(rm.np.array([0, pos_a[1], pos_a[2]]))
# # mgm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgb=rm.np.zeros(3), alpha=.3], len_solid=.005, len_interval=.005).attach_to(base)
# pos_start = rotmat_a.dot(rm.np.array([0, pos_a[1], pos_a[2]]))
# pos_end = rotmat_a.dot(rm.np.array([0, 0, pos_a[2]]))
# mgm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgb=[0,0,0,.3], len_solid=.005, len_interval=.005).attach_to(base)


# cvt to sigma o
pos_o = rotmat_a.dot(pos_a)

# mgm.gen_dashstick(pos_o, rm.np.array([pos_o[0], pos_o[1], 0]), major_radius=.001, rgb=[0,0,0,.3], len_solid=.005, len_interval=.005).attach_to(base)
mgm.gen_stick(pos_o, rm.np.array([pos_o[0], pos_o[1], 0]), radius=.001, rgb=rm.np.zeros(3), alpha=.3).attach_to(base)
mgm.gen_dashed_stick(rm.np.array([pos_o[0], pos_o[1], 0]), rm.np.array([pos_o[0], 0, 0]), radius=.001,
                    rgb=rm.np.zeros(3), alpha=.3,
                    len_solid=.005, len_interval=.005).attach_to(base)

# mgm.gen_dashstick(pos_o, rm.np.array([pos_o[0], 0, pos_o[2]]), major_radius=.001, rgb=[0,0,0,.3], len_solid=.005, len_interval=.005).attach_to(base)
mgm.gen_stick(pos_o, rm.np.array([pos_o[0], 0, pos_o[2]]), radius=.001, rgb=rm.np.zeros(3), alpha=.3).attach_to(base)
mgm.gen_dashed_stick(rm.np.array([pos_o[0], 0, pos_o[2]]), rm.np.array([pos_o[0], 0, 0]), radius=.001,
                    rgb=rm.np.zeros(3), alpha=.3,
                    len_solid=.005, len_interval=.005).attach_to(base)

# mgm.gen_dashstick(pos_o, rm.np.array([pos_o[0], pos_o[1], 0]), major_radius=.001, rgb=[0,0,0,.3], len_solid=.005, len_interval=.005).attach_to(base)
mgm.gen_dashed_stick(rm.np.array([pos_o[0], pos_o[1], 0]), rm.np.array([0, pos_o[1], 0]), radius=.001,
                    rgb=rm.np.zeros(3), alpha=.3,
                    len_solid=.005, len_interval=.005).attach_to(base)
# mgm.gen_dashstick(pos_o, rm.np.array([0, pos_o[1], pos_o[2]]), major_radius=.001, rgb=[0,0,0,.3], len_solid=.005, len_interval=.005).attach_to(base)
mgm.gen_stick(pos_o, rm.np.array([0, pos_o[1], pos_o[2]]), radius=.001, rgb=rm.np.zeros(3), alpha=.3).attach_to(base)
mgm.gen_dashed_stick(rm.np.array([0, pos_o[1], pos_o[2]]), rm.np.array([0, pos_o[1], 0]), radius=.001,
                    rgb=rm.np.zeros(3), alpha=.3,
                    len_solid=.005, len_interval=.005).attach_to(base)

# mgm.gen_dashstick(pos_o, rm.np.array([pos_o[0], 0, pos_o[2]]), major_radius=.001, rgb=[0,0,0,.3], len_solid=.005, len_interval=.005).attach_to(base)
mgm.gen_dashed_stick(rm.np.array([pos_o[0], 0, pos_o[2]]), rm.np.array([0, 0, pos_o[2]]), radius=.001,
                    rgb=rm.np.zeros(3), alpha=.3,
                    len_solid=.005, len_interval=.005).attach_to(base)
# mgm.gen_dashstick(pos_o, rm.np.array([0, pos_o[1], pos_o[2]]), major_radius=.001, rgb=[0,0,0,.3], len_solid=.005, len_interval=.005).attach_to(base)
mgm.gen_dashed_stick(rm.np.array([0, pos_o[1], pos_o[2]]), rm.np.array([0, 0, pos_o[2]]), radius=.001,
                    rgb=rm.np.zeros(3), alpha=.3,
                    len_solid=.005, len_interval=.005).attach_to(base)
# #
mgm.gen_sphere(pos=pos_o, radius=.005, rgb=[0, 0, 0, 1]).attach_to(base)
# mgm.gen_dashstick(rm.np.zeros(3), pos_o, major_radius=.003, rgb=[.3, .3, .3, 1], len_solid=.01, len_interval=.01).attach_to(base)
mgm.gen_stick(rm.np.zeros(3), pos_o, radius=.003, rgb=[.3, .3, .3, 1]).attach_to(base)

base.run()
