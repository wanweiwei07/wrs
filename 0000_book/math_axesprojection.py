import visualization.panda.world as wd
import modeling.geometric_model as gm
import basis.robot_math as rm
import math
import numpy as np

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0], toggle_debug=True)
frame_o = gm.gen_frame(axis_length=.2)
frame_o.attach_to(base)
# rotmat = rm.rotmat_from_axangle([1,1,1],math.pi/4)
rotmat_a = rm.rotmat_from_euler(math.pi / 3, -math.pi / 6, math.pi / 3)
# frame_a = gm.gen_mycframe(axis_length=.2, rotmat=rotmat)
frame_a = gm.gen_dashed_frame(axis_length=.2, rotmat=rotmat_a)
frame_a.attach_to(base)

# point in a
pos_a = np.array([.15, .07, .05])

# pos_start = rotmat_a.dot(pos_a)
# pos_end = rotmat_a.dot(np.array([pos_a[0], pos_a[1], 0]))
# # gm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)
# gm.gen_stick(pos_start, pos_end, major_radius=.001, rgba=[0, 0, 0, .3]).attach_to(base)
# pos_start = rotmat_a.dot(np.array([pos_a[0], pos_a[1], 0]))
# pos_end = rotmat_a.dot(np.array([pos_a[0], 0, 0]))
# gm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)
#
# pos_start = rotmat_a.dot(pos_a)
# pos_end = rotmat_a.dot(np.array([pos_a[0], 0, pos_a[2]]))
# # gm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)
# gm.gen_stick(pos_start, pos_end, major_radius=.001, rgba=[0, 0, 0, .3]).attach_to(base)
# pos_start = rotmat_a.dot(np.array([pos_a[0], 0, pos_a[2]]))
# pos_end = rotmat_a.dot(np.array([pos_a[0], 0, 0]))
# gm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)
#
# # pos_start = rotmat_a.dot(pos_a)
# # pos_end = rotmat_a.dot(np.array([pos_a[0], pos_a[1], 0]))
# # gm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)
# pos_start = rotmat_a.dot(np.array([pos_a[0], pos_a[1], 0]))
# pos_end = rotmat_a.dot(np.array([0, pos_a[1], 0]))
# gm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)
#
# pos_start = rotmat_a.dot(pos_a)
# pos_end = rotmat_a.dot(np.array([0, pos_a[1], pos_a[2]]))
# # gm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)
# gm.gen_stick(pos_start, pos_end, major_radius=.001, rgba=[0, 0, 0, .3]).attach_to(base)
# pos_start = rotmat_a.dot(np.array([0, pos_a[1], pos_a[2]]))
# pos_end = rotmat_a.dot(np.array([0, pos_a[1], 0]))
# gm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)
#
# # pos_start = rotmat_a.dot(pos_a)
# # pos_end = rotmat_a.dot(np.array([pos_a[0], 0, pos_a[2]]))
# # gm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)
# pos_start = rotmat_a.dot(np.array([pos_a[0], 0, pos_a[2]]))
# pos_end = rotmat_a.dot(np.array([0, 0, pos_a[2]]))
# gm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)
# # pos_start = rotmat_a.dot(pos_a)
# # pos_end = rotmat_a.dot(np.array([0, pos_a[1], pos_a[2]]))
# # gm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)
# pos_start = rotmat_a.dot(np.array([0, pos_a[1], pos_a[2]]))
# pos_end = rotmat_a.dot(np.array([0, 0, pos_a[2]]))
# gm.gen_dashstick(pos_start, pos_end, major_radius=.001, rgba=[0,0,0,.3], len_solid=.005, len_interval=.005).attach_to(base)


# cvt to sigma o
pos_o = rotmat_a.dot(pos_a)

# gm.gen_dashstick(pos_o, np.array([pos_o[0], pos_o[1], 0]), major_radius=.001, rgba=[0,0,0,.3], len_solid=.005, len_interval=.005).attach_to(base)
gm.gen_stick(pos_o, np.array([pos_o[0], pos_o[1], 0]), radius=.001, rgba=[0, 0, 0, .3]).attach_to(base)
gm.gen_dashed_stick(np.array([pos_o[0], pos_o[1], 0]), np.array([pos_o[0], 0, 0]), radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)

# gm.gen_dashstick(pos_o, np.array([pos_o[0], 0, pos_o[2]]), major_radius=.001, rgba=[0,0,0,.3], len_solid=.005, len_interval=.005).attach_to(base)
gm.gen_stick(pos_o, np.array([pos_o[0], 0, pos_o[2]]), radius=.001, rgba=[0, 0, 0, .3]).attach_to(base)
gm.gen_dashed_stick(np.array([pos_o[0], 0, pos_o[2]]), np.array([pos_o[0], 0, 0]), radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)

# gm.gen_dashstick(pos_o, np.array([pos_o[0], pos_o[1], 0]), major_radius=.001, rgba=[0,0,0,.3], len_solid=.005, len_interval=.005).attach_to(base)
gm.gen_dashed_stick(np.array([pos_o[0], pos_o[1], 0]), np.array([0, pos_o[1], 0]), radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)
# gm.gen_dashstick(pos_o, np.array([0, pos_o[1], pos_o[2]]), major_radius=.001, rgba=[0,0,0,.3], len_solid=.005, len_interval=.005).attach_to(base)
gm.gen_stick(pos_o, np.array([0, pos_o[1], pos_o[2]]), radius=.001, rgba=[0, 0, 0, .3]).attach_to(base)
gm.gen_dashed_stick(np.array([0, pos_o[1], pos_o[2]]), np.array([0, pos_o[1], 0]), radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)

# gm.gen_dashstick(pos_o, np.array([pos_o[0], 0, pos_o[2]]), major_radius=.001, rgba=[0,0,0,.3], len_solid=.005, len_interval=.005).attach_to(base)
gm.gen_dashed_stick(np.array([pos_o[0], 0, pos_o[2]]), np.array([0, 0, pos_o[2]]), radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)
# gm.gen_dashstick(pos_o, np.array([0, pos_o[1], pos_o[2]]), major_radius=.001, rgba=[0,0,0,.3], len_solid=.005, len_interval=.005).attach_to(base)
gm.gen_dashed_stick(np.array([0, pos_o[1], pos_o[2]]), np.array([0, 0, pos_o[2]]), radius=.001, rgba=[0, 0, 0, .3], len_solid=.005, len_interval=.005).attach_to(base)
# #
gm.gen_sphere(pos=pos_o, radius=.005, rgba=[0,0,0,1]).attach_to(base)
# gm.gen_dashstick(np.zeros(3), pos_o, major_radius=.003, rgba=[.3, .3, .3, 1], len_solid=.01, len_interval=.01).attach_to(base)
gm.gen_stick(np.zeros(3), pos_o, radius=.003, rgba=[.3, .3, .3, 1]).attach_to(base)

base.run()
