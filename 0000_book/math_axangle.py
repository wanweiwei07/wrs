import visualization.panda.world as wd
import modeling.geometric_model as gm
import basis.robot_math as rm
import math
import numpy as np

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0], toggle_debug=True)
gm.gen_frame(axis_length=.2).attach_to(base)
gm.gen_torus(axis=[0, 0, 1], portion=1, radius=.2, thickness=.003, rgba=[1, 1, 0, 1],
             sections=16, discretization=64).attach_to(base)
rotmat = rm.rotmat_from_euler(math.pi / 3, -math.pi / 6, math.pi / 3)
gm.gen_dashed_frame(axis_length=.2, rotmat=rotmat).attach_to(base)
gm.gen_dashtorus(axis=rotmat[:3,2], portion=1, radius=.2, thickness=.003, rgba=[1, 1, 0, 1],
                 len_interval=.007, len_solid=.01, n_sec=16, n_sec_major=64).attach_to(base)
# mgm.gen_sphere(major_radius=.2, rgba=[.67,.67,.67,.9], sphere_ico_level=5).attach_to(base)
#
# cross_vec = rm.unit_vector(np.cross(np.array([0,0,1]), rotmat[:3,2]))
# # mgm.gen_dasharrow(epos=cross_vec*.2, rgba=[1,1,0,1], len_solid=.01, len_interval=.0077).attach_to(base)
# mgm.gen_arrow(epos=cross_vec*.2, rgba=[1,1,0,1]).attach_to(base)
# mgm.gen_sphere(major_radius=.005, pos=cross_vec*.2, rgba=[0,0,0,1]).attach_to(base)
# #
# nxt_vec_uvw = rotmat.dot(cross_vec)
# # mgm.gen_dasharrow(epos=nxt_vec_uvw*.2, rgba=[1,1,0,1], len_solid=.015, len_interval=.007).attach_to(base)
# mgm.gen_arrow(epos=nxt_vec_uvw*.2, rgba=[1,1,0,1]).attach_to(base)
# mgm.gen_sphere(major_radius=.005, pos=nxt_vec_uvw*.2, rgba=[0,0,0,1]).attach_to(base)
# #
# pre_vec_xyz = rotmat.T.dot(cross_vec)
# mgm.gen_arrow(epos=pre_vec_xyz*.2, rgba=[1,1,0,1]).attach_to(base)
# mgm.gen_sphere(major_radius=.005, pos=pre_vec_xyz*.2, rgba=[0,0,0,1]).attach_to(base)
#
ax, angle = rm.axangle_between_rotmat(np.eye(3), rotmat)
gm.gen_arrow(epos=ax*.4, rgba=[0,0,0,1]).attach_to(base)
for step_angle in np.linspace(0, angle, 10).tolist():
    rotmat = rm.rotmat_from_axangle(ax, step_angle)
    gm.gen_dashed_frame(axis_length=.2, rotmat=rotmat).attach_to(base)
    gm.gen_dashtorus(axis=rotmat[:3,2], portion=1, radius=.2, thickness=.003, rgba=[1, 1, 0, 1],
                     len_interval=.007, len_solid=.01, n_sec=16, n_sec_major=64).attach_to(base)
gm.gen_sphere(radius=.2, rgba=[.67,.67,.67,.9], ico_level=5).attach_to(base)
# major_radius, _ = rm.unit_vector(cross_vec*.2-cross_vec.dot(ax)*ax*.2, toggle_length=True)
# print(major_radius)
# mgm.gen_circarrow(ax, portion=1, center=cross_vec.dot(ax)*ax*.2, major_radius=major_radius, n_sec_major=64, n_sec_minor=16, major_radius=.003, rgba=[0,0,0,1]).attach_to(base)

# print(ax)
# vec=np.array([1,0,0])
# major_radius, _ = rm.unit_vector(vec*.2-vec.dot(ax)*ax*.2, toggle_length=True)
# mgm.gen_circarrow(ax, starting_vector=vec-vec.dot(ax)*ax, portion=.57, center=vec.dot(ax)*ax*.2, major_radius=major_radius, n_sec_major=64, n_sec_minor=16, major_radius=.003, rgba=[1,0,0,1]).attach_to(base)
# vec=np.array([0,1,0])
# major_radius, _ = rm.unit_vector(vec*.2-vec.dot(ax)*ax*.2, toggle_length=True)
# mgm.gen_circarrow(ax, portion=1, center=vec.dot(ax)*ax*.2, major_radius=major_radius, n_sec_major=64, n_sec_minor=16, major_radius=.003, rgba=[0,1,0,1]).attach_to(base)
# vec=np.array([0,0,1])
# major_radius, _ = rm.unit_vector(vec*.2-vec.dot(ax)*ax*.2, toggle_length=True)
# mgm.gen_circarrow(ax, starting_vector=vec-vec.dot(ax)*ax, portion=.57, center=vec.dot(ax)*ax*.2, major_radius=major_radius, n_sec_major=64, n_sec_minor=16, major_radius=.003, rgba=[0,0,1,1]).attach_to(base)


base.run()
