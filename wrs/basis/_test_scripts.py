# import numpy as np
# import matplotlib.pyplot as plt
# import visualization.panda.world as world
# import robot_math as rm
# import modeling.geometric_model as mgm
#
# sp2d = rm.gen_2d_spiral_points(max_radius=.2, radial_granularity=.001, tangential_granularity=.01)
# plt.plot(sp2d[:,0], sp2d[:,1])
# plt.show()
#
# base = world.World(cam_pos=np.array([1, 1, 1]), lookat_pos=np.array([0, 0, 0.25]))
# sp = rm.gen_3d_spiral_points(pos=np.array([0, 0, .25]),
#                              rotmat=rm.rotmat_from_axangle(np.array([1, 0, 0]), np.pi / 6),
#                              max_radius=.20,
#                              radial_granularity=.001,
#                              tangential_granularity=.01,)
# for id in range(len(sp) - 1):
#     pnt0 = sp[id, :]
#     pnt1 = sp[id + 1, :]
#     mgm.gen_stick(spos=pnt0, epos=pnt1, end_type="round").attach_to(base)
# base.run()

import time
import numpy as np
import math
import matplotlib.pyplot as plot


def concentric_circle_hex_polar(layer, radians, start_rot_angle=0.):
    def get_param(layer_id):
        radians_base = 0.866025 * radians * (layer_id + 1)
        n_list = np.linspace(layer_id - 1, 0, int((layer_id + 1) / 2))
        angle_diff = np.append(np.array([math.pi / 6]), np.arctan(n_list / ((layer_id + 1) * 1.732051)))
        # print("angle_diff:", angle_diff)
        angle_minus = np.zeros(len(angle_diff))
        angle_minus[:-1] = angle_diff[1:]
        angle_half = angle_diff - angle_minus
        angle_list = np.append(angle_half, np.zeros(int(layer_id / 2)))
        angle_list = angle_list + angle_list[::-1]
        # print("angle_list:", len(angle_list), angle_list)
        angle_diff_total = np.append(angle_diff[1:], angle_diff[1:][::-1][(layer_id % 2):])
        radians_list = np.append(radians_base / np.cos(angle_diff_total), radians * (layer_id + 1))
        # print("radiasn_list:", len(radians_list), radians_list)
        return angle_list, radians_list

    def get_pose_from_angle(angle_list, radians_list):
        # angle_list[0]  += start_rot_angle
        angle_list_total = np.cumsum(np.tile(angle_list, 6))
        angle_list_total = angle_list_total + np.array([start_rot_angle]).repeat(len(angle_list_total))
        radians_list_total = np.tile(radians_list, 6)
        x_list = radians_list_total * np.sin(angle_list_total)
        y_list = radians_list_total * np.cos(angle_list_total)
        return x_list, y_list

    x_list = np.array([])
    y_list = np.array([])
    for layer_id in range(layer):
        # print("layer_id", layer_id)
        angle_list, radians_list = get_param(layer_id)
        x_layer, y_layer = get_pose_from_angle(angle_list, radians_list)
        x_list = np.append(x_list, x_layer)
        y_list = np.append(y_list, y_layer)

    return x_list, y_list


def concentric_circle_hex_equipartition(layer, radians, start_rot_angle=0.):
    def get_hex(layer_id):
        angle_list = np.arange(start_rot_angle, (math.pi * 2 + start_rot_angle), math.pi / 3)
        angle_list = np.append(angle_list, start_rot_angle)
        x_vertex = np.sin(angle_list) * radians * (layer_id + 1)
        y_vertex = np.cos(angle_list) * radians * (layer_id + 1)
        return x_vertex, y_vertex

    x_list = np.array([])
    y_list = np.array([])
    for layer_id in range(layer):
        x_vertex, y_vertex = get_hex(layer_id)
        for i in range(6):
            x_list = np.append(x_list, np.linspace(x_vertex[i], x_vertex[i + 1], num=layer_id + 1, endpoint=False))
            y_list = np.append(y_list, np.linspace(y_vertex[i], y_vertex[i + 1], num=layer_id + 1, endpoint=False))
    return x_list, y_list


def gen_regpoly(radius, nedges=12):
    angle_list = np.linspace(0, np.pi * 2, nedges+1, endpoint=True)
    x_vertex = np.sin(angle_list) * radius
    y_vertex = np.cos(angle_list) * radius
    return np.column_stack((x_vertex, y_vertex))

def gen_2d_isosceles_verts(nlevel, edge_length, nedges=12):
    xy_array = np.asarray([[0,0]])
    for level in range(nlevel):
        xy_vertex = gen_regpoly(radius=edge_length*(level+1), nedges=nedges)
        for i in range(nedges):
            xy_array = np.append(xy_array,
                                np.linspace(xy_vertex[i, :], xy_vertex[i + 1, :], num=level + 1, endpoint=False),
                                axis=0)
    return xy_array

def gen_2d_equilateral_verts(nlevel, edge_length):
    return gen_2d_isosceles_verts(nlevel=nlevel, edge_length=edge_length, nedges=6)

def gen_3d_isosceles_verts(pos, rotmat, nlevel=5, edge_length=0.001, nedges=12):
    xy_array = gen_2d_isosceles_verts(nlevel=nlevel, edge_length=edge_length, nedges=nedges)
    xyz_array = np.pad(xy_array, ((0,0), (0,1)), mode='constant', constant_values=0)
    return rotmat.dot((xyz_array+pos).T).T

def gen_3d_equilateral_verts(pos, rotmat, nlevel=5, edge_length=0.001):
    return gen_3d_isosceles_verts(pos=pos, rotmat=rotmat, nlevel=nlevel, edge_length=edge_length, nedges=6)

def gen_2d_equilaterial_verts(nlevel, edge_length):
    nangle = 12
    levels = np.arange(1, nlevel + 1, 1) * edge_length
    angles = np.linspace(0, np.pi * 2, nangle+1, endpoint=True)
    x_verts = np.outer(levels, np.sin(angles)).flatten()
    y_verts = np.outer(levels, np.cos(angles)).flatten()
    xy_vertex = np.row_stack((x_verts, y_verts)).T
    xy_list = np.empty((0, 2))
    for level in range(nlevel):
        for i in range(nangle):
            xy_list = np.append(xy_list,
                                np.linspace(xy_vertex[level*(nangle+1)+i, :], xy_vertex[level*(nangle+1)+i + 1, :], num=level + 1, endpoint=False),
                                axis=0)
    return xy_list

if __name__ == "__main__":
    # tic = time.time()
    # for i in range(200):
    #     x_list, y_list = concentric_circle_hex_polar(5, 1, math.pi / 8)
    # toc1 = time.time()
    # print(toc1 - tic)
    # tic = time.time()
    # for i in range(200):
    #     x_list, y_list = concentric_circle_hex_equipartition(5, 1, math.pi / 8)
    # toc1 = time.time()
    # print(toc1 - tic)
    # tic = time.time()
    # for i in range(200):
    #     xy_list = gen_2d_isosceles_verts(5, 1,12)
    # toc1 = time.time()
    # print(toc1 - tic)
    # # for i in range(200):
    # #     xy_list = gen_2d_equilaterial_verts(5, 1)
    # # toc1 = time.time()
    # # print(toc1 - tic)
    #
    # fig = plot.figure()
    # ax = fig.add_subplot(111)
    # ax.set_aspect('equal', 'box')
    #
    # plot.plot(xy_list[:,0], xy_list[:,1], "o-")
    # # plot.plot(x_list[:], y_list[:], "o-")
    # plot.show()

    def func(x,y,z):
        return x+y+z

    a = [1,2,3]
    print(func(*a))