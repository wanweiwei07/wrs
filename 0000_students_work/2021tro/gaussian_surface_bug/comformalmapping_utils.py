# import igl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from localenv import envloader as el
import utiltools.robotmath as rm
import utils.drawpath_utils as du
import utils.pcd_utils as pcdu
import time


def downsample(vertices, faces, downsampling_ratio=0.99, downsampling_step=None):
    origin_num_faces = len(faces)
    if downsampling_step:
        downsampled_num_faces = origin_num_faces - downsampling_step
    else:
        downsampled_num_faces = int(origin_num_faces * downsampling_ratio)
    o3d_trimesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                            o3d.utility.Vector3iVector(faces))

    # TODO: change to ratio implementation
    # o3d_trimesh = o3d_trimesh.simplify_quadric_decimation(5000)
    o3d_trimesh = o3d_trimesh.simplify_quadric_decimation(downsampled_num_faces)
    o3d_trimesh = o3d_trimesh.remove_unreferenced_vertices()
    o3d_trimesh = o3d_trimesh.remove_degenerate_triangles()
    o3d_trimesh = o3d_trimesh.compute_triangle_normals()

    vertices = np.array(o3d_trimesh.vertices)
    faces = np.array(o3d_trimesh.triangles)
    normals = np.array(o3d_trimesh.triangle_normals)

    return vertices, faces, normals


def find_corners(p_narray):
    x_max = max([v[0] for v in p_narray])
    x_min = min([v[0] for v in p_narray])
    y_max = max([v[1] for v in p_narray])
    y_min = min([v[1] for v in p_narray])
    z_max = max([v[2] for v in p_narray])
    z_min = min([v[2] for v in p_narray])
    return x_max, x_min, y_max, y_min, z_max, z_min


def rot_uv(uv, angle=30, toggledebug=False):
    plt.scatter(np.asarray([v[0] for v in uv]), np.asarray([v[1] for v in uv]), color='red', marker='.')
    uv_3d = np.asarray([(v[0], v[1], 0) for v in uv])
    uv_3d = pcdu.trans_pcd(uv_3d, rm.homobuild(np.asarray([0, 0, 0]), rm.rodrigues((0, 0, 1), angle)))
    uv_new = np.asarray([(v[0], v[1]) for v in uv_3d])
    if toggledebug:
        plt.scatter(np.asarray([v[0] for v in uv_new]), np.asarray([v[1] for v in uv_new]), color='green', marker='.')
        plt.show()
    return uv_new


def remove_edge(objcm, edge=5.0):
    vs = objcm.trimesh.vertices
    x_max, x_min, y_max, y_min, z_max, z_min = find_corners(vs)
    print("obj size", x_max - x_min, y_max - y_min, z_max - z_min)
    vs = np.array([v for v in vs if x_min + edge < v[0] < x_max - edge and v[2] > z_min + edge])
    vs_range_x = []
    # for v in vs:
    #     if x_min + edge < v[0] < x_max - edge:
    #         vs_range_x.append(v)
    #     elif v[0] > x_max - edge:
    #         vs_range_x.append((x_max - edge, v[1], v[2]))
    #     else:
    #         vs_range_x.append((x_min + edge, v[1], v[2]))
    vs_range_z = []
    for v in vs:
        if v[2] > z_min + edge:
            vs_range_z.append(v)
        else:
            # pass
            vs_range_z.append((v[0], v[1], z_min + edge))
    print("num of vertices", len(vs_range_z))

    return pcdu.reconstruct_surface(np.asarray(vs_range_z))


def lscm_pcd(objpcd, objnrmls, downsampling_ratio=0.99, downsampling_step=None, toggledebug=False):
    """
    parametrization

    :param objpcd:
    :param objnrmls:
    :param downsampling_ratio:
    :param downsampling_step:
    :param toggledebug:
    :return:
    """
    # TODO: fix normal index error
    objcm = pcdu.reconstruct_surface(objpcd)
    time_start = time.time()
    # obj_cmodel = remove_edge(obj_cmodel, edge=5)
    # obj_cmodel = remove_edge(obj_cmodel, edge=20)
    objcm.trimesh.remove_unreferenced_vertices()
    objcm.trimesh.remove_degenerate_faces()

    vs = objcm.trimesh.vertices
    faces = objcm.trimesh.faces

    print("num of vertices:", len(vs))
    print("num of faces:", len(faces))
    nrmls = []
    objpcd_str = [str(p) for p in objpcd]

    for p in vs:
        try:
            inx = objpcd_str.index(str(p))
            nrmls.append(objnrmls[inx])
        except:
            nrmls.append([0, 0, 1])

    uv = lscm_parametrization(vs, np.array(faces).astype(int))
    while len(uv) == 0:
        print("LSCM failed, downsampling mesh and re-run LSCM")
        vs, faces, nrmls = downsample(vs, faces, downsampling_ratio, downsampling_step)
        uv = lscm_parametrization(vs, faces)
    # uv = rot_uv(uv, angle=18)
    # uv = rot_uv(uv, angle=-60, toggledebug=toggledebug)

    scale_list = []
    for face in faces:
        mesh_3d = [vs[i] for i in face]
        mesh_2d = [uv[i] for i in face]
        scale_list.append(np.linalg.norm(mesh_3d[0] - mesh_3d[1]) / np.linalg.norm(mesh_2d[0] - mesh_2d[1]))
        # scale_list.append(np.linalg.norm(mesh_3d[1] - mesh_3d[2]) / np.linalg.norm(mesh_2d[1] - mesh_2d[2]))
    print("average parametrization scale", np.mean(scale_list))
    print("lscm parametrization time cost", time.time() - time_start)

    # obj_cmodel = pcdu.reconstruct_surface(vertices)
    # obj_cmodel.setColor(0.5, 0.3, 0.5, 1)
    # obj_cmodel.reparentTo(base.render)

    if toggledebug:
        # kdt = KDTree(np.asarray(uv), leaf_size=100, metric='euclidean')
        X = np.asarray([v[0] for v in uv])
        y = np.asarray([v[1] for v in uv])
        plt.scatter(X, y, color='red', marker='.')

        # x_min = np.min(X)
        # x_max = np.max(X)
        # y_max = np.max(y)
        # _, cor_inx_1 = kdt.query([(x_min, y_max)], k=1, return_distance=True)
        # _, cor_inx_2 = kdt.query([(x_max, y_max)], k=1, return_distance=True)
        # cor_1 = uv[cor_inx_1[0][0]]
        # cor_2 = uv[cor_inx_2[0][0]]
        # plt.scatter([cor_1[0]], [cor_1[1]], color='yellow', marker='.')
        # plt.scatter([cor_2[0]], [cor_2[1]], color='yellow', marker='.')

        plt.show()

    return uv, vs, nrmls, faces, scale_list


def lscm_objcm(objcm, toggledebug=False, downsampling_ratio=0.99, downsampling_step=None):
    """
    parametrization

    :param objcm:
    :param toggledebug:
    :param downsampling_ratio:
    :param downsampling_step:
    :return:
    """
    time_start = time.time()
    # obj_cmodel = remove_edge(obj_cmodel, edge=5)
    objcm.trimesh.remove_unreferenced_vertices()
    objcm.trimesh.remove_degenerate_faces()

    vertices = objcm.trimesh.vertices
    faces = objcm.trimesh.faces
    normals = objcm.trimesh.face_normals
    uv = lscm_parametrization(vertices, np.array(faces).astype(int))

    while len(uv) == 0:
        print("LSCM failed, downsampling mesh and re-run LSCM")
        vertices, faces, normals = downsample(vertices, faces, downsampling_ratio, downsampling_step)
        uv = lscm_parametrization(vertices, faces)
    # uv = rot_uv(uv, angle=-60, toggledebug=toggledebug)

    scale_list = []
    for face in faces:
        mesh_3d = [vertices[i] for i in face]
        mesh_2d = [uv[i] for i in face]
        scale_list.append(np.linalg.norm(mesh_3d[0] - mesh_3d[1]) / np.linalg.norm(mesh_2d[0] - mesh_2d[1]))
        # scale_list.append(np.linalg.norm(mesh_3d[1] - mesh_3d[2]) / np.linalg.norm(mesh_2d[1] - mesh_2d[2]))

    print("average parametrization scale", np.mean(scale_list))
    print("lscm parametrization time cost", time.time() - time_start)

    # obj_cmodel = pcdu.reconstruct_surface(vertices)
    # obj_cmodel.setColor(0.5, 0.3, 0.5, 1)
    # obj_cmodel.reparentTo(base.render)

    if toggledebug:
        # kdt = KDTree(np.asarray(uv), leaf_size=100, metric='euclidean')
        X = np.asarray([v[0] for v in uv])
        y = np.asarray([v[1] for v in uv])
        plt.scatter(X, y, color='red', marker='.')

        # x_min = np.min(X)
        # x_max = np.max(X)
        # y_max = np.max(y)
        # _, cor_inx_1 = kdt.query([(1x_min, y_max)], k=1, return_distance=True)
        # _, cor_inx_2 = kdt.query([(x_max, y_max)], k=1, return_distance=True)
        # cor_1 = uv[cor_inx_1[0][0]]
        # cor_2 = uv[cor_inx_2[0][0]]
        # plt.scatter([cor_1[0]], [cor_1[1]], color='yellow', marker='.')
        # plt.scatter([cor_2[0]], [cor_2[1]], color='yellow', marker='.')
        #
        # lr = linear_model.LinearRegression()
        # lr.fit(np.array([[v] for v in X]), y)
        # ransac = linear_model.RANSACRegressor()
        # ransac.fit(np.array([[v] for v in X]), y)
        # line_X = np.array([[X.min()], [X.max()]])
        # ransac_coef = ransac.estimator_.coef_[0]
        # lr_coef = lr.coef_[0]
        # print("coef", lr_coef, ransac_coef)
        # line_y = lr.predict(line_X)
        # line_y_ransac = ransac.predict(line_X)
        # plt.plot(line_X, line_y, color='navy', linewidth=2)
        # plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=2, label='RANSAC regressor')
        plt.show()

    # if True:
    #     obj_cmodel.setColor(1, 1, 1, 0.7)
    #     obj_cmodel.reparentTo(base.render)
    #     for i, n in enumerate(normals):
    #         import random
    #         v = random.choice(range(0, 10))
    #         if v == 1:
    #             base.pggen.plotArrow(base.render, spos=vertices[faces[i][0]], epos=vertices[faces[i][0]] + 10 * n)
    #     base.run()

    return uv, vertices, normals, faces, scale_list


def lscm_parametrization_objcm_temp(objcm, toggledebug=True, sample_num=50000):
    time_start = time.time()
    vs = objcm.trimesh.vertices
    faces = objcm.trimesh.faces
    nrmls = objcm.trimesh.face_normals
    uv = lscm_parametrization(vs, np.array(faces).astype(int))
    # print(max(uv[:, 0]), min(uv[:, 0]), max(uv[:, 1]), min(uv[:, 1]))

    scale_list = []
    for face in faces:
        mesh_3d = [vs[i] for i in face]
        mesh_2d = [uv[i] for i in face]
        scale_list.append(np.linalg.norm(mesh_3d[0] - mesh_3d[1]) / np.linalg.norm(mesh_2d[0] - mesh_2d[1]))

    avg_scale = np.mean(scale_list)
    print("average parametrization scale", avg_scale)
    print("lscm parametrization time cost", time.time() - time_start)

    sampled_xs = np.random.uniform(min(uv[:, 0]), max(uv[:, 0]), sample_num)
    sampled_ys = np.random.uniform(min(uv[:, 1]), max(uv[:, 1]), sample_num)
    random_ps = np.array((sampled_xs, sampled_ys)).T
    plt.scatter(random_ps[:, 0], random_ps[:, 1], color='green', marker='.')

    new_uv = []
    new_vs = []
    new_nrmls = []
    for p_uv in random_ps:
        for face_id, face in enumerate(faces):
            polygon = [uv[i] for i in face]
            if is_in_polygon(p_uv, polygon):
                v_draw = p_uv - polygon[0]
                v_draw = (v_draw[0], v_draw[1], 0)
                rotmat = rm.rotmat_betweenvector(np.array([0, 0, 1]), nrmls[face_id])
                project_p = vs[face[0]] + np.dot(rotmat, v_draw) * scale_list[face_id]
                # base.pggen.plotSphere(base.render, project_p, major_radius=1, rgba=(1, 1, 0, 1))

                new_uv.append(p_uv)
                new_vs.append(project_p)
                new_nrmls.append(nrmls[face_id])
                break

    # obj_cmodel = pcdu.reconstruct_surface(np.asarray(new_vs))
    # vs = obj_cmodel.trimesh.vertices
    # faces = obj_cmodel.trimesh.faces
    # nrmls = []
    # new_uv_sort = []
    # for i, p_i in enumerate(vs):
    #     for j, p_j in enumerate(new_vs):
    #         if p_j is None:
    #             continue
    #         if str(p_i) == str(p_j):
    #             new_uv_sort.append(new_uv[j])
    #             if p_j[2] == 100.:
    #                 nrmls.append([0, 0, 1])
    #             else:
    #                 nrmls.append([0, -1, 0])
    #             new_vs[j] = None
    #             break
    # print(len(vs), len(new_uv), len(new_uv_sort))

    # obj_cmodel.setColor(0.5, 0.3, 0.5, 1)
    # obj_cmodel.reparentTo(base.render)
    # base.run()

    if toggledebug:
        plt.scatter(uv[:, 0], uv[:, 1], color='red', marker='.')
        plt.show()
    return np.asarray(new_uv), new_vs, np.asarray(new_nrmls), faces, scale_list


def lscm_parametrization(v, f):
    print("num of vertices:", len(v))
    print("num of faces:", len(f))
    bnd = igl.boundary_loop(f)

    # for bnd_value in bnd_value_list:
    #     base.pggen.plotSphere(base.render, bnd_value, major_radius=2, rgba=(1, 1, 0, 0.5))

    # b = np.asarray([bnd[0], bnd[int(bnd.size / 2)]])
    b = np.asarray([bnd[0], bnd[int(bnd.size) - 1]])
    bc = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    _, uv = igl.lscm(v, f, b, bc)

    return uv


def get_uv_center(uv):
    return np.array((np.mean(uv[:, 0]), np.mean(uv[:, 1])))


def flatten_nested_list(nested_list):
    return [p for s in nested_list for p in s]


def is_in_polygon(pt, poly):
    result = False
    i = -1
    l = len(poly)
    j = l - 1
    while i < l - 1:
        i += 1
        # print(i, poly[i], j, poly[j])
        if (poly[i][0] <= pt[0] < poly[j][0]) or (poly[j][0] <= pt[0] < poly[i][0]):
            if pt[1] < (poly[j][1] - poly[i][1]) * (pt[0] - poly[i][0]) / (poly[j][0] - poly[i][0]) + poly[i][1]:
                result = not result
        j = i
    return result


if __name__ == '__main__':
    '''
    set up env and param
    '''
    import pandaplotutils.pandactrl as pc

    base = pc.World(camp=[0, 0, 1000], lookatpos=[0, 0, 0])
    sample_num = 10000

    objcm = el.loadObj("cylinder_surface.stl")
    objcm.reparentTo(base.render)
    # el.loadObj("cylinder.stl", transparency=1).reparentTo(base.render)

    drawpath_ms = du.gen_grid(side_len=80)
    # objpcd, objnormals = pcdu.get_objpcd_withnormals(obj_cmodel, objmat4=np.eye(4), sample_num=sample_num, toggledebug=False)
    # uvs, vs, normals, avg_scale = lscm_pcd(objpcd, objnormals, toggledebug=True)
    uvs, vs, normals, faces, avg_scale = lscm_objcm(objcm, toggledebug=True)
    uv_center = get_uv_center(uvs)
