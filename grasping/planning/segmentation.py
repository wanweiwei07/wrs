import math
import numpy as np
import random as random
import basis.robot_math as rm
import basis.trimesh.graph as graph
import basis.trimesh.geometry as geometry
import basis.trimesh.grouping as grouping
from scipy.spatial.ckdtree import cKDTree


def extract_boundary(vertices, facet):
    """
    :param facet:
    :return:
    author: weiwei
    date: 20210122
    """
    edges = geometry.faces_to_edges(facet)
    edges_sorted = np.sort(edges, axis=1)
    edges_boundary = edges[grouping.group_rows(edges_sorted, require_count=1)]
    return vertices[edges_boundary].reshape(-1, 2, 3)


def expand_adj(vertices, faces, face_normals, seed_face_id, rgt_ele_mat, sel_mat, max_normal_bias_angle):
    """
    find the adjacency of a face
    the normal of the newly added face should be coherent with the normal of the seed_face
    this is an iterative function
    :param: vertices, faces, face_normals: np.arrays
    :param: seed_face_id: Index of the face to expand
    :param: face_angle: the angle of adjacent faces that are taken as coplanar
    :param: angle_limited_adjacency: should be computed outside to avoid repeatition
    :return: a list of face_ids
    author: weiwei
    date: 20161213, 20210119
    """
    seed_face_normal = face_normals[seed_face_id]
    # find all angle-limited faces connected to seed_face_id
    adj_face_id_set = set([seed_face_id])
    open_set = set([seed_face_id])
    close_set = set()
    # pr.enable()
    while True:
        if len(open_set) == 0:
            break
        face_id = list(open_set)[0]
        next_adj_list = rgt_ele_mat[:, face_id][sel_mat[:, face_id]].tolist()
        next_adj_set = set(next_adj_list).difference(adj_face_id_set)
        next_adj_list = list(next_adj_set)
        angle_array = np.arccos(np.clip(face_normals[next_adj_list].dot(seed_face_normal.T), -1.0, 1.0))
        next_adj_array = np.array(next_adj_list)
        sel_array = angle_array < max_normal_bias_angle
        selected_next_adj_set = set(next_adj_array[sel_array].tolist())
        adj_face_id_set.update(selected_next_adj_set)
        close_set.add(face_id)
        open_set.remove(face_id)
        open_set.update(selected_next_adj_set.difference(close_set))
    # pr.disable()
    # pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)
    ## compute curvature TODO all differnece
    # normal angle
    adj_face_id_list = list(adj_face_id_set)
    adj_face_normal_array = face_normals[adj_face_id_list]
    angle_array = np.arccos(np.clip(adj_face_normal_array.dot(adj_face_normal_array.T), -1.0, 1.0))
    max_angle = np.amax(angle_array)
    # surface distance
    adj_id_pair_for_curvature = np.unravel_index(angle_array.argmax(), angle_array.shape)
    face_id_pair_for_curvature = (
        adj_face_id_list[adj_id_pair_for_curvature[0]], adj_face_id_list[adj_id_pair_for_curvature[1]])
    face0_center = np.mean(vertices[faces[face_id_pair_for_curvature[0]]], axis=0)
    face1_center = np.mean(vertices[faces[face_id_pair_for_curvature[1]]], axis=0)
    distance = np.linalg.norm(face0_center - face1_center)
    # curvature
    curvature = max_angle / distance if distance != 0 else 0
    # boundary
    adj_faces = faces[adj_face_id_list]
    boundary_edges = extract_boundary(vertices, adj_faces)
    return adj_face_id_list, boundary_edges, curvature, face_id_pair_for_curvature # todo list to nparray


def over_segmentation(objcm, max_normal_bias_angle=rm.math.pi / 12,
                      toggle_face_id_pair_for_curvature=False):
    """ TODO replace np.arccos with math.cos
    compute the clusters using mesh oversegmentation
    :param objcm: objcm
    :param max_normal_bias_angle: the angle between two adjacent faces that are taken as coplanar
    :param seg_angle: the angle between two adjacent segmentations that are taken as coplanar
    :return:
    author: weiwei
    date: 20161116cancun, 20210119osaka
    """
    vertices = objcm.objtrm.vertices
    faces = objcm.objtrm.faces
    face_normals = objcm.objtrm.face_normals
    nfaces = len(faces)
    ## angle-limited adjacency -> selection matrix
    angle_limited_adjacency = graph.adjacency_angle(objcm.objtrm, max_normal_bias_angle)
    adjacency_mat = np.tile(np.vstack((angle_limited_adjacency, np.fliplr(angle_limited_adjacency))), nfaces)
    face_id_mat = np.tile(np.array([range(nfaces)]).T, len(adjacency_mat)).T
    rgt_ele_mat = adjacency_mat[:, 1::2]
    sel_mat = (adjacency_mat[:, ::2] - face_id_mat == 0)
    # prepare return values
    seg_nested_face_id_list = []
    seg_nested_edge_list = []
    seg_seed_face_id_list = []
    seg_normal_list = []
    seg_curvature_list = []
    seg_face_id_pair_list_for_curvature = []
    # randomize first seed
    face_ids = list(range(nfaces))
    while True:
        if len(face_ids) == 0:
            break
        # random.shuffle(face_ids) # costly!
        current_face_id = face_ids[0]
        adj_face_id_list, edge_list, curvature, face_id_pair_for_curvature = \
            expand_adj(vertices, faces, face_normals, current_face_id, rgt_ele_mat, sel_mat, max_normal_bias_angle)
        seg_nested_face_id_list.append(adj_face_id_list)
        seg_nested_edge_list.append(edge_list)
        seg_seed_face_id_list.append(current_face_id)
        seg_normal_list.append(face_normals[current_face_id])
        seg_curvature_list.append(curvature)
        seg_face_id_pair_list_for_curvature.append(face_id_pair_for_curvature)
        face_ids = list(set(face_ids) - set(sum(seg_nested_face_id_list, [])))
    if toggle_face_id_pair_for_curvature:
        return [seg_nested_face_id_list, seg_nested_edge_list, seg_seed_face_id_list, seg_normal_list,
                seg_curvature_list, seg_face_id_pair_list_for_curvature]
    else:
        return [seg_nested_face_id_list, seg_nested_edge_list, seg_seed_face_id_list, seg_normal_list,
                seg_curvature_list]


def edge_points(objcm, radius=.005, max_normal_bias_angle=rm.math.pi / 12):
    """
    get a bunch of points on the edges of a objcm
    :param radius:
    :return:
    author: weiwei
    date: 20210120
    """
    threshold = math.cos(max_normal_bias_angle)
    points, point_face_ids = objcm.sample_surface(radius=.001)
    kdt = cKDTree(points)
    point_pairs = np.array(list(kdt.query_pairs(radius)))
    point_normals0 = objcm.objtrm.face_normals[point_face_ids[point_pairs[:, 0].tolist()]]
    point_normals1 = objcm.objtrm.face_normals[point_face_ids[point_pairs[:, 1].tolist()]]
    return points[point_pairs[np.sum(point_normals0 * point_normals1, axis=1) < threshold].ravel()].reshape(-1, 3)


if __name__ == '__main__':
    import os
    import math
    import numpy as np
    import basis
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.collision_model as cm
    import modeling.geometric_model as gm
    import basis.trimesh_generator as tg
    import cProfile as profile
    import pstats

    pr = profile.Profile()
    pr.disable()
    base = wd.World(cam_pos=[.3, .3, .3], lookat_pos=[0, 0, 0], toggle_debug=True)
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    bunnycm = cm.CollisionModel(objpath)
    pr.enable()
    facet_nested_face_id_list, seg_nested_edge_list, facet_seed_list, facet_normal_list, facet_curvature_list, face_id_pair_list_for_curvature = over_segmentation(
        bunnycm, max_normal_bias_angle=math.pi / 6, toggle_face_id_pair_for_curvature=True)
    pr.disable()
    pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)
    # TODO Extract Facet
    for i in range(len(facet_nested_face_id_list)):
        offset_pos = facet_normal_list[i] * np.random.rand() * .0
        # segment
        tmp_trm = tg.extract_subtrimesh(bunnycm.objtrm, facet_nested_face_id_list[i], offset_pos)  # TODO submesh
        tmp_gm = gm.StaticGeometricModel(tmp_trm, btwosided=True)
        tmp_gm.attach_to(base)
        tmp_gm.set_rgba(rm.random_rgba())
        # edge
        edge_list = (np.array(seg_nested_edge_list[i])+offset_pos).tolist()
        gm.gen_linesegs(edge_list, thickness=.05, rgba=[1,0,0,1]).attach_to(base)
        # seed segment
        tmp_trm = tg.extract_subtrimesh(bunnycm.objtrm, facet_seed_list[i], offset_pos)
        tmp_gm = gm.StaticGeometricModel(tmp_trm, btwosided=True)
        tmp_gm.attach_to(base)
        tmp_gm.set_rgba([1, 0, 0, 1])
        # face center and normal
        seed_center = np.mean(tmp_trm.vertices, axis=0)
        gm.gen_sphere(pos=seed_center, radius=.001).attach_to(base)
        gm.gen_arrow(spos=seed_center, epos=seed_center + tmp_trm.face_normals[0] * .01, thickness=.0006).attach_to(
            base)
        for face_id_for_curvature in face_id_pair_list_for_curvature[i]:
            rgba = [1, 1, 1, 1]
            tmp_trm = tg.extract_subtrimesh(bunnycm.objtrm, face_id_for_curvature, offset_pos)
            tmp_gm = gm.StaticGeometricModel(tmp_trm, btwosided=True)
            tmp_gm.attach_to(base)
            tmp_gm.set_rgba(rgba)
            seed_center = np.mean(tmp_trm.vertices, axis=0)
            gm.gen_sphere(pos=seed_center, radius=.001, rgba=rgba).attach_to(base)
            gm.gen_arrow(spos=seed_center, epos=seed_center + tmp_trm.face_normals[0] * .01, thickness=.0006,
                         rgba=rgba).attach_to(base)
    base.run()
