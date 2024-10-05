import numpy as np
from scipy.spatial import cKDTree
import wrs.basis.robot_math as rm
import wrs.basis.trimesh_factory as trf
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
from wrs.basis.trimesh import graph, geometry, grouping


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


def expand_adj(seed_face_id, trm_mesh, angle_limited_adjacency_mat, max_normal_bias_angle):
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
    date: 20161213, 20210119, 20240321
    """
    vertices = trm_mesh.vertices
    faces = trm_mesh.faces
    face_normals = trm_mesh.face_normals
    seed_face_normal = face_normals[seed_face_id]
    # find all angle-limited faces connected to seed_face_id
    open_set = {seed_face_id}
    close_set = set()
    # pr.enable()
    while True:
        if len(open_set) == 0:
            break
        face_id = list(open_set)[0]
        next_adj_list = angle_limited_adjacency_mat[angle_limited_adjacency_mat[:, 0] == face_id, 1]
        next_adj_set = set(next_adj_list).difference(open_set, close_set)
        next_adj_list = list(next_adj_set)
        angle_array = np.arccos(face_normals[next_adj_list].dot(seed_face_normal.T))  # it seems clipping is not needed
        next_adj_array = np.array(next_adj_list)
        selected_next_adj_set = set(next_adj_array[angle_array < max_normal_bias_angle].tolist())
        close_set.add(face_id)
        open_set.remove(face_id)
        open_set.update(selected_next_adj_set.difference(close_set))
    # pr.disable()
    # pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)
    ## compute curvature TODO all differnece
    # normal angle
    adj_face_id_list = list(close_set)
    adj_face_normal_array = face_normals[adj_face_id_list]
    # angle_array = np.arccos(np.clip(adj_face_normal_array.dot(adj_face_normal_array.T), -1.0, 1.0))
    angle_array = np.arccos(adj_face_normal_array.dot(adj_face_normal_array.T))
    max_angle = np.amax(angle_array)
    # surface linear_distance
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
    return adj_face_id_list, boundary_edges, curvature, face_id_pair_for_curvature  # todo list to nparray


def overlapped_segmentation(model, max_normal_bias_angle=np.pi / 12, toggle_face_id_pair_for_curvature=False):
    """ TODO replace np.arccos with math.cos
    compute the clusters using mesh oversegmentation
    :param model: modeling.CollisionModel or Trimesh
    :param max_normal_bias_angle: the angle between two adjacent faces that are taken as coplanar
    :param seg_angle: the angle between two adjacent segmentations that are taken as coplanar
    :return:
    author: weiwei
    date: 20161116cancun, 20210119osaka, 20240321osaka
    """
    trm_mesh = model
    if isinstance(model, mcm.CollisionModel):
        trm_mesh = model.trm_mesh
    vertices = trm_mesh.vertices
    faces = trm_mesh.faces
    face_normals = trm_mesh.face_normals
    n_faces = len(faces)
    ## angle-limited adjacency -> selection matrix
    angle_limited_adjacency = graph.adjacency_angle(trm_mesh, max_normal_bias_angle)
    angle_limited_adjacency_mat = np.vstack((angle_limited_adjacency, np.fliplr(angle_limited_adjacency)))
    # prepare return values
    seg_nested_face_id_list = []
    seg_nested_edge_list = []
    seg_seed_face_id_list = []
    seg_normal_list = []
    seg_curvature_list = []
    seg_face_id_pair_list_for_curvature = []
    # randomize first seed
    face_ids = list(range(n_faces))
    while True:
        if len(face_ids) == 0:
            break
        # random.shuffle(face_ids) # costly!
        current_face_id = face_ids[0]
        adj_face_id_list, edge_list, curvature, face_id_pair_for_curvature = expand_adj(current_face_id,
                                                                                        trm_mesh,
                                                                                        angle_limited_adjacency_mat,
                                                                                        max_normal_bias_angle)
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


def edge_points(obj_cmodel, radius=.005, max_normal_bias_angle=rm.math.pi / 12):
    """
    get a bunch of points on the edges of an obj_cmodel
    :param radius:
    :return:
    author: weiwei
    date: 20210120
    """
    threshold = math.cos(max_normal_bias_angle)
    points, point_face_ids = obj_cmodel.sample_surface(radius=.001)
    kdt = cKDTree(points)
    point_pairs = np.array(list(kdt.query_pairs(radius)))
    point_normals0 = obj_cmodel.trm_mesh.face_normals[point_face_ids[point_pairs[:, 0].tolist()]]
    point_normals1 = obj_cmodel.trm_mesh.face_normals[point_face_ids[point_pairs[:, 1].tolist()]]
    return points[point_pairs[np.sum(point_normals0 * point_normals1, axis=1) < threshold].ravel()].reshape(-1, 3)


if __name__ == '__main__':
    import os
    import math
    import numpy as np
    import wrs.basis.robot_math as rm
    import wrs.visualization.panda.world as wd
    import cProfile as profile
    import pstats

    pr = profile.Profile()
    pr.disable()
    base = wd.World(cam_pos=[.3, .3, .3], lookat_pos=[0, 0, 0], toggle_debug=True)
    obj_path = os.path.join(os.path.dirname(rm.__file__), 'objects', 'bunnysim.stl')
    bunny_cm = mcm.CollisionModel(obj_path)
    pr.enable()
    facet_nested_face_id_list, seg_nested_edge_list, facet_seed_list, facet_normal_list, facet_curvature_list, face_id_pair_list_for_curvature = overlapped_segmentation(
        bunny_cm, max_normal_bias_angle=math.pi / 6, toggle_face_id_pair_for_curvature=True)
    pr.disable()
    pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)
    # TODO Extract Facet
    for i in range(len(facet_nested_face_id_list)):
        offset_pos = facet_normal_list[i] * np.random.rand() * .0
        # segment
        tmp_trm = trf.facet_as_trm(bunny_cm.trm_mesh, facet_nested_face_id_list[i], offset_pos)  # TODO submesh
        tmp_gm = mgm.StaticGeometricModel(tmp_trm, toggle_twosided=True)
        tmp_gm.attach_to(base)
        tmp_gm.rgba = (rm.random_rgba())
        # edge
        edge_list = (np.array(seg_nested_edge_list[i]) + offset_pos).tolist()
        mgm.gen_linesegs(edge_list, thickness=.001, rgb=rm.const.red).attach_to(base)
        # seed segment
        tmp_trm = trf.facet_as_trm(bunny_cm.trm_mesh, facet_seed_list[i], facet_normal_list[i] * .001)
        tmp_gm = mgm.StaticGeometricModel(tmp_trm, toggle_twosided=True)
        tmp_gm.attach_to(base)
        tmp_gm.rgba = np.array([1, 0, 0, 1])
        # face center and normal
        seed_center = np.mean(tmp_trm.vertices, axis=0)
        mgm.gen_sphere(pos=seed_center, radius=.0003).attach_to(base)
        mgm.gen_arrow(spos=seed_center, epos=seed_center + tmp_trm.face_normals[0] * .01, stick_radius=.0002).attach_to(
            base)
        for face_id_for_curvature in face_id_pair_list_for_curvature[i]:
            tmp_trm = trf.facet_as_trm(bunny_cm.trm_mesh, face_id_for_curvature, offset_pos)
            tmp_gm = mgm.StaticGeometricModel(tmp_trm, toggle_twosided=True)
            tmp_gm.attach_to(base)
            tmp_gm.rgba = np.array([1, 1, 1, 1])
            seed_center = np.mean(tmp_trm.vertices, axis=0)
            mgm.gen_sphere(pos=seed_center, radius=.0003, rgb=rm.const.white, alpha=1).attach_to(base)
            mgm.gen_arrow(spos=seed_center, epos=seed_center + tmp_trm.face_normals[0] * .01, stick_radius=.0002,
                          rgb=rm.const.white, alpha=1).attach_to(base)
    base.run()
