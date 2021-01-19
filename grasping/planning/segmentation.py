import numpy as np
import random as random
import basis.robotmath as rm
import basis.trimesh.graph as tgph


def _expand_adj(objtrm, seed_face_id, second_ele_mat, sel_mat, face_angle):
    """
    find the adjacency of a face
    the normal of the newly added face should be coherent with the normal of the seed_face
    this is an iterative function
    :param: objtrm: basis.trimesh.Trimesh
    :param: seed_face_id: Index of the face to expand
    :param: face_angle: the angle of adjacent faces that are taken as coplanar
    :param: angle_limited_adjacency: should be computed outside to avoid repeatition
    :return: a list of face_ids
    author: weiwei
    date: 20161213, 20210119
    """
    seed_face_normal = objtrm.face_normals[seed_face_id]
    # find all angle-limited faces connected to seed_face_id
    adj_face_id_set = set([seed_face_id])
    open_set = set([seed_face_id])
    close_set = set()
    while True:
        if len(open_set) == 0:
            break
        face_id = list(open_set)[0]
        close_face_normals = objtrm.face_normals[list(adj_face_id_set)]
        next_adj_list = second_ele_mat[:, face_id][sel_mat[:, face_id]].tolist()
        next_adj_set = set(next_adj_list).difference(adj_face_id_set)
        next_adj_list = list(next_adj_set)
        angle_array = np.arccos(np.clip(objtrm.face_normals[next_adj_list].dot(close_face_normals.T), -1.0, 1.0))
        next_adj_array = np.array(next_adj_list)
        sel_array = np.amax(angle_array, axis=1) < face_angle
        selected_next_adj_set = set(next_adj_array[sel_array].tolist())
        adj_face_id_set.update(selected_next_adj_set)
        close_set.add(face_id)
        open_set.remove(face_id)
        open_set.update(selected_next_adj_set.difference(close_set))
    ## compute curvature
    # normal angle
    adj_face_id_list = list(adj_face_id_set)
    angle_array = np.arccos(np.clip(objtrm.face_normals[adj_face_id_list].dot(seed_face_normal), -1.0, 1.0))
    max_angle = np.amax(angle_array)
    # surface distance
    max_id = np.argmax(angle_array)
    seed_face_center = np.mean(objtrm.vertices[objtrm.faces[seed_face_id]], axis=1)
    max_angle_face_center = np.mean(objtrm.vertices[objtrm.faces[adj_face_id_list[max_id]]], axis=1)
    distance = np.linalg.norm(max_angle_face_center - seed_face_center)
    # curvature
    curvature = max_angle / distance if distance != 0 else 0
    return adj_face_id_list, curvature


def facets_from_over_segmentation(objcm, face_angle=rm.math.pi / 3):
    """
    compute the clusters using mesh oversegmentation
    :param objcm: objcm
    :param face_angle: the angle between two adjacent faces that are taken as coplanar
    :param seg_angle: the angle between two adjacent segmentations that are taken as coplanar
    :return:
    author: weiwei
    date: 20161116cancun, 20210119osaka
    """
    nfaces = len(objcm.objtrm.faces)
    # angle-limited adjacency -> selection matrix
    angle_limited_adjacency = tgph.adjacency_angle(objcm.objtrm, face_angle)
    tmp_flipped = np.fliplr(angle_limited_adjacency)
    adjacency_mat = np.tile(np.array(angle_limited_adjacency.tolist() + tmp_flipped.tolist()), nfaces)
    face_id_mat = np.tile(np.array([range(nfaces)]).T, adjacency_mat.shape[0]).T
    second_ele_mat = adjacency_mat[:, 1::2]
    sel_mat = (adjacency_mat[:, ::2] - face_id_mat == 0)
    # prepare return values
    facet_nested_face_id_list = []
    facet_seed_list = []
    facet_normal_list = []
    facet_curvature_list = []
    # randomize first seed
    face_ids = list(range(len(objcm.objtrm.faces)))
    while True:
        if len(face_ids) == 0:
            break
        random.shuffle(face_ids)
        current_face_id = face_ids[0]
        adj_face_id_list, curvature = _expand_adj(objcm.objtrm, current_face_id, second_ele_mat, sel_mat, face_angle)
        facet_nested_face_id_list.append(adj_face_id_list)
        facet_seed_list.append(current_face_id)
        facet_normal_list.append(objcm.objtrm.face_normals[current_face_id])
        facet_curvature_list.append(curvature)
        face_ids = list(set(face_ids)-set(sum(facet_nested_face_id_list, [])))
    return [facet_nested_face_id_list, facet_seed_list, facet_normal_list, facet_curvature_list]


if __name__ == '__main__':
    import os
    import math
    import numpy as np
    import basis
    import basis.robotmath as rm
    import visualization.panda.world as wd
    import modeling.collisionmodel as cm
    import basis.trimesh as trm

    base = wd.World(campos=[.3, .3, .3], lookatpos=[0, 0, 0], toggledebug=True)
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    bunnycm = cm.CollisionModel(objpath)
    facet_nested_face_id_list, facet_seed_list, facet_normal_list, facet_curvature_list = facets_from_over_segmentation(bunnycm)
    # TODO Extract Facet
    for i in range(len(facet_nested_face_id_list)):
        tmp_vertices = bunnycm.objtrm.vertices[bunnycm.objtrm.faces[facet_nested_face_id_list[i]].flatten()]
        tmp_faces = np.array(range(len(tmp_vertices))).reshape(-1, 3)
        tmp_cm = cm.CollisionModel(trm.Trimesh(vertices=tmp_vertices, faces=tmp_faces), btwosided=True)
        tmp_cm.attach_to(base)
        tmp_cm.set_rgba(rm.random_rgba())
        current_pos = tmp_cm.get_pos()
        new_pos = current_pos+facet_normal_list[i]*np.random.rand()*.1
        tmp_cm.set_pos(new_pos)
        tmp_vertices = bunnycm.objtrm.vertices[bunnycm.objtrm.faces[facet_seed_list[i]].flatten()]
        tmp_faces = np.array(range(len(tmp_vertices))).reshape(-1, 3)
        tmp_cm = cm.CollisionModel(trm.Trimesh(vertices=tmp_vertices, faces=tmp_faces), btwosided=True)
        tmp_cm.attach_to(base)
        tmp_cm.set_rgba([1,0,0,1])
        tmp_cm.set_pos(new_pos)
    base.run()
