import numpy as np
import random as random
import basis.robotmath as rm
import basis.trimesh.graph as tgph


def _expand_adj(objtrm, seed_face_id, angle_limited_adjacency):
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
    # find all angle-limited faces connected to seed_face_id
    angle_limited_adjacency[:,0] == seed_face_id
    sel_array0 = objtrm.face_adjacency[:, 0] == seed_face_id
    sel_array1 = objtrm.face_adjacency[:, 1] == seed_face_id
    adj_face_ids = objtrm.face_adjacency[sel_array0][:, 0].tolist()+objtrm.face_adjacency[sel_array1][:, 1].tolist()
    ## compute curvature
    # normal angle
    seed_face_normal = objtrm.face_normals[seed_face_id]
    angle_array = np.arccos(objtrm.face_normals[adj_face_ids].dot(seed_face_normal))
    max_angle = np.amax(angle_array)
    # surface distance
    max_id = np.argmax(angle_array)
    seed_face_center = np.mean(objtrm.vertices[objtrm.faces[seed_face_id]], axis=1)
    max_angle_face_center = np.mean(objtrm.vertices[adj_face_ids[max_id]], axis=1)
    distance = np.linalg.norm(max_angle_face_center-seed_face_center)
    # curvature
    curvature = max_angle/distance if distance != 0 else 0
    return adj_face_ids, curvature


def facets_from_over_segmentation(objcm, face_angle=rm.math.pi/6, seg_angle=rm.math.pi/7):
    """
    compute the clusters using mesh oversegmentation
    :param objcm: objcm
    :param face_angle: the angle between two adjacent faces that are taken as coplanar
    :param seg_angle: the angle between two adjacent segmentations that are taken as coplanar
    :return: the same as facets
    author: weiwei
    date: 20161116cancun, 20210119osaka
    """
    # angle-limited adjacency
    angle_limited_adjacency = tgph.adjacency_angle(objcm.objtrm, face_angle)
    # convert seg_angle to cosine
    theshold_newseg = rm.math.cos(seg_angle)
    face_ids = list(range(len(objcm.objtrm.faces)))
    random.shuffle(face_ids)
    seed_face_list = [face_ids[0]]
    for i in face_ids[1:]:
        if np.any(objcm.objtrm.face_normals[np.array(seed_face_list)].dot(objcm.objtrm.face_normals[i])>theshold_newseg):
            seed_face_list.append(i)
    facet_nested_face_ids = []
    facet_normal_list = []
    facet_curvature_list = []
    for face_id in seed_face_list:
        face_ids, curvature = _expand_adj(objcm.objtrm, face_id, angle_limited_adjacency)
        facet_nested_face_ids.append(face_ids)
        facet_normal_list.append(objcm.objtrm.face_normals[face_id])
        facet_curvature_list.append(curvature)
    return [facet_nested_face_ids, facet_normal_list, facet_curvature_list]

if __name__ == '__main__':
    pass
