import numpy as np
import copy
import random
import networkx as nx
import math

from collections import deque

from .constants import log, tol
from .grouping import group, group_rows, boolean_rows
from .geometry import faces_to_edges
from .util import diagonal_dot

try:
    from graph_tool import Graph as GTGraph
    from graph_tool.topology import label_components

    _has_gt = True
except:
    _has_gt = False
    log.warning('graph-tool unavailable, some operations will be much slower')


def face_adjacency(faces, return_edges=False):
    '''
    Returns an (n,2) list of face indices.
    Each pair of faces in the list shares an edge, making them adjacent.


    Arguments
    ----------
    faces: (n, d) int, set of faces referencing vertices by index
    return_edges: bool, return the edges shared by adjacent faces

    Returns
    ---------
    adjacency: (m,2) int, indexes of faces that are adjacent
    
    if return_edges: 
         edges: (m,2) int, indexes of vertices which make up the
                 edges shared by the adjacent faces

    Example
    ----------
    This is useful for lots of things, such as finding connected components:

    graph = nx.Graph()
    graph.add_edges_from(mesh.face_adjacency)
    groups = nx.connected_components(graph_connected)
    '''

    # first generate the list of edges for the current faces
    # also return the index for which face the edge is from
    edges, edge_face_index = faces_to_edges(faces, return_index=True)
    edges.sort(axis=1)
    # this will return the indices for duplicate edges
    # every edge appears twice in a well constructed mesh
    # so for every row in edge_idx, edges[edge_idx[*][0]] == edges[edge_idx[*][1]]
    # in this call to group rows, we discard edges which don't occur twice
    edge_groups = group_rows(edges, require_count=2)

    if len(edge_groups) == 0:
        log.error('No adjacent faces detected! Did you merge vertices?')

    # the pairs of all adjacent faces
    # so for every row in face_idx, self.faces[face_idx[*][0]] and 
    # self.faces[face_idx[*][1]] will share an edge
    face_adjacency = edge_face_index[edge_groups]
    if return_edges:
        face_adjacency_edges = edges[edge_groups[:, 0]]
        return face_adjacency, face_adjacency_edges
    return face_adjacency


def adjacency_angle(mesh, angle, direction=np.less, return_edges=False):
    """
    return the adjacent faces of a mesh only if the faces are at less than a specified angle.
    :param mesh: trimesh
    :param angle: float, radians; faces at angles LARGER than this will be considered NOT adjacenct
    :param direction: function, used to test face angle against angle kwarg by default set to np.less
    :param return_edges: bool, return edges affiliated with adjacency or not
    :return: adjacency: (n,2) int list of face indices in mesh
    if return_edges:
        edges: (n,2) int list of vertex indices in mesh (edges)
    """
    # use the cached adjacency if possible (n,2)
    adjacency = mesh.face_adjacency
    # normal vectors for adjacent faces (n, 2, 3)
    normals = mesh.face_normals[adjacency]
    # dot products of normals (n)
    dots = diagonal_dot(normals[:, 0], normals[:, 1])
    # clip for floating point error
    dots = np.clip(dots, -1.0, 1.0)
    adj_ok = direction(np.abs(np.arccos(dots)), angle)
    # result is (m,2)
    new_adjacency = adjacency[adj_ok]
    if return_edges:
        edges = mesh.face_adjacency_edges[adj_ok]
        return new_adjacency, edges
    return new_adjacency


def shared_edges(faces_a, faces_b):
    '''
    Given two sets of faces, find the edges which are in both sets.

    Arguments
    ---------
    faces_a: (n,3) int, set of faces
    faces_b: (m,3) int, set of faces

    Returns
    ---------
    shared: (p, 2) int, set of edges
    '''
    e_a = np.sort(faces_to_edges(faces_a), axis=1)
    e_b = np.sort(faces_to_edges(faces_b), axis=1)
    shared = boolean_rows(e_a, e_b, operation=set.intersection)
    return shared


def connected_edges(G, nodes):
    '''
    Given graph G and list of nodes, return the list of edges that 
    are connected to nodes
    '''
    nodes_in_G = deque()
    for node in nodes:
        if not G.has_node(node): continue
        nodes_in_G.extend(nx.node_connected_component(G, node))
    edges = G.subgraph(nodes_in_G).edges()
    return edges


def facets(mesh):
    """
    Find the list of coplanar and adjacent faces.
    :param mesh: Trimesh
    :return: nx[face indices list]
    """
    def facets_nx():
        graph_parallel = nx.from_edgelist(face_idx[parallel])
        facets_idx = np.array([list(i) for i in nx.connected_components(graph_parallel)])
        #### commented by weiwei
        # should also return the single triangles
        facets_idx_extra = copy.deepcopy(facets_idx.tolist())
        for item in range(mesh.faces.shape[0]):
            if item not in [i for subitem in facets_idx.tolist() for i in subitem]:
                facets_idx_extra.append([item])
        return np.array(facets_idx_extra)
        # return facets_idx

    def facets_gt():
        graph_parallel = GTGraph()
        graph_parallel.add_edge_list(face_idx[parallel])
        connected = label_components(graph_parallel, directed=False)[0].a
        facets_idx = group(connected, min_len=2)
        return facets_idx

    # (n,2) list of adjacent face indices
    face_idx = mesh.face_adjacency
    # test adjacent faces for angle
    normal_pairs = mesh.face_normals[tuple([face_idx])]
    normal_dot = (np.sum(normal_pairs[:, 0, :] * normal_pairs[:, 1, :], axis=1) - 1) ** 2
    # if normals are actually equal, they are parallel with a high degree of confidence
    parallel = normal_dot < tol.zero
    non_parallel = np.logical_not(parallel)
    # saying that two faces are *not* parallel is susceptible to error
    # so we add a major_radius check which computes the linear_distance between face
    # centroids and divides it by the dot product of the normals
    # this means that small angles between big faces will have a large
    # major_radius which we can filter out easily.
    # if you don't do this, floating point error on tiny faces can push
    # the normals past a pure angle threshold even though the actual 
    # deviation across the face is extremely small. 
    center = mesh.triangles.mean(axis=1)
    center_sq = np.sum(np.diff(center[face_idx],
                               axis=1).reshape((-1, 3)) ** 2, axis=1)
    radius_sq = center_sq[non_parallel] / normal_dot[non_parallel]
    parallel[non_parallel] = radius_sq > tol.facet_rsq
    #### commented by weiwei, always use graph-nx
    # graph-tool is ~6x faster than networkx but is more difficult to install
    # if _has_gt: return facets_gt()
    # else:       return facets_nx()
    return facets_nx()


def facets_over_segmentation(mesh, faceangle=.9, segangle=.9):
    """
    compute the clusters using mesh oversegmentation

    :param mesh: Trimesh object
    :param faceangle: the angle between two adjacent faces that are taken as coplanar
    :param segangle: the angle between two adjacent segmentations that are taken as coplanar
    :return: the same as facets

    author: weiwei
    date: 20161116cancun, 20210119osaka
    """

    def __expand_adj(mesh, face_id, reference_normal, adjacent_face_list, face_angle=.9):
        """
        find the adjacency of a face
        the normal of the newly added face should be coherent with the reference normal
        this is an iterative function
        :param: mesh: Trimesh object
        :param: face_id: Index of the face to expand
        :param: reference_normal: The normal of the reference face
        :param: adjacent_face_list: The adjacent list of a face
        :param: face_angle: the angle of adjacent faces that are taken as coplanar
        :return: None or a list of face_ids
        author: weiwei
        date: 20161213, 20210119
        """
        curvature = 0
        face_center = np.mean(mesh.vertices[mesh.faces[face_id]], axis=1)
        bool_list0 = np.asarray(adjacent_face_list[:, 0] == face_id)
        bool_list1 = np.asarray(adjacent_face_list[:, 1] == face_id)
        xadjid0 = adjacent_face_list[bool_list1][:, 0]
        xadjid1 = adjacent_face_list[bool_list0][:, 1]
        xadjid = np.append(xadjid0, xadjid1)
        returnlist = []
        faceadjx = adjacent_face_list[np.logical_not(np.logical_or(bool_list0, bool_list1))]
        for j in xadjid:
            newnormal = mesh.face_normals[j]
            dotnorm = np.dot(reference_normal, newnormal)
            if dotnorm > face_angle:
                newfacecenter = np.mean(mesh.vertices[mesh.faces[j]], axis=1)
                d = np.linalg.norm(newfacecenter - face_center)
                if dotnorm > 1.0:
                    dotnorm = 1.0
                tmp_curvature = math.acos(dotnorm) / d
                if tmp_curvature > curvature:
                    curvature = tmp_curvature
                returnlist.append(j)
        finalreturnlist = [face_id]
        while returnlist:
            finalreturnlist.extend(returnlist)
            finalreturnlist = list(set(finalreturnlist))
            newreturnlist = []
            for id, j in enumerate(returnlist):
                bool_list0 = np.asarray(faceadjx[:, 0] == j)
                bool_list1 = np.asarray(faceadjx[:, 1] == j)
                xadjid0 = faceadjx[bool_list1][:, 0]
                xadjid1 = faceadjx[bool_list0][:, 1]
                xadjid = np.append(xadjid0, xadjid1)
                faceadjx = faceadjx[np.logical_not(np.logical_or(bool_list0, bool_list1))]
                for k in xadjid:
                    newnormal = mesh.face_normals[k]
                    dotnorm = np.dot(reference_normal, newnormal)
                    if dotnorm > face_angle:
                        newfacecenter = np.mean(mesh.vertices[mesh.faces[j]], axis=1)
                        d = np.linalg.norm(newfacecenter - face_center)
                        if dotnorm > 1.0:
                            dotnorm = 1.0
                        tmp_curvature = math.acos(dotnorm) / d
                        if tmp_curvature > curvature:
                            curvature = tmp_curvature
                        newreturnlist.append(k)
            returnlist = list(set(newreturnlist))
        return finalreturnlist, curvature

    # plot using panda3d
    # import trimesh.visual as visual

    # from panda3d.core import GeomNode, NodePath, Vec4
    # import pandaplotutils.pandactrl as pandactrl
    # import pandaplotutils.pandageom as pandageom
    # base = pandactrl.World(camp=[700, 300, 700], lookatp=[0, 0, 0])

    # the approach using random start
    # faceadj  = mesh.face_adjacency
    # removelist = []
    # faceids = list(range(len(mesh.faces)))
    # random.shuffle(faceids)
    # for i in faceids:
    #     if i not in removelist:
    #         print i, len(mesh.faces)
    #         rndcolor = visual.color_to_float(visual.random_color())
    #         adjidlist = __expand_adj(mesh, i, mesh.face_normals[i], faceadj)
    #         removelist.extend(adjidlist)
    #         for j in adjidlist:
    #             vertxid = mesh.faces[j, :]
    #             vert0 = mesh.vertices[vertxid[0]]
    #             vert1 = mesh.vertices[vertxid[1]]
    #             vert2 = mesh.vertices[vertxid[2]]
    #             vertices = np.array([vert0, vert1, vert2])
    #             normals = mesh.face_normals[j].reshape(1,3)
    #             triangles = np.array([[0, 1, 2]])
    #             geom = pandageom.packpandageom(vertices, normals, triangles)
    #             node = GeomNode('piece')
    #             node.addGeom(geom)
    #             star = NodePath('piece')
    #             star.attachNewNode(node)
    #             star.setColor(Vec4(rndcolor[0], rndcolor[1], rndcolor[2], rndcolor[3]))
    #             star.setTwoSided(True)
    #             star.reparentTo(base.render)
    # base.run()

    # the approach using large normal difference
    faceadj = mesh.face_adjacency
    faceids = list(range(len(mesh.faces)))
    random.shuffle(faceids)
    knownfacetnormals = np.array([])
    knownfacets = []
    knowncurvature = []
    for i in faceids:
        if knownfacetnormals.size:
            potentialfacetsidx = np.where(np.dot(knownfacetnormals, mesh.face_normals[i]) > segangle)[0]
            # for parallel faces
            potentialfaceids = []
            if potentialfacetsidx.size:
                for pfi in potentialfacetsidx:
                    potentialfaceids.extend(knownfacets[pfi])
                if i in potentialfaceids:
                    continue
        # rndcolor = visual.color_to_float(visual.random_color())
        adjidlist, curvature = __expand_adj(mesh, i, mesh.face_normals[i], faceadj, faceangle)
        facetnormal = np.sum(mesh.face_normals[adjidlist], axis=0)
        facetnormal = facetnormal / np.linalg.norm(facetnormal)
        if knownfacetnormals.size:
            knownfacetnormals = np.vstack((knownfacetnormals, facetnormal))
        else:
            knownfacetnormals = np.hstack((knownfacetnormals, facetnormal))
        knownfacets.append(adjidlist)
        knowncurvature.append(curvature)
    return [np.array(knownfacets), np.array(knownfacetnormals), np.array(knowncurvature)]

    # plot using panda3d
    #     for j in adjidlist:
    #         vertxid = mesh.faces[j, :]
    #         vert0 = mesh.vertices[vertxid[0]]+.012*i*facet_normal
    #         vert1 = mesh.vertices[vertxid[1]]+.012*i*facet_normal
    #         vert2 = mesh.vertices[vertxid[2]]+.012*i*facet_normal
    #         vertices = np.array([vert0, vert1, vert2])
    #         normals = mesh.face_normals[j].reshape(1,3)
    #         triangles = np.array([[0, 1, 2]])
    #         geom = pandageom.packpandageom(vertices, normals, triangles)
    #         node = GeomNode('piece')
    #         node.addGeom(geom)
    #         star = NodePath('piece')
    #         star.attachNewNode(node)
    #         star.setColor(Vec4(rndcolor[0], rndcolor[1], rndcolor[2], rndcolor[3]))
    #         star.setTwoSided(True)
    #         star.reparentTo(base.render)
    # base.run()

    # for i in range(1122,1123):
    #     rndcolor = visual.color_to_float(visual.DEFAULT_COLOR)
    #     vertxid = mesh.faces[i, :]
    #     vert0 = mesh.vertices[vertxid[0]]
    #     vert1 = mesh.vertices[vertxid[1]]
    #     vert2 = mesh.vertices[vertxid[2]]
    #     vertices = [[vert0, vert1, vert2]]
    #     tri = Poly3DCollection(vertices)
    #     tri.set_color([rndcolor])
    #     ax.add_collection3d(tri)
    # plt.show()


def facets_noover(mesh, faceangle=.9):
    """
    compute the clusters using mesh segmentation
    "noover" means overlap is not considered

    :param mesh: Trimesh object
    :param faceangle: the angle between two adjacent faces that are taken as coplanar
    :return: the same as facets

    author: weiwei
    date: 20161116, cancun
    """

    def __expand_adj(mesh, faceid, refnormal, faceadj, faceangle, knownfacets):
        """
        find the adjacency of a face
        the normal of the newly added face should be coherent with the reference normal
        this is an iterative function

        :param: mesh: Trimesh object
        :param: faceid: Index of the face to expand
        :param: refnormal: The normal of the reference face
        :param: faceadj: The adjacent list of a face
        :param: faceangle: the angle of adjacent faces that are taken as coplanar
        :return: None or a list of faceids

        author: weiwei
        date: 20161213, update faceangle
        """

        knownidlist = [i for facet in knownfacets for i in facet]

        curvature = 0
        facecenter = np.mean(mesh.vertices[mesh.faces[faceid]], axis=1)

        boollist0 = np.asarray(faceadj[:, 0] == faceid)
        boollist1 = np.asarray(faceadj[:, 1] == faceid)
        xadjid0 = faceadj[boollist1][:, 0]
        xadjid1 = faceadj[boollist0][:, 1]
        xadjid = np.append(xadjid0, xadjid1)
        returnlist = []
        faceadjx = faceadj[np.logical_not(np.logical_or(boollist0, boollist1))]
        for j in xadjid:
            if j in knownidlist:
                continue
            newnormal = mesh.face_normals[j]
            dotnorm = np.dot(refnormal, newnormal)
            if dotnorm > faceangle:
                newfacecenter = np.mean(mesh.vertices[mesh.faces[j]], axis=1)
                d = np.linalg.norm(newfacecenter - facecenter)
                if dotnorm > 1.0:
                    dotnorm = 1.0
                tempcurvature = math.acos(dotnorm) / d
                if tempcurvature > curvature:
                    curvature = tempcurvature
                returnlist.append(j)
        finalreturnlist = [faceid]
        while returnlist:
            finalreturnlist.extend(returnlist)
            finalreturnlist = list(set(finalreturnlist))
            newreturnlist = []
            for id, j in enumerate(returnlist):
                if j in knownidlist:
                    continue
                boollist0 = np.asarray(faceadjx[:, 0] == j)
                boollist1 = np.asarray(faceadjx[:, 1] == j)
                xadjid0 = faceadjx[boollist1][:, 0]
                xadjid1 = faceadjx[boollist0][:, 1]
                xadjid = np.append(xadjid0, xadjid1)
                faceadjx = faceadjx[np.logical_not(np.logical_or(boollist0, boollist1))]
                for k in xadjid:
                    newnormal = mesh.face_normals[k]
                    dotnorm = np.dot(refnormal, newnormal)
                    if dotnorm > faceangle:
                        newfacecenter = np.mean(mesh.vertices[mesh.faces[j]], axis=1)
                        d = np.linalg.norm(newfacecenter - facecenter)
                        if dotnorm > 1.0:
                            dotnorm = 1.0
                        tempcurvature = math.acos(dotnorm) / d
                        if tempcurvature > curvature:
                            curvature = tempcurvature
                        newreturnlist.append(k)
            returnlist = list(set(newreturnlist))
        return finalreturnlist, curvature

    # the approach using large normal difference
    knownfacetnormals = np.array([])
    knownfacets = []
    knowncurvature = []
    faceadj = mesh.face_adjacency
    faceids = list(range(len(mesh.faces)))
    while True:
        random.shuffle(faceids)
        i = faceids[0]
        adjidlist, curvature = __expand_adj(mesh, i, mesh.face_normals[i], faceadj, faceangle, knownfacets)
        facetnormal = np.sum(mesh.face_normals[adjidlist], axis=0)
        facetnormal = facetnormal / np.linalg.norm(facetnormal)
        if knownfacetnormals.size:
            knownfacetnormals = np.vstack((knownfacetnormals, facetnormal))
        else:
            knownfacetnormals = np.hstack((knownfacetnormals, facetnormal))
        knownfacets.append(adjidlist)
        knowncurvature.append(curvature)
        faceids = list(set(faceids) - set(adjidlist))
        if len(faceids) == 0:
            break
    return [np.array(knownfacets), np.array(knownfacetnormals), np.array(knowncurvature)]


def split(mesh, only_watertight=True, adjacency=None):
    '''
    Given a mesh, will split it up into a list of meshes based on face connectivity
    If only_watertight is true, it will only return meshes where each face has
    exactly 3 adjacent faces.

    Arguments
    ----------
    mesh: Trimesh 
    only_watertight: if True, only return watertight components
    adjacency: (n,2) list of face adjacency to override using the plain
               adjacency calculated automatically. 

    Returns
    ----------
    meshes: list of Trimesh objects
    '''

    def split_nx():
        adjacency_graph = nx.from_edgelist(adjacency)
        components = nx.connected_components(adjacency_graph)
        result = mesh.submesh(components, only_watertight=only_watertight)
        return result

    def split_gt():
        g = GTGraph()
        g.add_edge_list(adjacency)
        component_labels = label_components(g, directed=False)[0].a
        components = group(component_labels)
        result = mesh.submesh(components, only_watertight=only_watertight)
        return result

    if adjacency is None:
        adjacency = mesh.face_adjacency

    if _has_gt:
        return split_gt()
    else:
        return split_nx()


def smoothed(mesh, angle):
    '''
    Return a non- watertight version of the mesh which will
    render nicely with smooth shading. 

    Arguments
    ---------
    mesh:  Trimesh object
    angle: float, angle in radians, adjacent faces which have normals
           below this angle will be smoothed.

    Returns
    ---------
    smooth: Trimesh object
    '''
    adjacency = adjacency_angle(mesh, angle)
    graph = nx.from_edgelist(adjacency)
    graph.add_nodes_from(np.arange(len(mesh.faces)))
    smooth = mesh.submesh(nx.connected_components(graph),
                          only_watertight=False,
                          append=True)
    return smooth


def is_watertight(edges, return_winding=False):
    '''
    Arguments
    ---------
    edges: (n,2) int, set of vertex indices
    
    Returns
    ---------
    watertight: boolean, whether every edge is contained by two faces
    '''
    edges_sorted = np.sort(edges, axis=1)
    groups = group_rows(edges_sorted, require_count=2)
    watertight = (len(groups) * 2) == len(edges)
    if return_winding:
        opposing = edges[groups].reshape((-1, 4))[:, 1:3].T
        reversed = np.equal(*opposing).all()
        return watertight, reversed
    return watertight
