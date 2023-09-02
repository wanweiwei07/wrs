#!/usr/bin/env python3

from panda3d.core import *
from panda3d.bullet import (BulletWorld,
                            BulletPlaneShape,
                            BulletTriangleMesh,
                            BulletRigidBodyNode,
                            BulletTriangleMeshShape)


def rayHit(pfrom, pto, geom):
    """
    NOTE: this function is quite slow
    find the nearest collision point
    between vec (pto-pfrom) and the mesh of nodepath

    :param pfrom: starting point of the ray, Point3
    :param pto: ending point of the ray, Point3
    :param geom: meshmodel, a panda3d datatype
    :return: None or Point3

    author: weiwei
    date: 20161201
    """

    bulletworld = BulletWorld()
    facetmesh = BulletTriangleMesh()
    facetmesh.addGeom(geom)
    facetmeshnode = BulletRigidBodyNode('facet')
    bullettmshape = BulletTriangleMeshShape(facetmesh, dynamic=True)
    bullettmshape.setMargin(0)
    facetmeshnode.addShape(bullettmshape)
    bulletworld.attachRigidBody(facetmeshnode)
    result = bulletworld.rayTestClosest(pfrom, pto)

    if result.hasHit():
        return result.getHitPos()
    else:
        return None


# def genCollisionMeshNp(nodepath, basenodepath=None, name='autogen'):
#     """
#     generate the collision mesh of a nodepath using nodepath
#     this function suppose the nodepath is a single model with one geomnode

#     :param nodepath: the panda3d nodepath of the object
#     :param basenodepath: the nodepath to compute relative transform,
#                          identity if none
#     :param name: the name of the rigidbody
#     :return: bulletrigidbody

#     author: weiwei
#     date: 20161212, tsukuba
#     """

#     geomnodepath = nodepath.find("**/+GeomNode")
#     geombullnode = BulletRigidBodyNode(name)
#     geom = geomnodepath.node().getGeom(0)
#     geomtf = nodepath.getTransform(base.render)
#     if basenodepath is not None:
#         geomtf = nodepath.getTransform(basenodepath)
#     geombullmesh = BulletTriangleMesh()
#     geombullmesh.addGeom(geom)
#     bullettmshape = BulletTriangleMeshShape(geombullmesh, dynamic=True)
#     bullettmshape.setMargin(0)
#     geombullnode.addShape(bullettmshape, geomtf)
#     return geombullnode


def genCollisionMeshMultiNp(base, nodepath, basenodepath=None, name='autogen'):
    """
    generate the collision mesh of a nodepath using nodepath
    this function suppose the nodepath has multiple models with many geomnodes

    use genCollisionMeshMultiNp instead of genCollisionMeshNp for generality

    :param nodepath: the panda3d nodepath of the object
    :param basenodepath: the nodepath to compute relative transform,
                         identity if none
    :param name: the name of the rigidbody
    :return: bulletrigidbody

    author: weiwei
    date: 20161212, tsukuba
    """

    gndcollection = nodepath.findAllMatches("**/+GeomNode")
    geombullnode = BulletRigidBodyNode(name)
    for gnd in gndcollection:
        geom = gnd.node().getGeom(0)
        geomtf = gnd.getTransform(base.render)
        if basenodepath is not None:
            geomtf = gnd.getTransform(basenodepath)
        geombullmesh = BulletTriangleMesh()
        geombullmesh.addGeom(geom)
        bullettmshape = BulletTriangleMeshShape(geombullmesh, dynamic=True)
        bullettmshape.setMargin(0)
        geombullnode.addShape(bullettmshape, geomtf)
    return geombullnode


def genCollisionMeshGeom(geom, name='autogen'):
    """
    generate the collision mesh of a nodepath using geom

    :param geom: the panda3d geom of the object
    :param basenodepath: the nodepath to compute relative transform
    :return: bulletrigidbody

    author: weiwei
    date: 20161212, tsukuba
    """

    geomtf = TransformState.makeIdentity()
    geombullmesh = BulletTriangleMesh()
    geombullmesh.addGeom(geom)
    geombullnode = BulletRigidBodyNode(name)
    bullettmshape = BulletTriangleMeshShape(geombullmesh, dynamic=True)
    bullettmshape.setMargin(0)
    geombullnode.addShape(bullettmshape, geomtf)
    return geombullnode


def genCollisionPlane(updirection=Vec3(0, 0, 1), offset=0, name='autogen'):
    """
    generate a plane bulletrigidbody node

    :param updirection: the normal parameter of bulletplaneshape at panda3d
    :param offset: the d parameter of bulletplaneshape at panda3d
    :param name:
    :return: bulletrigidbody

    author: weiwei
    date: 20170202, tsukuba
    """

    bulletplnode = BulletRigidBodyNode(name)
    bulletplshape = BulletPlaneShape(Vec3(0, 0, 1), offset)
    bulletplshape.setMargin(0)
    bulletplnode.addShape(bulletplshape)
    return bulletplnode
