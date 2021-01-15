from panda3d.bullet import BulletWorld, BulletRigidBodyNode, BulletPlaneShape, BulletBoxShape
from panda3d.bullet import BulletTriangleMeshShape, BulletTriangleMesh
from panda3d.bullet import BulletConvexHullShape
from panda3d.core import TransformState, Vec3, CollisionBox
import copy
import numpy as np
import basis.robotmath as rm
import basis.dataadapter as da
import basis.trimeshgenerator as tg

# box collision model
def is_box2box_collided(objcm_list0, objcm_list1):
    """
    check if two objects objcm0 as objcm1 are in collision with each other
    the two objects are in the form of collision model
    the AABB boxlist will be used
    type "box" is required
    :param objcm_list0: the first object or list
    :param objcm_list1: the second object or list
    :return: boolean value showing if the two objects are in collision
    author: weiwei
    date: 20190313
    """
    objcm0boxbullnode = _gen_box_cdmesh(objcm_list0)
    objcm1boxbullnode = _gen_box_cdmesh(objcm_list1)
    result = base.physicsworld.contactTestPair(objcm0boxbullnode, objcm1boxbullnode)
    return True if result.getNumContacts() else False


# 2 box - triangles
def is_box2triangles_collided(objcm_list0, objcm_list1):
    """
    check if two objects objcm0 as objcm1 are in collision with each other
    the two objects are in the form of collision model
    the AABB boxlist will be used
    type "box" is required
    :param objcm_list0: the first object
    :param objcm_list1: the second object
    :return: boolean value showing if the two objects are in collision
    author: weiwei
    date: 201903138
    """
    objcm0boxbullnode = _gen_box_cdmesh(objcm_list0)
    objcm1bullnode = _gen_triangles_cdmesh(objcm_list1)
    result = base.physicsworld.contactTestPair(objcm0boxbullnode, objcm1bullnode)
    return True if result.getNumContacts() else False


def show_box_cdmesh(objcm_list):
    """
    show the AABB collision meshes of the given objects
    :param objcm_list
    author: weiwei
    date: 20190313
    :return:
    """
    if not base.toggledebug:
        print("Toggling on base.physicsworld debug mode...")
        base.change_debugstatus(True)
    objcmboxbullnode = _gen_box_cdmesh(objcm_list)
    base.physicsworld.attach(objcmboxbullnode)
    base.physicsbodylist.append(objcmboxbullnode)
    return objcmboxbullnode


# triangles collision model
def is_triangles2triangles_collided(objcm_list0, objcm_list1):
    """
    check if two objects objcm0 and objcm1 are in collision with each other
    the two objects are in the form of collision model
    the bulletmeshes will be used
    :param objcm_list0: the first object
    :param objcm_list1: the second object
    :return: boolean value showing if the two objects are in collision
    author: weiwei
    date: 20190313
    """
    objcm0bullnode = _gen_triangles_cdmesh(objcm_list0)
    objcm1bullnode = _gen_triangles_cdmesh(objcm_list1)
    result = base.physicsworld.contactTestPair(objcm0bullnode, objcm1bullnode)
    if result.getNumContacts():
        import modeling.geometricmodel as gm
        for contact in result.getContacts():
            gm.gen_sphere(pos = da.pdv3_to_npv3(contact.getManifoldPoint().getPositionWorldOnA()), radius=0.001).attach_to(base)
            gm.gen_sphere(pos = da.pdv3_to_npv3(contact.getManifoldPoint().getPositionWorldOnB()), radius=0.001).attach_to(base)
    return True if result.getNumContacts() else False


def rayhit_triangles_closet(pfrom, pto, objcm):
    """
    :param pfrom:
    :param pto:
    :param objcm:
    :return:
    author: weiwei
    date: 20190805
    """
    tmptrimesh = objcm.trimesh.copy()
    tmptrimesh.apply_transform(objcm.get_homomat())
    geom = da.pandageom_from_vfnf(tmptrimesh.vertices, tmptrimesh.face_normals, tmptrimesh.faces)
    targetobjmesh = BulletTriangleMesh()
    targetobjmesh.addGeom(geom)
    bullettmshape = BulletTriangleMeshShape(targetobjmesh, dynamic=True)
    bullettmshape.setMargin(1e-6)
    targetobjmeshnode = BulletRigidBodyNode('facet')
    targetobjmeshnode.addShape(bullettmshape)
    base.physicsworld.attach(targetobjmeshnode)
    result = base.physicsworld.rayTestClosest(da.npv3_to_pdv3(pfrom), da.npv3_to_pdv3(pto))
    base.physicsworld.removeRigidBody(targetobjmeshnode)
    if result.hasHit():
        return [da.pdv3_to_npv3(result.getHitPos()), da.pdv3_to_npv3(result.getHitNormal())]
    else:
        return [None, None]


def rayhit_triangles_all(pfrom, pto, objcm):
    """
    :param pfrom:
    :param pto:
    :param objcm:
    :return:
    author: weiwei
    date: 20190805
    """
    tmptrimesh = objcm.trimesh.copy()
    tmptrimesh.apply_transform(objcm.gethomomat())
    geom = da.pandageom_from_vfnf(tmptrimesh.vertices, tmptrimesh.face_normals, tmptrimesh.faces)
    targetobjmesh = BulletTriangleMesh()
    targetobjmesh.addGeom(geom)
    bullettmshape = BulletTriangleMeshShape(targetobjmesh, dynamic=True)
    bullettmshape.setMargin(1e-6)
    targetobjmeshnode = BulletRigidBodyNode('facet')
    targetobjmeshnode.addShape(bullettmshape)
    base.physicsworld.attach(targetobjmeshnode)
    result = base.physicsworld.rayTestAll(da.npv3_to_pdv3(pfrom), da.npv3_to_pdv3(pto))
    base.physicsworld.removeRigidBody(targetobjmeshnode)
    if result.hasHits():
        allhits = []
        for hit in result.getHits():
            allhits.append([da.pdv3_to_npv3(hit.getHitPos()), da.pdv3_to_npv3(-hit.getHitNormal())])
        return allhits
    else:
        return []


def show_triangles_cdmesh(objcm_list):
    """
    show the collision meshes of the given objects
    :param objcm_list environment.collisionmodel
    :return:
    author: weiwei
    date: 20190313
    """
    if not base.toggledebug:
        print("Toggling on base.physicsworld debug mode...")
        base.change_debugstatus(True)
    objcmmeshbullnode = _gen_triangles_cdmesh(objcm_list)
    base.physicsworld.attach(objcmmeshbullnode)
    base.physicsbodylist.append(objcmmeshbullnode)
    return objcmmeshbullnode

# convexhull collision model
def is_convexhull2triangles_collided(objcm_list0, objcm_list1):
    """
    check if two objects objcm0 and objcm1 are in collision with each other
    the two objects are in the form of collision model
    the bulletmeshes will be used
    :param objcm_list0: the first object
    :param objcm_list1: the second object
    :return: boolean value showing if the two objects are in collision
    author: weiwei
    date: 20190313
    """
    objcm0bullnode = _gen_convexhull_cdmesh(objcm_list0)
    objcm1bullnode = _gen_triangles_cdmesh(objcm_list1)
    result = base.physicsworld.contactTestPair(objcm0bullnode, objcm1bullnode)
    return True if result.getNumContacts() else False

def is_convexhull2convexhull_collided(objcm_list0, objcm_list1):
    """
    check if two objects objcm0 and objcm1 are in collision with each other
    the two objects are in the form of collision model
    the bulletmeshes will be used
    :param objcm_list0: the first object
    :param objcm_list1: the second object
    :return: boolean value showing if the two objects are in collision
    author: weiwei
    date: 20190313
    """
    objcm0bullnode = _gen_convexhull_cdmesh(objcm_list0)
    objcm1bullnode = _gen_convexhull_cdmesh(objcm_list1)
    result = base.physicsworld.contactTestPair(objcm0bullnode, objcm1bullnode)
    return True if result.getNumContacts() else False


def show_convexhull_cdmesh(objcm_list):
    """
    show the collision meshes of the given objects
    :param objcm_list environment.collisionmodel
    :return:
    author: weiwei
    date: 20190313
    """
    if not base.toggledebug:
        print("Toggling on base.physicsworld debug mode...")
        base.change_debugstatus(True)
    objcmmeshbullnode = _gen_convexhull_cdmesh(objcm_list)
    base.physicsworld.attach(objcmmeshbullnode)
    base.physicsbodylist.append(objcmmeshbullnode)
    return objcmmeshbullnode


def unshow_all():
    """
    unshow everything
    author: weiwei
    date: 20180621
    :return:
    """
    print(base.physicsbodylist)
    for physicsbody in base.physicsbodylist:
        base.physicsworld.remove(physicsbody)
    base.physicsbodylist = []


def unshow(cmbullnode):
    base.physicsworld.remove(cmbullnode)


# util functions
def _gen_box_cdmesh(objcm_list, name='autogen'):
    """
    generate a bullet cd obj using the AABB boundary of a obstacle collision model
    :param objcm_list: a collision model or a list of collision model
    :return: bulletrigidbody
    author: weiwei
    date: 20190313, toyonaka
    """
    if not isinstance(objcm_list, list):
        objcm_list = [objcm_list]
    geombullnode = BulletRigidBodyNode(name)
    for objcm in objcm_list:
        bottom_left, top_right = objcm.pdnp_raw.getTightBounds()
        bound_tf = np.eye(4)
        bound_tf[:3,3] = da.pdv3_to_npv3((bottom_left+top_right)/2)
        extent = da.pdv3_to_npv3(top_right-bottom_left)
        objtrm_box = tg.gen_box(extent, bound_tf)
        geom = da.pandageom_from_vvnf(objtrm_box.vertices, objtrm_box.vertex_normals, objtrm_box.faces)
        geom.transformVertices(objcm.pdnp.getMat())
        geombullmesh = BulletTriangleMesh()
        geombullmesh.addGeom(geom)
        bullet_triangles_shape = BulletTriangleMeshShape(geombullmesh, dynamic=True)
        bullet_triangles_shape.setMargin(0)
        geombullnode.addShape(bullet_triangles_shape)
    return geombullnode

def _gen_triangles_cdmesh(objcm_list, name='autogen'):
    """
    generate the collision mesh of a nodepath using nodepath
    this function suppose the nodepath has multiple models with many geomnodes
    use genCollisionMeshMultiNp instead of genCollisionMeshNp for generality
    :param nodepath: the panda3d nodepath of the object
    :param basenodepath: the nodepath to compute relative transform, identity if none
    :param name: the name of the rigidbody
    :return: bulletrigidbody
    author: weiwei
    date: 20161212, tsukuba
    """
    if not isinstance(objcm_list, list):
        objcm_list = [objcm_list]
    geombullnode = BulletRigidBodyNode(name)
    for objcm in objcm_list:
        gndcollection = objcm.pdnp_raw.findAllMatches("+GeomNode")
        for gnd in gndcollection:
            geom = copy.deepcopy(gnd.node().getGeom(0))
            geom.transformVertices(objcm.pdnp.getMat())
            geombullmesh = BulletTriangleMesh()
            geombullmesh.addGeom(geom)
            bullettmshape = BulletTriangleMeshShape(geombullmesh, dynamic=True)
            bullettmshape.setMargin(0)
            geombullnode.addShape(bullettmshape)
    return geombullnode

def _gen_convexhull_cdmesh(objcm_list, name='autogen'):
    """
    generate the collision mesh of a nodepath using nodepath
    this function suppose the nodepath has multiple models with many geomnodes
    use genCollisionMeshMultiNp instead of genCollisionMeshNp for generality
    :param nodepath: the panda3d nodepath of the object
    :param basenodepath: the nodepath to compute relative transform, identity if none
    :param name: the name of the rigidbody
    :return: bulletrigidbody
    author: weiwei
    date: 20161212, tsukuba
    """
    if not isinstance(objcm_list, list):
        objcm_list = [objcm_list]
    geombullnode = BulletRigidBodyNode(name)
    for objcm in objcm_list:
        objtrm_cvx = objcm.trimesh.convex_hull
        geom = da.pandageom_from_vvnf(objtrm_cvx.vertices, objtrm_cvx.vertex_normals, objtrm_cvx.faces)
        geom.transformVertices(objcm.pdnp.getMat())
        geombullmesh = BulletTriangleMesh()
        geombullmesh.addGeom(geom)
        bullettmshape = BulletTriangleMeshShape(geombullmesh, dynamic=True)
        bullettmshape.setMargin(0)
        geombullnode.addShape(bullettmshape)
    return geombullnode

def _gen_triangles_cdmesh_from_geom(geom, name='autogen'):
    """
    generate the collision mesh of a nodepath using geom
    :param geom: the panda3d geom of the object
    :param basenodepath: the nodepath to compute relative transform
    :return: bulletrigidbody
    author: weiwei
    date: 20161212, tsukuba
    """
    geombullmesh = BulletTriangleMesh()
    geombullmesh.addGeom(geom)
    geombullnode = BulletRigidBodyNode(name)
    bullettmshape = BulletTriangleMeshShape(geombullmesh, dynamic=True)
    bullettmshape.setMargin(0)
    geombullnode.addShape(bullettmshape)
    return geombullnode


def _gen_plane_cdmesh(updirection=np.array([0, 0, 1]), offset=0, name='autogen'):
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
    bulletplshape = BulletPlaneShape(Vec3(updirection[0], updirection[1], updirection[2]), offset)
    bulletplshape.setMargin(0)
    bulletplnode.addShape(bulletplshape)
    return bulletplnode


def _rayhit_geom(pfrom, pto, geom):
    """
    TODO: To be deprecated, 20201119
    NOTE: this function is quite slow
    find the nearest collision point between vec(pto-pfrom) and the mesh of nodepath
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
    bullettmshape.setMargin(1e-6)
    facetmeshnode.addShape(bullettmshape)
    bulletworld.attach(facetmeshnode)
    result = bulletworld.rayTestClosest(pfrom, pto)
    return result.getHitPos() if result.hasHit() else None


if __name__ == '__main__':
    import os, math, basis
    import numpy as np
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm
    import modeling.collisionmodel as cm

    wd.World(campos=[1.0, 1, .0, 1.0], lookatpos=[0, 0, 0])
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    objcm = cm.CollisionModel(objpath)
    homomat = np.eye(4)
    homomat[:3, :3] = rm.rotmat_from_axangle([0, 0, 1], math.pi / 2)
    homomat[:3, 3] = np.array([0, 0.02, 0])
    objcm.set_homomat(homomat)
    pfrom = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    pto = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -0.9])
    hitpos, hitnrml = rayhit_triangles_closet(pfrom=pfrom, pto=pto, objcm=objcm)
    objcm.attach_to(base)
    objcm.show_cdmesh(type='box')
    objcm.show_cdmesh(type='convexhull')
    # gm.gen_sphere(hitpos, radius=.003, rgba=np.array([0, 1, 1, 1])).attach_to(base)
    # gm.gen_stick(spos=pfrom, epos=pto, thickness=.002).attach_to(base)
    # gm.gen_arrow(spos=hitpos, epos=hitpos + hitnrml * .07, thickness=.002, rgba=np.array([0, 1, 0, 1])).attach_to(base)
    base.run()
