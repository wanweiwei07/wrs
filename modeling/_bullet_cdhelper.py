from panda3d.bullet import BulletWorld, BulletRigidBodyNode, BulletPlaneShape, BulletBoxShape
from panda3d.bullet import BulletTriangleMeshShape, BulletTriangleMesh
from panda3d.bullet import BulletConvexHullShape
from panda3d.core import TransformState, Vec3, CollisionBox
import copy
import numpy as np
import basis.robotmath as rm
import basis.dataadapter as da
import basis.trimeshgenerator as tg

# util functions
def _gen_aabb_cdmesh(objcm_list, name='auto'):
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
        bottom_left, top_right = objcm.objpdnp_raw.getTightBounds()
        bound_tf = np.eye(4)
        bound_tf[:3,3] = da.pdv3_to_npv3((bottom_left+top_right)/2)
        extent = da.pdv3_to_npv3(top_right-bottom_left)
        objtrm_box = tg.gen_box(extent, bound_tf)
        geom = da.pandageom_from_vvnf(objtrm_box.vertices, objtrm_box.vertex_normals, objtrm_box.faces)
        geom.transformVertices(objcm.objpdnp.getMat())
        geombullmesh = BulletTriangleMesh()
        geombullmesh.addGeom(geom)
        bullet_triangles_shape = BulletTriangleMeshShape(geombullmesh, dynamic=True)
        bullet_triangles_shape.setMargin(0)
        geombullnode.addShape(bullet_triangles_shape)
    return geombullnode

def _gen_obb_cdmesh(objcm_list):
    """
    generate the a bullet cd obj using the OBB boundary of a objcm_list
    :param objcm_list: a collision model or a list of collision model
    :return: gimpact.TriMesh
    author: weiwei
    date: 20210116osaka-u
    """
    if not isinstance(objcm_list, list):
        objcm_list = [objcm_list]
    geombullnode = BulletRigidBodyNode(name, name='auto')
    for objcm in objcm_list:
        objtrm = objcm.objtrm.bounding_box_oriented
        geom = da.pandageom_from_vvnf(objtrm.vertices, objtrm.vertex_normals, objtrm.faces)
        geom.transformVertices(objcm.objpdnp.getMat())
        geombullmesh = BulletTriangleMesh()
        geombullmesh.addGeom(geom)
        bullettmshape = BulletTriangleMeshShape(geombullmesh, dynamic=True)
        bullettmshape.setMargin(0)
        geombullnode.addShape(bullettmshape)
    return geombullnode

def _gen_triangles_cdmesh(objcm_list, name='auto'):
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
        gndcollection = objcm.objpdnp_raw.findAllMatches("+GeomNode")
        for gnd in gndcollection:
            geom = copy.deepcopy(gnd.node().getGeom(0))
            geom.transformVertices(objcm.objpdnp.getMat())
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
        objtrm_cvx = objcm.objtrm.convex_hull
        geom = da.pandageom_from_vvnf(objtrm_cvx.vertices, objtrm_cvx.vertex_normals, objtrm_cvx.faces)
        geom.transformVertices(objcm.objpdnp.getMat())
        geombullmesh = BulletTriangleMesh()
        geombullmesh.addGeom(geom)
        bullettmshape = BulletTriangleMeshShape(geombullmesh, dynamic=True)
        bullettmshape.setMargin(0)
        geombullnode.addShape(bullettmshape)
    return geombullnode

def _cdmesh_from_objcm(objcm):
    """
    :param objcm: a collision model
    :return:
    """
    print(objcm.cdmesh_type)
    if objcm.cdmesh_type == 'aabb':
        gen_cdmesh_fn = _gen_aabb_cdmesh
    elif objcm.cdmesh_type == 'obb':
        gen_cdmesh_fn = _gen_obb_cdmesh
    elif objcm.cdmesh_type == 'convexhull':
        gen_cdmesh_fn = _gen_convexhull_cdmesh
    elif objcm.cdmesh_type == 'triangles':
        gen_cdmesh_fn = _gen_triangles_cdmesh
    return gen_cdmesh_fn(objcm)

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

def is_collided(objcm0, objcm1):
    """
    check if two objcm are collided after converting the specified cdmesh_type
    :param objcm0:
    :param objcm1:
    :return:
    author: weiwei
    date: 20210117
    """
    obj0 = _cdmesh_from_objcm(objcm0)
    obj1 = _cdmesh_from_objcm(objcm1)
    result = base.physicsworld.contactTestPair(obj0, obj1)
    contacts = result.getContacts()
    contact_points = [da.pdv3_to_npv3(ct.getManifoldPoint().getPositionWorldOnB()) for ct in contacts]
    return (True, contact_points) if len(contact_points)>0 else (False, contact_points)

def _show_aabb_cdmesh(objcm_list):
    """
    show the AABB collision meshes of the given objects
    :param objcm_list
    author: weiwei
    date: 20210116
    :return:
    """
    if not isinstance(objcm_list, list):
        objcm_list = [objcm_list]
    vertices = []
    faces = []
    for objcm in objcm_list:
        objtrm = objcm.objtrm.bounding_box
        homomat = objcm.get_homomat()
        vertices += rm.homomat_transform_points(homomat, objtrm.vertices).tolist()
        faces += (objtrm.faces+len(faces)).tolist()
    objwm = gm.WireFrameModel(tg.trm.Trimesh(np.array(vertices), np.array(faces)))
    objwm.attach_to(base)


def _show_obb_cdmesh(objcm_list):
    """
    show the OBB collision meshes of the given objects
    :param objcm_list
    author: weiwei
    date: 20210116
    :return:
    """
    if not isinstance(objcm_list, list):
        objcm_list = [objcm_list]
    vertices = []
    faces = []
    for objcm in objcm_list:
        objtrm = objcm.objtrm.bounding_box_oriented
        homomat = objcm.get_homomat()
        vertices += rm.homomat_transform_points(homomat, objtrm.vertices).tolist()
        faces += (objtrm.faces+len(faces)).tolist()
    objwm = gm.WireFrameModel(tg.trm.Trimesh(np.array(vertices), np.array(faces)))
    objwm.attach_to(base)


def _show_convexhull_cdmesh(objcm_list):
    """
    show the convex hull collision meshes of the given objects
    :param objcm_list environment.collisionmodel
    :return:
    author: weiwei
    date: 20210117
    """
    if not isinstance(objcm_list, list):
        objcm_list = [objcm_list]
    vertices = []
    faces = []
    for objcm in objcm_list:
        objtrm = objcm.objtrm.convex_hull
        homomat = objcm.get_homomat()
        vertices += rm.homomat_transform_points(homomat, objtrm.vertices).tolist()
        faces += (objtrm.faces+ len(faces)).tolist()
    objwm = gm.WireFrameModel(tg.trm.Trimesh(np.array(vertices), np.array(faces)))
    objwm.attach_to(base)

def _show_triangles_cdmesh(objcm_list):
    """
    show the collision meshes of the given objects
    :param objcm_list environment.collisionmodel
    :return:
    author: weiwei
    date: 20210116
    """
    if not isinstance(objcm_list, list):
        objcm_list = [objcm_list]
    vertices = []
    faces = []
    for objcm in objcm_list:
        homomat = objcm.get_homomat()
        vertices += rm.homomat_transform_points(homomat, objcm.objtrm.vertices).tolist()
        faces += (objcm.objtrm.faces + len(faces)).tolist()
    objwm = gm.WireFrameModel(tg.trm.Trimesh(np.array(vertices), np.array(faces)))
    objwm.attach_to(base)
    return objwm


def show_cdmesh(objcm):
    """
    :param objcm: a collision model
    :return:
    """
    if objcm.cdmesh_type == 'aabb':
        show_cdmesh_fn = _show_aabb_cdmesh
    elif objcm.cdmesh_type == 'obb':
        show_cdmesh_fn = _show_obb_cdmesh
    elif objcm.cdmesh_type == 'convexhull':
        show_cdmesh_fn = _show_convexhull_cdmesh
    elif objcm.cdmesh_type == 'triangles':
        show_cdmesh_fn = _show_triangles_cdmesh
    return show_cdmesh_fn(objcm)


def unshow_cdmsh(objwm):
    if not isinstance(objwm, gm.WireFrameModel):
        raise ValueError("The objwm must be a gm.WireFrameModel instance!")
    objwm.detach()

def rayhit_triangles_closet(pfrom, pto, objcm):
    """
    :param pfrom:
    :param pto:
    :param objcm:
    :return:
    author: weiwei
    date: 20190805
    """
    tmptrimesh = objcm.objtrm.copy()
    tmptrimesh.apply_transform(objcm.get_homomat())
    geom = da.pandageom_from_vfnf(tmptrimesh.vertices, tmptrimesh.face_normals, tmptrimesh.faces)
    targetobjmesh = BulletTriangleMesh()
    targetobjmesh.addGeom(geom)
    bullettmshape = BulletTriangleMeshShape(targetobjmesh, dynamic=True)
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
    tmptrimesh = objcm.objtrm.copy()
    tmptrimesh.apply_transform(objcm.gethomomat())
    geom = da.pandageom_from_vfnf(tmptrimesh.vertices, tmptrimesh.face_normals, tmptrimesh.faces)
    targetobjmesh = BulletTriangleMesh()
    targetobjmesh.addGeom(geom)
    bullettmshape = BulletTriangleMeshShape(targetobjmesh, dynamic=True)
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
    objpath = os.path.join(basis.__path__[0], 'objects', 'yumifinger.stl')
    objcm1= cm.CollisionModel(objpath, cdmesh_type='triangles')
    homomat = np.array([[-0.5       , -0.82363909,  0.2676166 , -0.00203699],
                        [-0.86602539,  0.47552824, -0.1545085 ,  0.01272306],
                        [ 0.        , -0.30901703, -0.95105648,  0.12604253],
                        [ 0.        ,  0.        ,  0.        ,  1.        ]])
    # homomat = np.array([[ 1.00000000e+00,  2.38935501e-16,  3.78436685e-17, -7.49999983e-03],
    #                     [ 2.38935501e-16, -9.51056600e-01, -3.09017003e-01,  2.04893537e-02],
    #                     [-3.78436685e-17,  3.09017003e-01, -9.51056600e-01,  1.22025304e-01],
    #                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    objcm1.set_homomat(homomat)
    objcm1.set_rgba([1,1,.3,.2])

    objpath = os.path.join(basis.__path__[0], 'objects', 'tubebig.stl')
    objcm2= cm.CollisionModel(objpath, cdmesh_type='triangles')
    iscollided, contact_points = is_collided(objcm1, objcm2)
    # objcm1.show_cdmesh(type='box')
    # show_triangles_cdmesh(objcm1)
    # show_triangles_cdmesh(objcm2)
    show_cdmesh(objcm1)
    show_cdmesh(objcm2)
    # objcm1.show_cdmesh(type='box')
    # objcm2.show_cdmesh(type='triangles')
    objcm1.attach_to(base)
    objcm2.attach_to(base)
    print(iscollided)
    for ct_pnt in contact_points:
        gm.gen_sphere(ct_pnt, radius=.001).attach_to(base)
    base.run()

