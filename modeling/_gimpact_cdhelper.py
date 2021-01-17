import copy
import numpy as np
import basis.robotmath as rm
import basis.dataadapter as da
import basis.trimeshgenerator as tg
import modeling.geometricmodel as gm
import gimpact as gi


# util functions
def _gen_aabb_cdmesh(objcm_list):
    """
    generate the gimpact.TriMesh using the AABB boundary of a objcm_list
    :param objcm_list: a collision model or a list of collision model
    :return: gimpact.TriMesh
    author: weiwei
    date: 20210116osaka-u
    """
    if not isinstance(objcm_list, list):
        objcm_list = [objcm_list]
    vertices = []
    indices = []
    for objcm in objcm_list:
        objtrm = objcm.objtrm.bounding_box
        homomat = objcm.get_homomat()
        vertices += rm.homomat_transform_points(homomat, objtrm.vertices).tolist()
        indices += (objtrm.faces.flatten()+len(indices)).tolist()
    gtrm = gi.TriMesh(np.array(vertices), np.array(indices))
    return gtrm

def _gen_obb_cdmesh(objcm_list):
    """
    generate the gimpact.TriMesh using the OBB boundary of a objcm_list
    :param objcm_list: a collision model or a list of collision model
    :return: gimpact.TriMesh
    author: weiwei
    date: 20210116osaka-u
    """
    if not isinstance(objcm_list, list):
        objcm_list = [objcm_list]
    vertices = []
    indices = []
    for objcm in objcm_list:
        objtrm = objcm.objtrm.bounding_box_oriented
        homomat = objcm.get_homomat()
        vertices += rm.homomat_transform_points(homomat, objtrm.vertices).tolist()
        indices += (objtrm.faces.flatten()+len(indices)).tolist()
    gtrm = gi.TriMesh(np.array(vertices), np.array(indices))
    return gtrm

def _gen_convexhull_cdmesh(objcm_list):
    """
    generate gimpact.TriMesh using the convexhulls of a objcm_list
    :param objcm_list: a collision model or a list of collision model
    :return: gimpact.TriMesh
    author: weiwei
    date: 20210116osaka-u
    """
    if not isinstance(objcm_list, list):
        objcm_list = [objcm_list]
    vertices = []
    indices = []
    for objcm in objcm_list:
        objtrm = objcm.objtrm.convex_hull
        homomat = objcm.get_homomat()
        vertices += rm.homomat_transform_points(homomat, objtrm.vertices).tolist()
        indices += (objtrm.faces.flatten() + len(indices)).tolist()
    gtrm = gi.TriMesh(np.array(vertices), np.array(indices))
    return gtrm

def _gen_triangles_cdmesh(objcm_list):
    """
    generate gimpact.TriMesh of a objcm_list
    :param objcm_list: a collision model or a list of collision model
    :return: gimpact.TriMesh
    author: weiwei
    date: 20210116osaka-u
    """
    if not isinstance(objcm_list, list):
        objcm_list = [objcm_list]
    vertices = []
    indices = []
    for objcm in objcm_list:
        homomat = objcm.get_homomat()
        vertices += rm.homomat_transform_points(homomat, objcm.objtrm.vertices).tolist()
        indices += (objcm.objtrm.faces.flatten() + len(indices)).tolist()
    gtrm = gi.TriMesh(np.array(vertices), np.array(indices))
    return gtrm

def _cdmesh_from_objcmlist(objcm):
    """
    :param objcm: a collision model
    :return:
    """
    if objcm.cdmesh_type == 'aabb':
        gen_cdmesh_fn = _gen_aabb_cdmesh
    elif objcm.cdmesh_type == 'obb':
        gen_cdmesh_fn = _gen_obb_cdmesh
    elif objcm.cdmesh_type == 'convexhull':
        gen_cdmesh_fn = _gen_convexhull_cdmesh
    elif objcm.cdmesh_type == 'triangles':
        gen_cdmesh_fn = _gen_triangles_cdmesh
    return gen_cdmesh_fn(objcm)

def is_collided(objcm0, objcm1):
    """
    check if two objcm are collided after converting the specified cdmesh_type
    :param objcm0:
    :param objcm1:
    :return:
    author: weiwei
    date: 20210117
    """
    gtrm0 = _cdmesh_from_objcmlist(objcm0)
    gtrm1 = _cdmesh_from_objcmlist(objcm1)
    contacts = gi.trimesh_trimesh_collision(gtrm0, gtrm1)
    return (True, contacts) if len(contacts)>0 else (False, contacts)

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
    objcm1= cm.CollisionModel(objpath)
    homomat = np.eye(4)
    homomat[:3, :3] = rm.rotmat_from_axangle([0, 0, 1], math.pi / 2)
    homomat[:3, 3] = np.array([0.02, 0.02, 0])
    objcm1.set_homomat(homomat)
    objcm1.set_rgba([1,1,.3,.2])
    objcm2 = objcm1.copy()
    objcm2.set_pos(objcm1.get_pos()+np.array([.05,.02,.0]))
    objcm1.change_cdmesh_type('convexhull')
    objcm2.change_cdmesh_type('obb')
    iscollided, contacts = is_collided(objcm1, objcm2)
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
    for ct in contacts:
        gm.gen_sphere(ct.point, radius=.001).attach_to(base)
    # pfrom = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    # pto = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -0.9])
    # hitpos, hitnrml = rayhit_triangles_closet(pfrom=pfrom, pto=pto, objcm=objcm)
    # objcm.attach_to(base)
    # objcm.show_cdmesh(type='box')
    # objcm.show_cdmesh(type='convexhull')
    # gm.gen_sphere(hitpos, radius=.003, rgba=np.array([0, 1, 1, 1])).attach_to(base)
    # gm.gen_stick(spos=pfrom, epos=pto, thickness=.002).attach_to(base)
    # gm.gen_arrow(spos=hitpos, epos=hitpos + hitnrml * .07, thickness=.002, rgba=np.array([0, 1, 0, 1])).attach_to(base)
    base.run()
