import copy
import numpy as np
import basis.robotmath as rm
import basis.dataadapter as da
import basis.trimeshgenerator as tg
import modeling.geometricmodel as gm
import gimpact as gi

# box collision model
def is_box2box_collided(objcm_list0, objcm_list1):
    """
    check if two objects objcm0 as objcm1 are in collision with each other
    the two objects are in the form of collision model
    the AABB boxlist will be used
    :param objcm_list0: the first object or list
    :param objcm_list1: the second object or list
    :return: boolean value showing if the two objects are in collision, a list of contacts
    author: weiwei
    date: 20210116
    """
    gtrm0 = _gen_box_cdmesh(objcm_list0)
    gtrm1 = _gen_box_cdmesh(objcm_list1)
    contacts = gi.trimesh_trimesh_collision(gtrm0, gtrm1)
    return (True, contacts) if len(contacts)>0 else (False, contacts)


# 2 box - triangles
def is_box2triangles_collided(objcm_list0, objcm_list1):
    """
    check if two objects objcm0 as objcm1 are in collision with each other
    the two objects are in the form of collision model
    the AABB boxlist will be used for objcm_list0
    :param objcm_list0: the first object
    :param objcm_list1: the second object
    :return: boolean value showing if the two objects are in collision, a list of contacts
    author: weiwei
    date: 20210116
    """
    gtrm0 = _gen_box_cdmesh(objcm_list0)
    gtrm1 = _gen_triangles_cdmesh(objcm_list1)
    contacts = gi.trimesh_trimesh_collision(gtrm0, gtrm1)
    return True, contacts if len(contacts)>0 else False, contacts


# triangles collision model
def is_triangles2triangles_collided(objcm_list0, objcm_list1):
    """
    check if two objects objcm0 and objcm1 are in collision with each other
    the two objects are in the form of collision model
    :param objcm_list0: the first object
    :param objcm_list1: the second object
    :return: boolean value showing if the two objects are in collision, a list of contacts
    author: weiwei
    date: 20210116
    """
    gtrm0 = _gen_triangles_cdmesh(objcm_list0)
    gtrm1 = _gen_triangles_cdmesh(objcm_list1)
    contacts = gi.trimesh_trimesh_collision(gtrm0, gtrm1)
    return [True, contacts] if len(contacts)>0 else [False, contacts]
    # if result.getNumContacts():
    #     import modeling.geometricmodel as gm
    #     for contact in result.getContacts():
    #         gm.gen_sphere(pos = da.pdv3_to_npv3(contact.getManifoldPoint().getLocalPointA()), radius=0.001).attach_to(base)
    #         gm.gen_sphere(pos = da.pdv3_to_npv3(contact.getManifoldPoint().getLocalPointB()), radius=0.001).attach_to(base)
    #         gm.gen_arrow(spos = da.pdv3_to_npv3(contact.getManifoldPoint().getPositionWorldOnB()),
    #                      epos = da.pdv3_to_npv3(contact.getManifoldPoint().getPositionWorldOnB())+da.pdv3_to_npv3(contact.getManifoldPoint().getNormalWorldOnB())).attach_to(base)
    #         # gm.gen_sphere(pos = da.pdv3_to_npv3(contact.getManifoldPoint().getPositionWorldOnA()), radius=0.001, rgba=[0,1,0,1]).attach_to(base)
    #         # gm.gen_sphere(pos = da.pdv3_to_npv3(contact.getManifoldPoint().getPositionWorldOnA()), radius=0.001, rgba=[0,1,0,1]).attach_to(base)
    # return True if result.getNumContacts() else False


def show_box_cdmesh(objcm_list):
    """
    show the AABB collision meshes of the given objects
    :param objcm_list
    author: weiwei
    date: 20190313
    :return:
    """
    if not isinstance(objcm_list, list):
        objcm_list = [objcm_list]
    for objcm in objcm_list:
        bottom_left, top_right = objcm.objpdnp_raw.getTightBounds()
        extent = da.pdv3_to_npv3(top_right-bottom_left)
        bound_tf = objcm.get_homomat().dot(rm.homomat_from_posrot(pos=da.pdv3_to_npv3((bottom_left+top_right)/2)))
        trm_box = tg.gen_box(extent, bound_tf)
        wm_box = gm.WireFrameModel(trm_box)
        wm_box.attach_to(base)


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
def _gen_box_cdmesh(objcm_list):
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
        bottom_left, top_right = objcm.objpdnp_raw.getTightBounds()
        extent = da.pdv3_to_npv3(top_right-bottom_left)
        bound_tf = objcm.get_homomat().dot(rm.homomat_from_posrot(pos=da.pdv3_to_npv3((bottom_left+top_right)/2)))
        trm_box = tg.gen_box(extent, bound_tf)
        vertices += trm_box.vertices.tolist()
        indices += (trm_box.faces.flatten()+len(indices)).tolist()
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
        objtrm_cvx = objcm.objtrm.convex_hull
        homomat = objcm.get_homomat()
        vertices += rm.homomat_transform_points(homomat, objtrm_cvx.objtrm.vertices).tolist()
        indices += (objtrm_cvx.objtrm.faces.flatten() + len(indices)).tolist()
    gtrm = gi.TriMesh(vertices, indices)
    return gtrm

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
    objcm1= cm.CollisionModel(objpath)
    homomat = np.eye(4)
    homomat[:3, :3] = rm.rotmat_from_axangle([0, 0, 1], math.pi / 2)
    homomat[:3, 3] = np.array([0, 0.02, 0])
    objcm1.set_homomat(homomat)
    objcm1.set_rgba([1,1,.3,.2])
    objcm2 = objcm1.copy()
    objcm2.set_pos(objcm1.get_pos()+np.array([.0668,.03,0]))
    iscollided, contacts = is_box2box_collided(objcm1, objcm2)
    # objcm1.show_cdmesh(type='box')
    show_box_cdmesh(objcm1)
    objcm2.show_cdmesh(type='box')
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
