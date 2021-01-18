import numpy as np
import basis.robotmath as rm
import basis.dataadapter as da
from panda3d.ode import OdeTriMeshData, OdeTriMeshGeom, OdeUtil

# util functions
def _gen_cdmesh_vvnf(vertices, vertex_normals, faces):
    """
    generate cdmesh given vertices, vertex_normals, and faces
    :return: panda3d.ode.OdeTriMeshGeomm
    author: weiwei
    date: 20210118
    """
    objpdnp = da.nodepath_from_vvnf(vertices, vertex_normals, faces)
    obj_ot_geom = OdeTriMeshGeom(OdeTriMeshData(objpdnp, True))
    return obj_ot_geom

def is_collided(objcm0, objcm1):
    """
    check if two objcm are collided after converting the specified cdmesh_type
    :param objcm0: an instance of CollisionModel or CollisionModelCollection
    :param objcm1: an instance of CollisionModel or CollisionModelCollection
    :return:
    author: weiwei
    date: 20210118
    """
    obj_ot_geom0 = objcm0.cdmesh
    obj_ot_geom1 = objcm1.cdmesh
    contact_entry = OdeUtil.collide(obj_ot_geom0, obj_ot_geom1)
    contact_points = [da.pdv3_to_npv3(point) for point in contact_entry.getContactPoints()]
    return (True, contact_points) if len(contact_points)>0 else (False, contact_points)


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
    iscollided, contact_points = is_collided(objcm1, objcm2)
    # objcm1.show_cdmesh(type='box')
    # show_triangles_cdmesh(objcm1)
    # show_triangles_cdmesh(objcm2)
    objcm1.show_cdmesh()
    objcm2.show_cdmesh()
    # objcm1.show_cdmesh(type='box')
    # objcm2.show_cdmesh(type='triangles')
    objcm1.attach_to(base)
    objcm2.attach_to(base)
    print(iscollided)
    for ctpt in contact_points:
        gm.gen_sphere(ctpt, radius=.001).attach_to(base)
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

    # wd.World(campos=[1.0, 1, .0, 1.0], lookatpos=[0, 0, 0])
    # objpath = os.path.join(basis.__path__[0], 'objects', 'b.stl')
    # objcm1= cm.CollisionModel(objpath, cdmesh_type='triangles')
    # objcm1.set_rgba([1,1,.3,.2])
    # objpath = os.path.join(basis.__path__[0], 'objects', 'tubebig.stl')
    # objcm2= cm.CollisionModel(objpath, cdmesh_type='triangles')
    # iscollided, contact_points = is_collided(objcm1, objcm2)
    # # objcm1.show_cdmesh(type='box')
    # # show_triangles_cdmesh(objcm1)
    # # show_triangles_cdmesh(objcm2)
    # show_cdmesh(objcm1)
    # show_cdmesh(objcm2)
    # # objcm1.show_cdmesh(type='box')
    # # objcm2.show_cdmesh(type='triangles')
    # objcm1.attach_to(base)
    # objcm2.attach_to(base)
    # print(iscollided)
    # for ct_pnt in contact_points:
    #     gm.gen_sphere(ct_pnt, radius=.001).attach_to(base)
    # # pfrom = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    # # pto = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -0.9])
    # # hitpos, hitnrml = rayhit_triangles_closet(pfrom=pfrom, pto=pto, objcm=objcm)
    # # objcm.attach_to(base)
    # # objcm.show_cdmesh(type='box')
    # # objcm.show_cdmesh(type='convexhull')
    # # gm.gen_sphere(hitpos, radius=.003, rgba=np.array([0, 1, 1, 1])).attach_to(base)
    # # gm.gen_stick(spos=pfrom, epos=pto, thickness=.002).attach_to(base)
    # # gm.gen_arrow(spos=hitpos, epos=hitpos + hitnrml * .07, thickness=.002, rgba=np.array([0, 1, 0, 1])).attach_to(base)
    # base.run()
