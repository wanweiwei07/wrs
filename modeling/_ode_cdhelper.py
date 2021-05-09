import numpy as np
import basis.robot_math as rm
import basis.data_adapter as da
from panda3d.ode import OdeTriMeshData, OdeTriMeshGeom, OdeUtil, OdeRayGeom


# util functions
def gen_cdmesh_vvnf(vertices, vertex_normals, faces):
    """
    generate cdmesh given vertices, vertex_normals, and faces
    :return: panda3d.ode.OdeTriMeshGeomm
    author: weiwei
    date: 20210118
    """
    objpdnp = da.nodepath_from_vvnf(vertices, vertex_normals, faces)
    obj_ot_geom = OdeTriMeshGeom(OdeTriMeshData(objpdnp, True))
    return obj_ot_geom


# def gen_plane_cdmesh(updirection=np.array([0, 0, 1]), offset=0, name='autogen'):
#     """
#     generate a plane bulletrigidbody node
#     :param updirection: the normal parameter of bulletplaneshape at panda3d
#     :param offset: the d parameter of bulletplaneshape at panda3d
#     :param name:
#     :return: bulletrigidbody
#     author: weiwei
#     date: 20170202, tsukuba
#     """
#     bulletplnode = BulletRigidBodyNode(name)
#     bulletplshape = BulletPlaneShape(Vec3(updirection[0], updirection[1], updirection[2]), offset)
#     bulletplshape.setMargin(0)
#     bulletplnode.addShape(bulletplshape)
#     return bulletplnode

def is_collided(objcm0, objcm1):
    """
    check if two objcm are collided after converting the specified cdmesh_type
    :param objcm0: an instance of CollisionModel or CollisionModelCollection
    :param objcm1: an instance of CollisionModel or CollisionModelCollection
    :return:
    author: weiwei
    date: 20210118
    """
    obj0 = gen_cdmesh_vvnf(*objcm0.extract_rotated_vvnf())
    obj1 = gen_cdmesh_vvnf(*objcm1.extract_rotated_vvnf())
    contact_entry = OdeUtil.collide(obj0, obj1, max_contacts=10)
    contact_points = [da.pdv3_to_npv3(point) for point in contact_entry.getContactPoints()]
    return (True, contact_points) if len(contact_points) > 0 else (False, contact_points)

def rayhit_closet(pfrom, pto, objcm):
    """
    :param pfrom:
    :param pto:
    :param objcm:
    :return:
    author: weiwei
    date: 20190805
    """
    tgt_cdmesh = gen_cdmesh_vvnf(*objcm.extract_rotated_vvnf())
    ray = OdeRayGeom(length=1)
    length, dir = rm.unit_vector(pto - pfrom, toggle_length=True)
    ray.set(pfrom[0], pfrom[1], pfrom[2], dir[0], dir[1], dir[2])
    ray.setLength(length)
    contact_entry = OdeUtil.collide(ray, tgt_cdmesh, max_contacts=10)
    contact_points = [da.pdv3_to_npv3(point) for point in contact_entry.getContactPoints()]
    min_id = np.argmin(np.linalg.norm(pfrom-np.array(contact_points), axis=1))
    contact_normals = [da.pdv3_to_npv3(contact_entry.getContactGeom(i).getNormal()) for i in range(contact_entry.getNumContacts())]
    return contact_points[min_id], contact_normals[min_id]

def rayhit_all(pfrom, pto, objcm):
    """
    :param pfrom:
    :param pto:
    :param objcm:
    :return:
    author: weiwei
    date: 20190805
    """
    tgt_cdmesh = gen_cdmesh_vvnf(*objcm.extract_rotated_vvnf())
    ray = OdeRayGeom(length=1)
    length, dir = rm.unit_vector(pto-pfrom, toggle_length=True)
    ray.set(pfrom[0], pfrom[1], pfrom[2], dir[0], dir[1], dir[2])
    ray.setLength(length)
    hit_entry = OdeUtil.collide(ray, tgt_cdmesh)
    hit_points = [da.pdv3_to_npv3(point) for point in hit_entry.getContactPoints()]
    hit_normals = [da.pdv3_to_npv3(hit_entry.getContactGeom(i).getNormal()) for i in range(hit_entry.getNumContacts())]
    return hit_points, hit_normals


if __name__ == '__main__':
    import os, math, basis
    import numpy as np
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import modeling.collision_model as cm
    import basis.robot_math as rm

    wd.World(cam_pos=[1.0, 1, .0, 1.0], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    objcm1 = cm.CollisionModel(objpath)
    homomat = np.eye(4)
    homomat[:3, :3] = rm.rotmat_from_axangle([0, 0, 1], math.pi / 2)
    homomat[:3, 3] = np.array([0.02, 0.02, 0])
    objcm1.set_homomat(homomat)
    objcm1.set_rgba([1, 1, .3, .2])
    objcm2 = objcm1.copy()
    objcm2.set_pos(objcm1.get_pos() + np.array([.05, .02, .0]))
    objcm1.change_cdmesh_type('convex_hull')
    objcm2.change_cdmesh_type('obb')
    iscollided, contact_points = is_collided(objcm1, objcm2)
    objcm1.show_cdmesh()
    objcm2.show_cdmesh()
    objcm1.attach_to(base)
    objcm2.attach_to(base)
    print(iscollided)
    for ctpt in contact_points:
        gm.gen_sphere(ctpt, radius=.001).attach_to(base)
    pfrom = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    # pto = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -1.0])
    pto = np.array([0, 0, 0]) + np.array([0.02, 0.02, 0.02])
    # pfrom = np.array([0, 0, 0]) + np.array([0.0, 0.0, 1.0])
    # pto = np.array([0, 0, 0]) + np.array([0.0, 0.0, -1.0])
    # hit_point, hit_normal = rayhit_closet(pfrom=pfrom, pto=pto, objcm=objcm1)
    hit_points, hit_normals = rayhit_all(pfrom=pfrom, pto=pto, objcm=objcm1)
    # objcm.attach_to(base)
    # objcm.show_cdmesh(type='box')
    # objcm.show_cdmesh(type='convex_hull')
    # for hitpos, hitnormal in zip([hit_point], [hit_normal]):
    for hitpos, hitnormal in zip(hit_points, hit_normals):
        gm.gen_sphere(hitpos, radius=.003, rgba=np.array([0, 1, 1, 1])).attach_to(base)
        gm.gen_arrow(hitpos, epos=hitpos+hitnormal*.03, thickness=.002, rgba=np.array([0, 1, 1, 1])).attach_to(base)
    gm.gen_stick(spos=pfrom, epos=pto, thickness=.002).attach_to(base)
    # gm.gen_arrow(spos=hitpos, epos=hitpos + hitnrml * .07, thickness=.002, rgba=np.array([0, 1, 0, 1])).attach_to(base)
    base.run()
