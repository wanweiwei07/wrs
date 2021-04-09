import numpy as np
import basis.data_adapter as da
from panda3d.ode import OdeTriMeshData, OdeTriMeshGeom, OdeUtil

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
    contact_entry = OdeUtil.collide(obj0, obj1)
    contact_points = [da.pdv3_to_npv3(point) for point in contact_entry.getContactPoints()]
    return (True, contact_points) if len(contact_points)>0 else (False, contact_points)

# # incorrect result 20210119, from panda3d.ode import OdeRayGeom if toggled on
# def rayhit_closet(pfrom, pto, objcm):
#     """
#     :param pfrom:
#     :param pto:
#     :param objcm:
#     :return:
#     author: weiwei
#     date: 20190805
#     """
#     tgt_cdmesh = gen_cdmesh_vvnf(*objcm.extract_rotated_vvnf())
#     ray = OdeRayGeom(1e12)
#     ray.set(pfrom[0], pfrom[1], pfrom[2], pto[0], pto[1], pto[2])
#     contact_entry = OdeUtil.collide(ray, tgt_cdmesh)
#     contact_points = [da.pdv3_to_npv3(point) for point in contact_entry.getContactPoints()]
#     print(contact_points)


if __name__ == '__main__':
    import os, math, basis
    import numpy as np
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm
    import modeling.collisionmodel as cm
    import basis.robot_math as rm

    wd.World(cam_pos=[1.0, 1, .0, 1.0], lookat_pos=[0, 0, 0])
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    objcm1= cm.CollisionModel(objpath)
    homomat = np.eye(4)
    homomat[:3, :3] = rm.rotmat_from_axangle([0, 0, 1], math.pi / 2)
    homomat[:3, 3] = np.array([0.02, 0.02, 0])
    objcm1.set_homomat(homomat)
    objcm1.set_rgba([1,1,.3,.2])
    objcm2 = objcm1.copy()
    objcm2.set_pos(objcm1.get_pos()+np.array([.05,.02,.0]))
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
    # pfrom = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    # pto = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -0.9])
    # rayhit_closet(pfrom=pfrom, pto=pto, objcm=objcm2)
    # objcm.attach_to(base)
    # objcm.show_cdmesh(type='box')
    # objcm.show_cdmesh(type='convex_hull')
    # gm.gen_sphere(hitpos, radius=.003, rgba=np.array([0, 1, 1, 1])).attach_to(base)
    # gm.gen_stick(spos=pfrom, epos=pto, thickness=.002).attach_to(base)
    # gm.gen_arrow(spos=hitpos, epos=hitpos + hitnrml * .07, thickness=.002, rgba=np.array([0, 1, 0, 1])).attach_to(base)
    base.run()