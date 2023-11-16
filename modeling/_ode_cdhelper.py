import numpy as np
import basis.robot_math as rm
import basis.data_adapter as da
from panda3d.ode import OdeTriMeshData, OdeTriMeshGeom, OdeUtil, OdeRayGeom
import copy

ode_util = OdeUtil()

def copy_cdmesh(objcm):
    """
    deprecated 20230814
    this function was used to replace deep copy
    previously, direclty copying Geom invokes a deprecated getdata method,
     see my question&comments at https://discourse.panda3d.org/t/ode-odetrimeshdata-problem/28232 for details
    :param objcm:
    :return:
    """
    pdotrmgeom = OdeTriMeshGeom(objcm.cdmesh.getTriMeshData())
    return pdotrmgeom


def gen_cdmesh(trm_model):
    """
    generate cdmesh given vertices, vertex_normals, and faces
    :return: panda3d.ode.OdeTriMeshGeomm
    author: weiwei
    date: 20210118
    """
    pdgeom_ndp = da.pdgeomndp_from_vvnf(trm_model.vertices, trm_model.vertex_normals, trm_model.faces)
    pdotrmgeom = OdeTriMeshGeom(OdeTriMeshData(model=pdgeom_ndp, use_normals=True))  # otgeom = ode trimesh geom
    return pdotrmgeom


def update_pose(pdotrmgeom, objcm):
    """
    update obj_ode_trimesh using the transformation matrix of objcm.pdndp
    :param pdotrmgeom:
    :param pdndp:
    :return:
    author: weiwei
    date: 20211215
    """
    pdotrmgeom.setPosition(objcm.pdndp.getPos())
    pdotrmgeom.setQuaternion(objcm.pdndp.getQuat())


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

def is_collided(objcm_list0, objcm_list1, toggle_contacts=True):
    """
    check if two objcm lists are collided
    :param objcm_list0: an instance of OdeTriMeshGeom
    :param objcm_list1: an instance of OdeTriMeshGeom
    :param toggle_contactpoints: True default
    :return:
    author: weiwei
    date: 20210118, 20211215, 20230814
    """
    if not isinstance(objcm_list0, list):
        objcm_list0 = [objcm_list0]
    if not isinstance(objcm_list1, list):
        objcm_list1 = [objcm_list1]
    for objcm0 in objcm_list0:
        for objcm1 in objcm_list1:
            pdotrmgeom0 = objcm0.copy_transformed_cdmesh()
            pdotrmgeom1 = objcm1.copy_transformed_cdmesh()
            contact_entry = ode_util.collide(pdotrmgeom0, pdotrmgeom1, 10)
            contact_points = np.asarray([da.pdvec3_to_npvec3(point) for point in contact_entry.getContactPoints()])
            if toggle_contacts:
                return (True, contact_points) if len(contact_points) > 0 else (False, contact_points)
            else:
                return True if len(contact_points) > 0 else False


def rayhit_closet(spos, epos, objcm):
    """
    :param spos:
    :param epos:
    :param objcm:
    :return:
    author: weiwei
    date: 20190805
    """
    pdotrmgeom = objcm.copy_transformed_cdmesh()
    ray = OdeRayGeom(length=1)
    length, dir = rm.unit_vector(epos - spos, toggle_length=True)
    ray.set(spos[0], spos[1], spos[2], dir[0], dir[1], dir[2])
    ray.setLength(length)
    contact_entry = ode_util.collide(ray, pdotrmgeom, max_contacts=10)
    contact_points = [da.pdvec3_to_npvec3(point) for point in contact_entry.getContactPoints()]
    min_id = np.argmin(np.linalg.norm(spos - np.array(contact_points), axis=1))
    contact_normals = [da.pdvec3_to_npvec3(contact_entry.getContactGeom(i).getNormal()) for i in
                       range(contact_entry.getNumContacts())]
    return contact_points[min_id], contact_normals[min_id]


def rayhit_all(spos, epos, objcm):
    """
    :param spos:
    :param epos:
    :param objcm:
    :return:
    author: weiwei
    date: 20190805
    """
    pdotrmgeom = objcm.copy_transformed_cdmesh()
    ray = OdeRayGeom(length=1)
    length, dir = rm.unit_vector(epos - spos, toggle_length=True)
    ray.set(spos[0], spos[1], spos[2], dir[0], dir[1], dir[2])
    ray.setLength(length)
    hit_entry = ode_util.collide(ray, pdotrmgeom)
    hit_points = [da.pdvec3_to_npvec3(point) for point in hit_entry.getContactPoints()]
    hit_normals = [da.pdvec3_to_npvec3(hit_entry.getContactGeom(i).getNormal()) for i in
                   range(hit_entry.getNumContacts())]
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
    objcm1.change_cdmesh_type(cm.CDMeshType.CONVEX_HULL)
    objcm2.change_cdmesh_type(cm.CDMeshType.OBB)
    iscollided, contact_points = is_collided(objcm1, objcm2)
    objcm1.show_cdmesh()
    objcm2.show_cdmesh()
    objcm1.attach_to(base)
    objcm2.attach_to(base)
    print(iscollided)
    for ctpt in contact_points:
        gm.gen_sphere(ctpt, radius=.001).attach_to(base)
    pfrom = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    # epos = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -1.0])
    pto = np.array([0, 0, 0]) + np.array([0.02, 0.02, 0.02])
    # spos = np.array([0, 0, 0]) + np.array([0.0, 0.0, 1.0])
    # epos = np.array([0, 0, 0]) + np.array([0.0, 0.0, -1.0])
    # hit_point, hit_normal = rayhit_closet(spos=spos, epos=epos, objcm=objcm1)
    hit_points, hit_normals = rayhit_all(spos=pfrom, epos=pto, objcm=objcm2)
    # objcm.attach_to(base)
    # objcm.show_cdmesh(end_type='box')
    # objcm.show_cdmesh(end_type='convex_hull')
    # for hitpos, hitnormal in zip([hit_point], [hit_normal]):
    for hitpos, hitnormal in zip(hit_points, hit_normals):
        gm.gen_sphere(hitpos, radius=.003, rgba=np.array([0, 1, 1, 1])).attach_to(base)
        gm.gen_arrow(hitpos, epos=hitpos + hitnormal * .03, stick_radius=.002, rgba=np.array([0, 1, 1, 1])).attach_to(
            base)
    gm.gen_stick(spos=pfrom, epos=pto, radius=.002).attach_to(base)
    # mgm.gen_arrow(spos=hitpos, epos=hitpos + hitnrml * .07, major_radius=.002, rgba=np.array([0, 1, 0, 1])).attach_to(base)
    base.run()
