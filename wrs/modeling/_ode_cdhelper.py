import numpy as np
from panda3d.ode import OdeTriMeshData, OdeTriMeshGeom, OdeUtil, OdeRayGeom
import wrs.basis.robot_math as rm
import wrs.basis.data_adapter as da
import wrs.modeling.collision_model as mcm

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
    update obj_ode_trimesh using the transformation matrix of obj_cmodel.pdndp
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

def is_collided(cmodel_list0, cmodel_list1, toggle_contacts=True):
    """
    check if two obj_cmodel lists are collided
    :param cmodel_list0: an instance of OdeTriMeshGeom
    :param cmodel_list1: an instance of OdeTriMeshGeom
    :param toggle_contactpoints: True default
    :return:
    author: weiwei
    date: 20210118, 20211215, 20230814
    """
    if isinstance(cmodel_list0, mcm.CollisionModel):
        cmodel_list0 = [cmodel_list0]
    if isinstance(cmodel_list1,  mcm.CollisionModel):
        cmodel_list1 = [cmodel_list1]
    for objcm0 in cmodel_list0:
        for objcm1 in cmodel_list1:
            contact_entry = ode_util.collide(objcm0.cdmesh, objcm1.cdmesh, 10)
            contact_points = np.asarray([da.pdvec3_to_npvec3(point) for point in contact_entry.getContactPoints()])
            if len(contact_points) > 0:
                if toggle_contacts:
                    return (True, contact_points)
                else:
                    return True
    if toggle_contacts:
        return (False, np.asarray([]))
    return False


def rayhit_closet(spos, epos, target_cmodel):
    """
    :param spos:
    :param epos:
    :param target_cmodel:
    :return:
    author: weiwei
    date: 20190805
    """
    ray = OdeRayGeom(length=1)
    length, dir = rm.unit_vector(epos - spos, toggle_length=True)
    ray.set(spos[0], spos[1], spos[2], dir[0], dir[1], dir[2])
    ray.setLength(length)
    contact_entry = ode_util.collide(ray, target_cmodel.cdmesh, max_contacts=10)
    contact_points = [da.pdvec3_to_npvec3(point) for point in contact_entry.getContactPoints()]
    if len(contact_points) == 0:
        return None
    min_id = np.argmin(np.linalg.norm(spos - np.array(contact_points), axis=1))
    contact_normals = [da.pdvec3_to_npvec3(contact_entry.getContactGeom(i).getNormal()) for i in
                       range(contact_entry.getNumContacts())]
    return contact_points[min_id], contact_normals[min_id]


def rayhit_all(spos, epos, target_cmodel):
    """
    :param spos:
    :param epos:
    :param target_cmodel:
    :return:
    author: weiwei
    date: 20190805
    """
    ray = OdeRayGeom(length=1)
    length, dir = rm.unit_vector(epos - spos, toggle_length=True)
    ray.set(spos[0], spos[1], spos[2], dir[0], dir[1], dir[2])
    ray.setLength(length)
    hit_entry = ode_util.collide(ray, target_cmodel.cdmesh)
    hit_points = [da.pdvec3_to_npvec3(point) for point in hit_entry.getContactPoints()]
    if len(hit_points) == 0:
        return None
    hit_normals = [da.pdvec3_to_npvec3(hit_entry.getContactGeom(i).getNormal()) for i in
                   range(hit_entry.getNumContacts())]
    return hit_points, hit_normals


if __name__ == '__main__':
    import os, math
    import numpy as np
    import wrs.basis.robot_math as rm
    import wrs.modeling.geometric_model as mgm
    import wrs.visualization.panda.world as wd

    wd.World(cam_pos=[1.0, 1, .0, 1.0], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    objpath = os.path.join(os.path.dirname(rm.__file__), 'objects', 'bunnysim.stl')
    objcm1 = mcm.CollisionModel(objpath)
    homomat = np.eye(4)
    homomat[:3, :3] = rm.rotmat_from_axangle([0, 0, 1], math.pi / 2)
    homomat[:3, 3] = np.array([0.02, 0.02, 0])
    objcm1.homomat = homomat
    objcm1.rgba = np.array([1, 1, .3, .2])
    objcm2 = objcm1.copy()
    objcm2.pos = objcm1.pos + np.array([.05, .02, .0])
    objcm1.change_cdmesh_type(mcm.const.CDMeshType.CONVEX_HULL)
    objcm2.change_cdmesh_type(mcm.const.CDMeshType.OBB)
    iscollided, contact_points = is_collided(objcm1, objcm2)
    objcm1.show_cdmesh()
    objcm2.show_cdmesh()
    objcm1.attach_to(base)
    objcm2.attach_to(base)
    print(iscollided)
    for ctpt in contact_points:
        mgm.gen_sphere(ctpt, radius=.001).attach_to(base)
    pfrom = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    # epos = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -1.0])
    pto = np.array([0, 0, 0]) + np.array([0.02, 0.02, 0.02])
    # spos = np.array([0, 0, 0]) + np.array([0.0, 0.0, 1.0])
    # epos = np.array([0, 0, 0]) + np.array([0.0, 0.0, -1.0])
    # hit_point, hit_normal = rayhit_closet(spos=spos, epos=epos, obj_cmodel=objcm1)
    hit_points, hit_normals = rayhit_all(spos=pfrom, epos=pto, target_cmodel=objcm2)
    # obj_cmodel.attach_to(base)
    # obj_cmodel.show_cdmesh(end_type='box')
    # obj_cmodel.show_cdmesh(end_type='convex_hull')
    # for hitpos, hitnormal in zip([hit_point], [hit_normal]):
    for hitpos, hitnormal in zip(hit_points, hit_normals):
        mgm.gen_sphere(hitpos, radius=.003, rgb=np.array([0, 1, 1])).attach_to(base)
        mgm.gen_arrow(hitpos, epos=hitpos + hitnormal * .03, stick_radius=.002, rgb=np.array([0, 1, 1])).attach_to(base)
    mgm.gen_stick(spos=pfrom, epos=pto, radius=.002).attach_to(base)
    # mgm.gen_arrow(spos=hitpos, epos=hitpos + hitnrml * .07, major_radius=.002, rgba=np.array([0, 1, 0, 1])).attach_to(base)
    base.run()
