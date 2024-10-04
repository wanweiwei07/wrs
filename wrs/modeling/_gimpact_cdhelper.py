"""
gimpact is the foundamental collision detection library used by bullet and ode.
this file is deprecated in 2021. reason: traversing multiple collision objects is time-consuming.
"""

import numpy as np
import gimpact as gi

# util functions
def gen_cdmesh_vvnf(vertices, vertex_normals, faces):
    """
    generate cdmesh given vertices, _, and faces
    :return: gimpact.TriMesh (require gimpact to be installed)
    author: weiwei
    date: 20210118
    """
    return gi.TriMesh(vertices, faces.flatten())

def is_collided(objcm0, objcm1):
    """
    check if two obj_cmodel are collided after converting the specified cdmesh_type
    :param objcm0:
    :param objcm1:
    :return:
    author: weiwei
    date: 20210117
    """
    obj0 = gen_cdmesh_vvnf(*objcm0.get_cdmesh_vvnf())
    obj1 = gen_cdmesh_vvnf(*objcm1.get_cdmesh_vvnf())
    contacts = gi.trimesh_trimesh_collision(obj0, obj1)
    contact_points = [ct.point for ct in contacts]
    return (True, contact_points) if len(contact_points)>0 else (False, contact_points)

def gen_plane_cdmesh(updirection=np.array([0, 0, 1]), offset=0, name='autogen'):
    """
    generate a plane bulletrigidbody state
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


if __name__ == '__main__':
    import os
    from wrs import basis, modeling as gm, modeling as cm
    import numpy as np
    import wrs.visualization.panda.world as wd

    # wd.World(cam_pos=[1.0, 1, .0, 1.0], lookat_pos=[0, 0, 0])
    # obj_path = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    # objcm1= mcm.CollisionModel(obj_path)
    # pos = np.eye(4)
    # pos[:3, :3] = rm.rotmat_from_axangle([0, 0, 1], math.pi / 2)
    # pos[:3, 3] = np.array([0.02, 0.02, 0])
    # objcm1.set_homomat(pos)
    # objcm1.set_rgba([1,1,.3,.2])
    # objcm2 = objcm1.copy()
    # objcm2.set_pos(objcm1.get_pos()+np.array([.05,.02,.0]))
    # objcm1.change_cdmesh_type('convex_hull')
    # objcm2.change_cdmesh_type('obb')
    # iscollided, contacts = is_collided(objcm1, objcm2)
    # # objcm1.show_cdmesh(end_type='box')
    # # show_triangles_cdmesh(objcm1)
    # # show_triangles_cdmesh(objcm2)
    # show_cdmesh(objcm1)
    # show_cdmesh(objcm2)
    # # objcm1.show_cdmesh(end_type='box')
    # # objcm2.show_cdmesh(end_type='triangles')
    # objcm1.attach_to(base)
    # objcm2.attach_to(base)
    # print(iscollided)
    # for ct in contacts:
    #     mgm.gen_sphere(ct.point, major_radius=.001).attach_to(base)
    # # spos = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    # # epos = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -0.9])
    # # hitpos, hitnrml = rayhit_triangles_closet(spos=spos, epos=epos, obj_cmodel=obj_cmodel)
    # # obj_cmodel.attach_to(base)
    # # obj_cmodel.show_cdmesh(end_type='box')
    # # obj_cmodel.show_cdmesh(end_type='convex_hull')
    # # mgm.gen_sphere(hitpos, major_radius=.003, rgba=np.array([0, 1, 1, 1])).attach_to(base)
    # # mgm.gen_stick(spos=spos, epos=epos, major_radius=.002).attach_to(base)
    # # mgm.gen_arrow(spos=hitpos, epos=hitpos + hitnrml * .07, major_radius=.002, rgba=np.array([0, 1, 0, 1])).attach_to(base)
    # base.run()

    wd.World(cam_pos=[1.0, 1, .0, 1.0], lookat_pos=[0, 0, 0])
    objpath = os.path.join(basis.__path__[0], 'objects', 'yumifinger.stl')
    objcm1= cm.CollisionModel(objpath, cdmesh_type='triangles')
    homomat = np.array([[ 5.00000060e-01,  7.00629234e-01,  5.09036899e-01, -3.43725011e-02],
                        [ 8.66025329e-01, -4.04508471e-01, -2.93892622e-01,  5.41121606e-03],
                        [-2.98023224e-08,  5.87785244e-01, -8.09016943e-01,  1.13636881e-01],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    homomat = np.array([[ 1.00000000e+00,  2.38935501e-16,  3.78436685e-17, -7.49999983e-03],
                        [ 2.38935501e-16, -9.51056600e-01, -3.09017003e-01,  2.04893537e-02],
                        [-3.78436685e-17,  3.09017003e-01, -9.51056600e-01,  1.22025304e-01],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    objcm1.set_homomat(homomat)
    objcm1.set_rgba([1,1,.3,.2])

    objpath = os.path.join(basis.__path__[0], 'objects', 'tubebig.stl')
    objcm2= cm.CollisionModel(objpath, cdmesh_type='triangles')
    iscollided, contact_points = is_collided(objcm1, objcm2)
    # objcm1.show_cdmesh(end_type='box')
    # show_triangles_cdmesh(objcm1)
    # show_triangles_cdmesh(objcm2)
    objcm1.show_cdmesh()
    objcm2.show_cdmesh()
    # objcm1.show_cdmesh(end_type='box')
    # objcm2.show_cdmesh(end_type='triangles')
    objcm1.attach_to(base)
    objcm2.attach_to(base)
    print(iscollided)
    for ctpt in contact_points:
        gm.gen_sphere(ctpt, radius=.001).attach_to(base)
    # spos = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    # epos = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -0.9])
    # hitpos, hitnrml = rayhit_triangles_closet(spos=spos, epos=epos, obj_cmodel=obj_cmodel)
    # obj_cmodel.attach_to(base)
    # obj_cmodel.show_cdmesh(end_type='box')
    # obj_cmodel.show_cdmesh(end_type='convex_hull')
    # mgm.gen_sphere(hitpos, major_radius=.003, rgba=np.array([0, 1, 1, 1])).attach_to(base)
    # mgm.gen_stick(spos=spos, epos=epos, major_radius=.002).attach_to(base)
    # mgm.gen_arrow(spos=hitpos, epos=hitpos + hitnrml * .07, major_radius=.002, rgba=np.array([0, 1, 0, 1])).attach_to(base)
    base.run()
