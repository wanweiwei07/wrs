# primitive collision detection helper
import math
import numpy as np
import basis.data_adapter as da
from panda3d.core import NodePath, CollisionNode, CollisionTraverser, CollisionHandlerQueue, BitMask32
from panda3d.core import CollisionBox, CollisionSphere, CollisionPolygon, GeomVertexReader


def gen_box_cdnp(pdnp, name='cdnp_box', radius=0.01):
    """
    :param obstacle:
    :return:
    author: weiwei
    date: 20180811
    """
    bottom_left, top_right = pdnp.getTightBounds()
    center = (bottom_left + top_right) / 2.0
    # enlarge the bounding box
    bottom_left -= (bottom_left - center).normalize() * radius
    top_right += (top_right - center).normalize() * radius
    collision_primitive = CollisionBox(bottom_left, top_right)
    collision_node = CollisionNode(name)
    collision_node.addSolid(collision_primitive)
    return collision_node


def gen_cylindrical_cdnp(pdnp, name='cdnp_cylinder', radius=0.01):
    """
    :param trimeshmodel:
    :param name:
    :param radius:
    :return:
    author: weiwei
    date: 20200108
    """
    bottom_left, top_right = pdnp.getTightBounds()
    center = (bottom_left + top_right) / 2.0
    # enlarge the bounding box
    bottomleft_adjustvec = bottom_left - center
    bottomleft_adjustvec[2] = 0
    bottomleft_adjustvec.normalize()
    bottom_left += bottomleft_adjustvec * radius
    topright_adjustvec = top_right - center
    topright_adjustvec[2] = 0
    topright_adjustvec.normalize()
    top_right += topright_adjustvec * radius
    bottomleft_pos = da.pdv3_to_npv3(bottom_left)
    topright_pos = da.pdv3_to_npv3(top_right)
    collision_node = CollisionNode(name)
    for angle in np.nditer(np.linspace(math.pi / 10, math.pi * 4 / 10, 4)):
        ca = math.cos(angle)
        sa = math.sin(angle)
        new_bottomleft_pos = np.array([bottomleft_pos[0] * ca, bottomleft_pos[1] * sa, bottomleft_pos[2]])
        new_topright_pos = np.array([topright_pos[0] * ca, topright_pos[1] * sa, topright_pos[2]])
        new_bottomleft = da.npv3_to_pdv3(new_bottomleft_pos)
        new_topright = da.npv3_to_pdv3(new_topright_pos)
        collision_primitive = CollisionBox(new_bottomleft, new_topright)
        collision_node.addSolid(collision_primitive)
    return collision_node


def gen_polygons_cdnp(pdnp, name='cdnp_polygons', radius=.01):
    """
    :param trimeshmodel:
    :param name:
    :param radius: TODO
    :return:
    author: weiwei
    date: 20210204
    """
    collision_node = CollisionNode(name)
    # counter = 0
    for geom in pdnp.findAllMatches('**/+GeomNode'):
        geom_node = geom.node()
        for g in range(geom_node.getNumGeoms()):
            geom = geom_node.getGeom(g).decompose()
            vdata = geom.getVertexData()
            vreader = GeomVertexReader(vdata, 'vertex')
            for p in range(geom.getNumPrimitives()):
                prim = geom.getPrimitive(p)
                for p2 in range(prim.getNumPrimitives()):
                    s = prim.getPrimitiveStart(p2)
                    e = prim.getPrimitiveEnd(p2)
                    v = []
                    for vi in range(s, e):
                        vreader.setRow(prim.getVertex(vi))
                        # TODO expand radius by moving along normal directions
                        v.append(vreader.getData3f())
                    col_poly = CollisionPolygon(*v)
                    collision_node.addSolid(col_poly)
                    # print("polygon ", counter)
                    # counter += 1
    return collision_node


def gen_surfaceballs_cdnp(objtrm, name='cdnp_surfaceball', radius=0.01):
    """
    :param obstacle:
    :return:
    author: weiwei
    date: 20180811
    """
    nsample = int(math.ceil(objtrm.area / (radius * 0.3) ** 2))
    nsample = 120 if nsample > 120 else nsample  # threshhold
    samples = objtrm.sample_surface(nsample)
    collision_node = CollisionNode(name)
    for sglsample in samples:
        collision_node.addSolid(CollisionSphere(sglsample[0], sglsample[1], sglsample[2], radius=radius))
    return collision_node


def gen_pointcloud_cdnp(objtrm, name='cdnp_pointcloud', radius=0.02):
    """
    :param obstacle:
    :return:
    author: weiwei
    date: 20191210
    """
    collision_node = CollisionNode(name)
    for sglpnt in objtrm.vertices:
        collision_node.addSolid(CollisionSphere(sglpnt[0], sglpnt[1], sglpnt[2], radius=radius))
    return collision_node


def is_collided(objcm_list0, objcm_list1, toggle_contact_points=False, toggle_plot_cdprimit=False):
    """
    detect the collision between collision models
    :param: objcm_list0, a single collision model or a list of collision models
    :param: objcm_list1
    :return: True or False
    author: weiwei
    date: 20190312osaka, 20201214osaka
    """
    if not isinstance(objcm_list0, list):
        objcm_list0 = [objcm_list0]
    if not isinstance(objcm_list1, list):
        objcm_list1 = [objcm_list1]
    if toggle_plot_cdprimit:
        for one_objcm in objcm_list0:
            one_objcm.show_cdprimit()
        for one_objcm in objcm_list1:
            one_objcm.show_cdprimit()
    tmpnp = NodePath("collision nodepath")
    ctrav = CollisionTraverser()
    chan = CollisionHandlerQueue()
    for one_objcm in objcm_list0:
        ctrav.addCollider(one_objcm.copy_cdnp_to(tmpnp), chan)
    for one_objcm in objcm_list1:
        one_objcm.copy_cdnp_to(tmpnp)
    ctrav.traverse(tmpnp)
    if chan.getNumEntries() > 0:
        if toggle_contact_points:
            contact_points = [da.pdv3_to_npv3(cd_entry.getSurfacePoint(base.render)) for cd_entry in chan.getEntries()]
            return True, contact_points
        else:
            return True
    else:
        return False


if __name__ == '__main__':
    import os
    import time
    import basis
    import numpy as np
    import modeling.collision_model as cm
    import modeling.geometric_model as gm
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[.7, .7, .7], lookat_pos=[0, 0, 0])
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    objcm = cm.CollisionModel(objpath, cdprimit_type='polygons')
    objcm.set_rgba(np.array([.2, .5, 0, 1]))
    objcm.set_pos(np.array([.01, .01, .01]))
    objcm.attach_to(base)
    objcm.show_cdprimit()
    objcmlist = []
    for i in range(100):
        objcmlist.append(cm.CollisionModel(os.path.join(basis.__path__[0], 'objects', 'housing.stl'), cdprimit_type='box'))
        objcmlist[-1].set_pos(np.random.random_sample((3,)))
        objcmlist[-1].set_rgba(np.array([1, .5, 0, 1]))
        objcmlist[-1].attach_to(base)
        objcmlist[-1].show_cdprimit()

    tic = time.time()
    result = is_collided(objcm, objcmlist)
    toc = time.time()
    time_cost = toc - tic
    print(time_cost)
    print(result)
    # tic = time.time()
    # is_cmcmlist_collided2(objcm, objcmlist)
    # toc = time.time()
    # time_cost = toc-tic
    # print(time_cost)
    base.run()

    # NOTE 20210321, CollisionPolygon into CollisonPolygon detection is not available for 1.19
    # :collide(error): Invalid attempt to detect collision from CollisionPolygon!
    #
    # This means that a CollisionPolygon object was added to a
    # CollisionTraverser as if it were a colliding object.  However,
    # no implementation for this kind of object has yet been defined
    # to collide with other objects.
    # wd.World(cam_pos=[1.0, 1, .0, 1.0], lookat_pos=[0, 0, 0])
    # objpath = os.path.join(basis.__path__[0], 'objects', 'yumifinger.stl')
    # objcm1 = cm.CollisionModel(objpath, cdprimitive_type='polygons')
    # # homomat = np.array([[-0.5, -0.82363909, 0.2676166, -0.00203699],
    # #                     [-0.86602539, 0.47552824, -0.1545085, 0.01272306],
    # #                     [0., -0.30901703, -0.95105648, 0.12604253],
    # #                     [0., 0., 0., 1.]])
    # homomat = np.array([[ 1.00000000e+00,  2.38935501e-16,  3.78436685e-17, -7.49999983e-03],
    #                     [ 2.38935501e-16, -9.51056600e-01, -3.09017003e-01,  2.04893537e-02],
    #                     [-3.78436685e-17,  3.09017003e-01, -9.51056600e-01,  1.22025304e-01],
    #                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # objcm1.set_homomat(homomat)
    # objcm1.set_rgba([1, 1, .3, .2])
    #
    # objpath = os.path.join(basis.__path__[0], 'objects', 'tubebig.stl')
    # objcm2 = cm.CollisionModel(objpath, cdprimitive_type='polygons')
    # objcm2.set_rgba([1, 1, .3, .2])
    # iscollided, contact_points = is_collided(objcm1, objcm2, toggle_contact_points=True)
    # objcm1.show_cdmesh()
    # objcm2.show_cdmesh()
    # objcm1.attach_to(base)
    # objcm2.attach_to(base)
    # print(iscollided)
    # for ct_pnt in contact_points:
    #     gm.gen_sphere(ct_pnt, radius=.001).attach_to(base)
    # base.run()

