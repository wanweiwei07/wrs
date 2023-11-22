# primitive collision detection helper
# note: This script is not used by robot simulations, as collison checker is faster
import math
import copy
import numpy as np
import basis.robot_math as rm
import basis.data_adapter as da
import basis.trimesh.bounds as trm_bounds
from panda3d.core import NodePath, CollisionNode, CollisionTraverser, CollisionHandlerQueue, BitMask32
from panda3d.core import CollisionBox, CollisionSphere, CollisionCapsule, CollisionPolygon, GeomVertexReader
from panda3d.core import LPoint3, TransformState, LineSegs, TransparencyAttrib

BITMASK_EXT = BitMask32(2 ** 31)


def copy_cdprimitive(objcm):
    return copy.deepcopy(objcm.cdprimitive)


def copy_cdprimitive_attach_to(objcm,
                               tgt_pdndp,
                               homomat=None,
                               clear_mask=False) -> NodePath:
    return_pdcndp = objcm.copy_reference_cdprimitive()
    return_pdcndp.reparentTo(tgt_pdndp)
    if homomat is None:
        return_pdcndp.setMat(objcm.pdndp.getMat())
    else:
        return_pdcndp.setMat(da.npmat4_to_pdmat4(homomat))  # scale is reset to 1 1 1 after setMat to the given pos
        # return_pdcndp.setScale(objcm.pdndp.getScale()) # 20231117 scale is validated
    if clear_mask:
        change_cdmask(return_pdcndp, BitMask32(0x00), action="new", type="both")
    return return_pdcndp


# def change_cdmask(cdprimitive, collision_mask: BitMask32, type):
#     """
#     :param cdprimitive: NodePath of CollisionNode
#     :param collision_mask:
#     :param type: 'from', 'into', 'both'
#     :return:
#     """
#     if type == "both":
#         if cdprimitive.getName() == "cylinder":
#             cdprimitive.getChild(0).node().setCollideMask(collision_mask)
#             cdprimitive.getChild(1).node().setCollideMask(collision_mask)
#             cdprimitive.getChild(2).node().setCollideMask(collision_mask)
#         else:
#             cdprimitive.node().setCollideMask(collision_mask)
#     elif type == "from":
#         if cdprimitive.getName() == "cylinder":
#             cdprimitive.getChild(0).node().setFromCollideMask(collision_mask)
#             cdprimitive.getChild(1).node().setFromCollideMask(collision_mask)
#             cdprimitive.getChild(2).node().setFromCollideMask(collision_mask)
#         else:
#             cdprimitive.node().setFromCollideMask(collision_mask)
#     elif type == "into":
#         if cdprimitive.getName() == "cylinder":
#             cdprimitive.getChild(0).node().getIntoCollideMask(collision_mask)
#             cdprimitive.getChild(1).node().v(collision_mask)
#             cdprimitive.getChild(2).node().getIntoCollideMask(collision_mask)
#         else:
#             cdprimitive.node().getIntoCollideMask(collision_mask)
#     else:
#         raise KeyError("Type should be from, into, or both.")


def change_cdmask(cdprimitive, collision_mask: BitMask32, action="new", type="both"):
    """
    :param cdprimitive: NodePath of CollisionNode
    :param action: "add", "remove", "new"
    :param type: 'from', 'into', 'both'
    :param collision_mask:
    :return:
    """

    def _change_cdmask(collision_mask, action, get_method_to_call, set_method_to_call):
        """
        internal function to reduce repeatition
        :param collision_mask:
        :param action:
        :param get_method_to_call:
        :param set_method_to_call:
        :return:
        """
        if action == "new":
            set_method_to_call(collision_mask)
        elif action == "add":
            current_cdmask = get_method_to_call()
            new_cdmask = current_cdmask | collision_mask
            set_method_to_call(new_cdmask)
        elif action == "remove":
            current_cdmask = get_method_to_call()
            new_cdmask = current_cdmask & ~collision_mask
            set_method_to_call(new_cdmask)
        else:
            raise KeyError("Action should be add, remove, or new.")

    if type == "both":
        get_method_name = "getCollideMask"
        set_method_name = "setCollideMask"
    elif type == "from":
        get_method_name = "getFromCollideMask"
        set_method_name = "setFromCollideMask"
    elif type == "into":
        get_method_name = "getIntoCollideMask"
        set_method_name = "setIntoCollideMask"
    else:
        raise KeyError("Type should be from, into, or both.")
    if cdprimitive.getName() == "cylinder":
        # child 0
        get_method_to_call = getattr(cdprimitive.getChild(0).node(), get_method_name, None)
        set_method_to_call = getattr(cdprimitive.getChild(0).node(), set_method_name)
        _change_cdmask(collision_mask, action, get_method_to_call, set_method_to_call)
        # child 1
        get_method_to_call = getattr(cdprimitive.getChild(1).node(), get_method_name, None)
        set_method_to_call = getattr(cdprimitive.getChild(1).node(), set_method_name)
        _change_cdmask(collision_mask, action, get_method_to_call, set_method_to_call)
        # child 2
        get_method_to_call = getattr(cdprimitive.getChild(2).node(), get_method_name, None)
        set_method_to_call = getattr(cdprimitive.getChild(2).node(), set_method_name)
        _change_cdmask(collision_mask, action, get_method_to_call, set_method_to_call)
    else:
        get_method_to_call = getattr(cdprimitive.node(), get_method_name, None)
        set_method_to_call = getattr(cdprimitive.node(), set_method_name)
        _change_cdmask(collision_mask, action, get_method_to_call, set_method_to_call)


def update_pose(cdprimitive, objcm):
    """
    update panda3d collision nodepath using the pos and quat of objcm.pdndp
    :param cdprimitive:
    :param objcm:
    :return:
    author: weiwei
    date: 20230815
    """
    cdprimitive.setMat(objcm.pdndp.getMat())


def toggle_show_collision_node(cdprimitive, toggle_value=True):
    """
    :param cdprimitive:
    :param is_show:
    :return:
    """
    if cdprimitive.getName() == "cylinder":
        if toggle_value:
            cdprimitive.getChild(0).show()
            cdprimitive.getChild(1).show()
            cdprimitive.getChild(2).show()
        else:
            cdprimitive.getChild(0).hide()
            cdprimitive.getChild(1).hide()
            cdprimitive.getChild(2).hide()
    else:
        if toggle_value:
            cdprimitive.show()
        else:
            cdprimitive.hide()


def gen_box_pdcndp(trm_model, ex_radius=0.01):
    """
    :param obstacle:
    :return:
    author: weiwei
    date: 20180811
    """
    obb = trm_model.obb_bound
    sides = obb.extents / 2.0 + ex_radius
    collision_primitive = CollisionBox(center=LPoint3(0, 0, 0), x=sides[0], y=sides[1], z=sides[2])
    pdcnd = CollisionNode("auto")
    pdcnd.addSolid(collision_primitive)
    pdcnd.setTransform(TransformState.makeMat(da.npmat4_to_pdmat4(obb.homomat)))
    cdprimitive = NodePath(pdcnd)
    cdprimitive.setName("box")
    return cdprimitive


def gen_capsule_pdcndp(trm_model, ex_radius=0.01):
    """
    :param trm_model:
    :param radius:
    :return:
    author: weiwei
    date: 20230816
    """
    cyl = trm_model.cyl_bound
    collision_primitive = CollisionCapsule(a=LPoint3(0, 0, -cyl.height / 2),
                                           db=LPoint3(0, 0, cyl.height / 2),
                                           radius=cyl.radius + ex_radius)
    pdcnd = CollisionNode("auto")
    pdcnd.addSolid(collision_primitive)
    pdcnd.setTransform(TransformState.makeMat(da.npmat4_to_pdmat4(cyl.homomat)))
    cdprimitive = NodePath(pdcnd)
    cdprimitive.setName("capsule")
    return cdprimitive


def gen_cyl_pdcndp(trm_model, ex_radius=0.01):
    """
    approximate cylinder using 3 boxes (rotate around central cylinderical axis)
    :param trm_model:
    :param name:
    :param radius:
    :return:
    author: weiwei
    date: 20230819
    """
    n_vedge = 6  # must be even number
    angles = np.radians(np.linspace(start=0, stop=180, num=n_vedge // 2, endpoint=False))
    cyl = trm_model.cyl_bound
    x_side = cyl.radius + ex_radius
    collision_primitive = CollisionBox(center=LPoint3(0, 0, 0),
                                       x=x_side,
                                       y=math.tan(angles[1] / 2) * x_side,
                                       z=cyl.height / 2.0)
    pdcndp = NodePath("cylinder")
    for angle in angles:
        homomat = cyl.homomat @ rm.homomat_from_posrot(rotmat=rm.rotmat_from_axangle(np.array([0, 0, 1]), angle))
        pdcnd = CollisionNode("auto")
        pdcnd.addSolid(collision_primitive)
        pdcnd.setTransform(TransformState.makeMat(da.npmat4_to_pdmat4(homomat)))
        pdcndp.attachNewNode(pdcnd)
    return pdcndp


def gen_surfaceballs_pdcnd(trm_mesh, radius=0.01):
    """
    :param obstacle:
    :return:
    author: weiwei
    date: 20180811
    """

    n_sample = int(math.ceil(trm_mesh.area / (radius * 0.3) ** 2))
    n_sample = 120 if n_sample > 120 else n_sample  # threshhold
    sample_data = trm_mesh.sample_surface(n_sample)
    pdcnd = CollisionNode("auto")
    for point in sample_data:
        pdcnd.addSolid(CollisionSphere(cx=point[0],
                                       cy=point[1],
                                       cz=point[2],
                                       radius=radius))
    cdprimitive = NodePath(pdcnd)
    cdprimitive.setName("surface_balls")
    return cdprimitive


def gen_pointcloud_pdcndp(trm_mesh, radius=0.02):
    """
    trm_mesh only have vertices that are considered to be point cloud
    :param obstacle:
    :return:
    author: weiwei
    date: 20191210
    """
    pdcnd = CollisionNode("auto")
    for sglpnt in trm_mesh.vertices:
        pdcnd.addSolid(CollisionSphere(cx=sglpnt[0], cy=sglpnt[1], cz=sglpnt[2], radius=radius))
    cdprimitive = NodePath(pdcnd)
    cdprimitive.setName("pointcloud")
    return cdprimitive


def gen_pdndp_wireframe(trm_model,
                        thickness=0.0001,
                        rgba=np.array([0, 0, 0, 1])):
    """
    gen wireframe
    :param trm_model: Trimesh
    :param thickness:
    :param rgba:
    :return: NodePath
    author: weiwei
    date: 20230815
    """
    # Create a set of line segments
    ls = LineSegs()
    ls.setThickness(thickness * da.M_TO_PIXEL)
    ls.setColor(*rgba)
    for line_seg in trm_model.edges:
        ls.moveTo(*trm_model.vertices[line_seg[0]])
        ls.drawTo(*trm_model.vertices[line_seg[1]])
    # Create and return a node with the segments
    ls_pdndp = NodePath(ls.create())
    ls_pdndp.setTransparency(TransparencyAttrib.MDual)
    ls_pdndp.setLightOff()
    return ls_pdndp


def gen_box_from_pdndp(pdndp, ex_radius=0.01):
    """
    deprecated 20230816
    :param obstacle:
    :return:
    author: weiwei
    date: 20180811
    """
    bottom_left, top_right = pdndp.getTightBounds()  # bug 20230816, four corners returned
    center = (bottom_left + top_right) / 2.0
    # enlarge the bounding box
    bottom_left -= (bottom_left - center).normalize() * ex_radius
    top_right += (top_right - center).normalize() * ex_radius
    collision_primitive = CollisionBox(min=bottom_left, max=top_right)
    pdcnd = CollisionNode("auto")
    pdcnd.addSolid(collision_primitive)
    pdcndp = NodePath(pdcnd)
    pdcndp.setName("box")
    return pdcndp


def gen_cylinder_from_pdndp(pdndp, ex_radius=0.01):
    """
    deprecated 20230816; this function approximated a cylinder using boxes
    :param trimeshmodel:
    :param name:
    :param radius:
    :return:
    author: weiwei
    date: 20200108
    """
    bottom_left, top_right = pdndp.getTightBounds()
    center = (bottom_left + top_right) / 2.0
    # enlarge the bounding box
    bottomleft_adjustvec = bottom_left - center
    bottomleft_adjustvec[2] = 0
    bottomleft_adjustvec.normalize()
    bottom_left += bottomleft_adjustvec * ex_radius
    topright_adjustvec = top_right - center
    topright_adjustvec[2] = 0
    topright_adjustvec.normalize()
    top_right += topright_adjustvec * ex_radius
    bottomleft_pos = da.pdvec3_to_npvec3(bottom_left)
    topright_pos = da.pdvec3_to_npvec3(top_right)
    pdcnd = CollisionNode("auto")
    for angle in np.nditer(np.linspace(math.pi / 10, math.pi * 4 / 10, num=4)):
        ca = math.cos(angle)
        sa = math.sin(angle)
        new_bottomleft_pos = np.array([bottomleft_pos[0] * ca, bottomleft_pos[1] * sa, bottomleft_pos[2]])
        new_topright_pos = np.array([topright_pos[0] * ca, topright_pos[1] * sa, topright_pos[2]])
        new_bottomleft = da.npvec3_to_pdvec3(new_bottomleft_pos)
        new_topright = da.npvec3_to_pdvec3(new_topright_pos)
        pdcnd.addSolid(CollisionBox(min=new_bottomleft, max=new_topright))
    pdcndp = NodePath(pdcnd)
    pdcndp.setName("cylinder")
    return pdcndp


def gen_polygons_pdcnd_from_pdndp(pdndp):
    """
    deprecated 20230816; Panda3D does not support collision detection between CollisionPolygons
    :param trimeshmodel:
    :param name:
    :return:
    author: weiwei
    date: 20210204
    """
    pdcnd = CollisionNode("auto")
    # counter = 0
    for geom in pdndp.findAllMatches('**/+GeomNode'):
        geom_node = geom.node()
        for g in range(geom_node.getNumGeoms()):
            geom = geom_node.getGeom(g).decompose()
            vdata = geom.getVertexData()
            vreader = GeomVertexReader(vertex_data=vdata, name='vertex')
            for p in range(geom.getNumPrimitives()):
                prim = geom.getPrimitive(p)
                for p2 in range(prim.getNumPrimitives()):
                    s = prim.getPrimitiveStart(p2)
                    e = prim.getPrimitiveEnd(p2)
                    v = []
                    for vi in range(s, e):
                        vreader.setRow(prim.getVertex(vi))
                        v.append(vreader.getData3f())
                    pdcnd.addSolid(CollisionPolygon(*v))
    pdcndp = NodePath(pdcnd)
    pdcndp.setName("polygons")
    return pdcndp


def is_collided(objcm_list0, objcm_list1, toggle_contacts=False):
    """
    detect the collision between collision models
    :param: objcm_list0, a single collision model or a list of collision models
    :param: objcm_list1
    :param toggle_contactpoints: True default
    :return:
    author: weiwei
    date: 20190312osaka, 20201214osaka
    """
    if not isinstance(objcm_list0, list):
        objcm_list0 = [objcm_list0]
    if not isinstance(objcm_list1, list):
        objcm_list1 = [objcm_list1]
    cd_trav = CollisionTraverser()
    cd_handler = CollisionHandlerQueue()
    tgt_pdndp = NodePath("collision pdndp")
    for objcm in objcm_list0:
        objcm.attach_cdprimitive_to(tgt_pdndp)
        cd_trav.addCollider(collider=copy_cdprimitive_attach_to(objcm, tgt_pdndp), handler=cd_handler)
    for objcm in objcm_list1:
        copy_cdprimitive_attach_to(objcm, tgt_pdndp)
    cd_trav.traverse(tgt_pdndp)
    if cd_handler.getNumEntries() > 0:
        if toggle_contacts:
            contact_points = np.asarray([da.pdvec3_to_npvec3(cd_entry.getSurfacePoint(base.render)) for cd_entry in
                                         cd_handler.getEntries()])
            return (True, contact_points)
        else:
            return True
    else:
        return (False, np.asarray([])) if toggle_contacts else False


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
    objcm = cm.CollisionModel(objpath, cdprimitive_type='polygons')
    objcm.set_rgba(np.array([.2, .5, 0, 1]))
    objcm.set_pos(np.array([.01, .01, .01]))
    objcm.attach_to(base)
    objcm.show_cdprimit()
    objcmlist = []
    for i in range(100):
        objcmlist.append(
            cm.CollisionModel(os.path.join(basis.__path__[0], 'objects', 'housing.stl'), cdprimitive_type='box'))
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
    # objcm1 = mcm.CollisionModel(objpath, cdprimitive_type='polygons')
    # # pos = np.array([[-0.5, -0.82363909, 0.2676166, -0.00203699],
    # #                     [-0.86602539, 0.47552824, -0.1545085, 0.01272306],
    # #                     [0., -0.30901703, -0.95105648, 0.12604253],
    # #                     [0., 0., 0., 1.]])
    # pos = np.array([[ 1.00000000e+00,  2.38935501e-16,  3.78436685e-17, -7.49999983e-03],
    #                     [ 2.38935501e-16, -9.51056600e-01, -3.09017003e-01,  2.04893537e-02],
    #                     [-3.78436685e-17,  3.09017003e-01, -9.51056600e-01,  1.22025304e-01],
    #                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # objcm1.set_homomat(pos)
    # objcm1.set_rgba([1, 1, .3, .2])
    #
    # objpath = os.path.join(basis.__path__[0], 'objects', 'tubebig.stl')
    # objcm2 = mcm.CollisionModel(objpath, cdprimitive_type='polygons')
    # objcm2.set_rgba([1, 1, .3, .2])
    # iscollided, contact_points = is_collided(objcm1, objcm2, toggle_contacts=True)
    # objcm1.show_cdmesh()
    # objcm2.show_cdmesh()
    # objcm1.attach_to(base)
    # objcm2.attach_to(base)
    # print(iscollided)
    # for ct_pnt in contact_points:
    #     mgm.gen_sphere(ct_pnt, major_radius=.001).attach_to(base)
    # base.run()
