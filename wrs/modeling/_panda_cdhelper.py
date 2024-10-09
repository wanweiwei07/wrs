# primitive collision detection helper
# note: This script is not used by robot simulations, as collison checker is faster
import math
import copy
import numpy as np
from panda3d.core import NodePath, CollisionNode, CollisionTraverser, CollisionHandlerQueue, BitMask32, CollisionBox, \
    CollisionSphere, CollisionCapsule, CollisionPolygon, GeomVertexReader, LPoint3, TransformState, LineSegs, \
    TransparencyAttrib
import wrs.basis.robot_math as rm
import wrs.basis.data_adapter as da
import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm

BITMASK_EXT = BitMask32(2 ** 31)


def copy_cdprim(cmodel):
    return copy.deepcopy(cmodel.cdprim)


def copy_cdprim_attach_to(cmodel,
                          tgt_pdndp,
                          homomat=None,
                          clear_mask=False) -> NodePath:
    return_pdcndp = cmodel.copy_reference_cdprim()
    return_pdcndp.reparentTo(tgt_pdndp)
    if homomat is not None:
        return_pdcndp.setMat(da.npmat4_to_pdmat4(homomat))
    # if scale is not None
    # return_pdcndp.setScale(da.npvec3_to_pdvec3(scale)) # 20231117 scale is not supported
    if clear_mask:
        change_cdmask(return_pdcndp, BitMask32(0x00), action="new", type="both")
    return return_pdcndp


def detach_cdprim(cdprim):
    cdprim.removeNode()

def get_cdmask(cdprim, type = "from"):
    if type == "from":
        get_method_name = "getFromCollideMask"
    elif type == "into":
        get_method_name = "getIntoCollideMask"
    for child_pdndp in cdprim.getChildren():
        get_method_to_call = getattr(child_pdndp.node(), get_method_name)
    return get_method_to_call()


def change_cdmask(cdprim, collision_mask: BitMask32, action="new", type="both"):
    """
    :param cdprim: NodePath of CollisionNode
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
        change_cdmask(cdprim=cdprim, collision_mask=collision_mask, action=action, type="from")
        change_cdmask(cdprim=cdprim, collision_mask=collision_mask, action=action, type="into")
    else:
        if type == "from":
            get_method_name = "getFromCollideMask"
            set_method_name = "setFromCollideMask"
        elif type == "into":
            get_method_name = "getIntoCollideMask"
            set_method_name = "setIntoCollideMask"
        else:
            raise KeyError("Type should be from, into, or both.")
        for child_pdndp in cdprim.getChildren():
            get_method_to_call = getattr(child_pdndp.node(), get_method_name)
            set_method_to_call = getattr(child_pdndp.node(), set_method_name)
            _change_cdmask(collision_mask, action, get_method_to_call, set_method_to_call)


def update_pose(cdprim, cmodel):
    """
    update panda3d collision nodepath using the pos and quat of obj_cmodel.pdndp
    :param cdprim:
    :param cmodel:
    :return:
    author: weiwei
    date: 20230815
    """
    cdprim.setMat(cmodel.pdndp.getMat())


def toggle_show_collision_node(cdprim, toggle_show_on=True):
    """
    :param cdprim:
    :param is_show:
    :return:
    """
    if toggle_show_on:
        for child_pdndp in cdprim.getChildren():
            child_pdndp.show()
    else:
        for child_pdndp in cdprim.getChildren():
            child_pdndp.hide()


# ==================================
# generate cdprimitives from trimesh
# ==================================

def gen_aabb_box_pdcndp(trm_model, ex_radius=0.01):
    """
    :param obstacle:
    :return:
    author: weiwei
    date: 20180811, 20240305
    """
    aabb = trm_model.aabb_bound
    sides = aabb.extents / 2.0 + ex_radius
    collision_primitive = CollisionBox(center=LPoint3(0, 0, 0), x=sides[0], y=sides[1], z=sides[2])
    pdcnd = CollisionNode("aabb_box_cnode")
    pdcnd.addSolid(collision_primitive)
    pdcnd.setTransform(TransformState.makeMat(da.npmat4_to_pdmat4(aabb.homomat)))
    cdprim = NodePath("aabb_box")
    cdprim.attachNewNode(pdcnd)
    return cdprim


def gen_obb_box_pdcndp(trm_model, ex_radius=0.01):
    """
    :param obstacle:
    :return:
    author: weiwei
    date: 20180811, 20240305
    """
    obb = trm_model.obb_bound
    sides = obb.extents / 2.0 + ex_radius
    collision_primitive = CollisionBox(center=LPoint3(0, 0, 0), x=sides[0], y=sides[1], z=sides[2])
    pdcnd = CollisionNode("obb_box_cnode")
    pdcnd.addSolid(collision_primitive)
    pdcnd.setTransform(TransformState.makeMat(da.npmat4_to_pdmat4(obb.homomat)))
    cdprim = NodePath("obb_box")
    cdprim.attachNewNode(pdcnd)
    return cdprim


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
    pdcnd = CollisionNode("capsule_cnode")
    pdcnd.addSolid(collision_primitive)
    pdcnd.setTransform(TransformState.makeMat(da.npmat4_to_pdmat4(cyl.homomat)))
    cdprim = NodePath("capsule")
    cdprim.attachNewNode(pdcnd)
    return cdprim


def gen_cylinder_pdcndp(trm_model, ex_radius=0.01):
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
    cdprim = NodePath("cylinder")
    for i, angle in enumerate(angles):
        homomat = cyl.homomat @ rm.homomat_from_posrot(rotmat=rm.rotmat_from_axangle(np.array([0, 0, 1]), angle))
        pdcnd = CollisionNode("cylinder" + f"_cnode_{i}")
        pdcnd.addSolid(collision_primitive)
        pdcnd.setTransform(TransformState.makeMat(da.npmat4_to_pdmat4(homomat)))
        cdprim.attachNewNode(pdcnd)
    return cdprim


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
    pdcnd = CollisionNode("surface_balls_cnode")
    for point in sample_data:
        pdcnd.addSolid(CollisionSphere(cx=point[0],
                                       cy=point[1],
                                       cz=point[2],
                                       radius=radius))
    cdprim = NodePath("surface_balls")
    cdprim.attachNewNode(pdcnd)
    return cdprim


def gen_pointcloud_pdcndp(trm_mesh, radius=0.02):
    """
    trm_mesh only have vertices that are considered to be point cloud
    :param obstacle:
    :return:
    author: weiwei
    date: 20191210
    """
    pdcnd = CollisionNode("pointcloud_cnode")
    for point in trm_mesh.vertices:
        pdcnd.addSolid(CollisionSphere(cx=point[0], cy=point[1], cz=point[2], radius=radius))
    cdprim = NodePath("pointcloud")
    cdprim.attachNewNode(pdcnd)
    return cdprim


# ========================================
# generate wireframe NodePath from trimesh
# ========================================

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


# ==========================
# collision detection helper
# ==========================

# def is_collided(cmodel_list0, cmodel_list1, toggle_contacts=False):
#     """
#     detect the collision between collision models
#     :param: cmodel_list0, a single collision model or a list of collision models
#     :param: cmodel_list1
#     :param toggle_contacts: True default
#     :return:
#     author: weiwei
#     date: 20190312osaka, 20201214osaka, 20231123
#     """
#     if not isinstance(cmodel_list0, list):
#         cmodel_list0 = [cmodel_list0]
#     if not isinstance(cmodel_list1, list):
#         cmodel_list1 = [cmodel_list1]
#     cd_trav = CollisionTraverser()
#     cd_handler = CollisionHandlerQueue()
#     tgt_pdndp = NodePath("collision pdndp")
#     # attach to collision tree, change bitmasks, and add colliders
#     cprim_list0=[]
#     for cmodel in cmodel_list0:
#         cprim_list0.append(copy_cdprim_attach_to(cmodel, tgt_pdndp, homomat=cmodel.homomat, clear_mask=True))
#         change_cdmask(cprim_list0[-1], BITMASK_EXT, action="remove", type="into")
#         for child_pdcnd in cprim_list0[-1].getChildren():
#             cd_trav.addCollider(collider=child_pdcnd, handler=cd_handler)
#     cprim_list1=[]
#     for cmodel in cmodel_list1:
#         cprim_list1.append(copy_cdprim_attach_to(cmodel, tgt_pdndp, homomat=cmodel.homomat, clear_mask=True))
#     # perform collision detection
#     cd_trav.traverse(tgt_pdndp)
#     # detach from collision tree, change bitmasks, and remove colliders
#     for cdprim in cprim_list0:
#         for child_pdcnd in cdprim.getChildren():
#             cd_trav.removeCollider(child_pdcnd)
#         detach_cdprim(cdprim)
#     for cdprim in cprim_list1:
#         detach_cdprim(cdprim)
#     if cd_handler.getNumEntries() > 0:
#         if toggle_contacts:
#             contact_points = np.asarray([da.pdvec3_to_npvec3(cd_entry.getSurfacePoint(base.render)) for cd_entry in
#                                          cd_handler.getEntries()])
#             print(contact_points)
#             return True, contact_points
#         else:
#             return True
#     else:
#         return False, np.asarray([]) if toggle_contacts else False

def is_collided(cmodel_list0, cmodel_list1, toggle_contacts=False):
    """
    detect the collision between collision models
    :param: cmodel_list0, a single collision model or a list of collision models
    :param: cmodel_list1
    :param toggle_contacts: True default
    :return:
    author: weiwei
    date: 20190312osaka, 20201214osaka, 20231123
    """
    if not isinstance(cmodel_list0, list):
        cmodel_list0 = [cmodel_list0]
    if not isinstance(cmodel_list1, list):
        cmodel_list1 = [cmodel_list1]
    cd_trav = CollisionTraverser()
    cd_handler = CollisionHandlerQueue()
    tgt_pdndp = NodePath("collision pdndp")
    # attach to collision tree, change bitmasks, and add colliders
    for cmodel in cmodel_list0:
        cdprim = cmodel.attach_cdprim_to(tgt_pdndp)
        change_cdmask(cdprim, BITMASK_EXT, action="remove", type="into")
        for child_pdcnd in cdprim.getChildren():
            cd_trav.addCollider(collider=child_pdcnd, handler=cd_handler)
    for cmodel in cmodel_list1:
        cmodel.attach_cdprim_to(tgt_pdndp)
    # perform collision detection
    cd_trav.traverse(tgt_pdndp)
    # detach from collision tree, change bitmasks, and remove colliders
    for cmodel in cmodel_list0:
        cmodel.detach_cdprim()
        change_cdmask(cmodel.cdprim, BITMASK_EXT, action="add", type="into")
        for child_pdcnd in cmodel.cdprim.getChildren():
            cd_trav.removeCollider(child_pdcnd)
    for cmodel in cmodel_list1:
        cmodel.detach_cdprim()
    if cd_handler.getNumEntries() > 0:
        if toggle_contacts:
            contact_points = np.asarray([da.pdvec3_to_npvec3(cd_entry.getSurfacePoint(base.render)) for cd_entry in
                                         cd_handler.getEntries()])
            print(contact_points)
            return True, contact_points
        else:
            return True
    else:
        return False, np.asarray([]) if toggle_contacts else False


# *** deprecated ***
# ===================================
# generate cdprimitives from NodePath
# ===================================

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


if __name__ == '__main__':
    import os
    import time
    import numpy as np
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[.7, .7, .7], lookat_pos=[0, 0, 0])
    file_path = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    cmodel = mcm.CollisionModel(file_path, cdprim_type=mc.CDPrimType.CYLINDER)
    cmodel.rgba = np.array([.2, .5, 0, 1])
    cmodel.pos = np.array([.1, .01, .01])
    cmodel.attach_to(base)
    cmodel.show_cdprim()

    cmodel_list = []
    for i in range(100):
        cmodel_list.append(
            mcm.CollisionModel(os.path.join(basis.__path__[0], 'objects', 'housing.stl'),
                               cdprim_type=mc.CDPrimType.AABB))
        cmodel_list[-1].pos = np.random.random_sample((3,))
        cmodel_list[-1].rgba = np.array([1, .5, 0, 1])
        cmodel_list[-1].attach_to(base)
        cmodel_list[-1].show_cdprim()

    tic = time.time()
    result, contacts = is_collided(cmodel, cmodel_list, toggle_contacts=True)
    toc = time.time()
    time_cost = toc - tic
    print(time_cost)
    for cpoint in contacts:
        mgm.gen_sphere(pos=cpoint, radius=.001).attach_to(base)

    base.run()
