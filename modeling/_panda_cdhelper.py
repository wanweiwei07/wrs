# primitive collision detection helper
import math
import numpy as np
import basis.dataadapter as da
from panda3d.core import NodePath, CollisionNode, CollisionTraverser, CollisionHandlerQueue, BitMask32
from panda3d.core import CollisionBox, CollisionSphere

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


def is_collided(objcm_list0, objcm_list1, toggleplot=False):
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
    if toggleplot:
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
        return True
    else:
        return False


if __name__ == '__main__':
    import os
    import time
    import basis
    import numpy as np
    import modeling.collisionmodel as cm
    import visualization.panda.world as wd

    base = wd.World(campos=[.7, .7, .7], lookatpos=[0, 0, 0])
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    objcm = cm.CollisionModel(objpath)
    objcm.set_rgba(np.array([.2, .5, 0, 1]))
    objcm.set_pos(np.array([.01, .01, .01]))
    objcm.attach_to(base)
    objcm.show_cdprimit()
    objcmlist = []
    for i in range(100):
        objcmlist.append(cm.CollisionModel(os.path.join(basis.__path__[0], 'objects', 'housing.stl')))
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