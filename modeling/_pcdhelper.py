# primitive collision detection helper

from panda3d.core import NodePath, CollisionNode, CollisionTraverser, CollisionHandlerQueue, BitMask32

def is_cmcm_collided(objcm1, objcm2, toggleplot = False):
    """
    detect the collision between collision models
    :return: True or False
    author: weiwei
    date: 20190312osaka, 20201214osaka
    """
    if toggleplot:
        objcm1.show_cdprimit()
        objcm2.show_cdprimit()
    tmpnp = NodePath("collision nodepath")
    ctrav = CollisionTraverser()
    chan = CollisionHandlerQueue()
    ctrav.addCollider(objcm1.copy_cdnp_to(tmpnp, clearmask=False), chan)
    objcm2.copy_cdnp_to(tmpnp, clearmask=False)
    ctrav.traverse(tmpnp)
    if chan.getNumEntries() > 0:
        return True
    else:
        return False

def is_cmcmlist_collided(objcm, objcmlist, toggleplot = False):
    """
    detect the collision between a collision model and a collision model list
    :return: True or False
    author: weiwei
    date: 20190312osaka, 20201214osaka
    """
    if toggleplot:
        objcm.show_cdprimit()
        for one_objcm in objcmlist:
            one_objcm.show_cdprimit()
    tmpnp = NodePath("collision nodepath")
    ctrav = CollisionTraverser()
    chan = CollisionHandlerQueue()
    ctrav.addCollider(objcm.copy_cdnp_to(tmpnp, clearmask=False), chan)
    for objcm2 in objcmlist:
        objcm2.copy_cdnp_to(tmpnp, clearmask=False)
    ctrav.traverse(tmpnp)
    if chan.getNumEntries() > 0:
        return True
    else:
        return False


def is_cmlistcmlist_collided(objcmlist0, objcmlist1, toggleplot = False):
    """
    detect the collision between two collision model lists
    :return: True or False
    author: weiwei
    date: 20190422osaka, 20201214osaka
    """
    if toggleplot:
        for one_objcm in objcmlist0:
            one_objcm.show_cdprimit()
        for one_objcm in objcmlist1:
            one_objcm.show_cdprimit()
    tmpnp = NodePath("collision nodepath")
    ctrav = CollisionTraverser()
    chan = CollisionHandlerQueue()
    for one_objcm in objcmlist0:
        ctrav.addCollider(one_objcm.copy_cdnp_to(tmpnp, clearmask=False), chan)
    tmpnp = NodePath("collision nodepath")
    for objcm2 in objcmlist:
        objcm2.copy_cdnp_to(tmpnp, clearmask=False)
    ctrav.traverse(tmpnp)
    if chan.getNumEntries() > 0:
        return True
    else:
        return False

if __name__ == '__main__':
    import time
    import numpy as np
    import modeling.collisionmodel as cm
    import visualization.panda.world as wd

    base = wd.World(campos=[.7,.7,.7], lookatpos=[0, 0, 0])
    objcm = cm.CollisionModel("./objects/bunnysim.stl")
    objcm.set_color(np.array([.2,.5,0,1]))
    objcm.set_pos(np.array([.01,.01,.01]))
    objcm.attach_to(base)
    objcm.show_cdprimit()
    objcmlist = []
    for i in range(100):
        objcmlist.append(cm.CollisionModel("./objects/housing.stl"))
        objcmlist[-1].set_pos(np.random.random_sample((3,)))
        objcmlist[-1].set_color(np.array([1,.5,0,1]))
        objcmlist[-1].attach_to(base)
        objcmlist[-1].show_cdprimit()

    tic = time.time()
    result = is_cmcmlist_collided(objcm, objcmlist)
    toc = time.time()
    time_cost = toc-tic
    print(time_cost)
    print(result)
    # tic = time.time()
    # is_cmcmlist_collided2(objcm, objcmlist)
    # toc = time.time()
    # time_cost = toc-tic
    # print(time_cost)
    base.run()