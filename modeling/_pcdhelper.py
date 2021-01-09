# primitive collision detection helper

from panda3d.core import NodePath, CollisionNode, CollisionTraverser, CollisionHandlerQueue, BitMask32


def is_cdprimit2cdprimit_collided(objcm_list0, objcm_list1, toggleplot=False):
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
    result = is_cdprimit2cdprimit_collided(objcm, objcmlist)
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