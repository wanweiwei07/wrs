# primitive collision detection helper

from panda3d.core import NodePath, CollisionTraverser, CollisionHandlerQueue, BitMask32

def is_cmcm_collided(objcm1, objcm2, toggleplot = False):
    """
    detect the collision between collision models
    :return: True or False
    author: weiwei
    date: 20190312osaka
    """
    oocnp = NodePath("collision nodepath")
    obj1cnp = objcm1.copy_cdnp_to(oocnp)
    obj2cnp = objcm2.copy_cdnp_to(oocnp)
    if toggleplot:
        oocnp.reparentTo(base.render)
        obj1cnp.show()
        obj2cnp.show()
    ctrav = CollisionTraverser()
    chan = CollisionHandlerQueue()
    ctrav.addCollider(obj1cnp, chan)
    ctrav.traverse(oocnp)
    if chan.getNumEntries() > 0:
        return True
    else:
        return False

def is_cmcmlist_collided(objcm, objcmlist, toggleplot = False):
    """
    detect the collision between a collision model and a collision model list
    :return: True or False
    author: weiwei
    date: 20190312osaka
    """
    oocnp = NodePath("collision nodepath")
    objcnp = objcm.copy_cdnp_to(oocnp)
    objcnplist = []
    for objcm2 in objcmlist:
        objcnplist.append(objcm2.copy_cdnp_to(oocnp))
    if toggleplot:
        oocnp.reparentTo(base.render)
        objcnp.show()
        for obj2cnp in objcnplist:
            obj2cnp.show()
    ctrav = CollisionTraverser()
    chan = CollisionHandlerQueue()
    ctrav.addCollider(objcnp, chan)
    ctrav.traverse(oocnp)
    if chan.getNumEntries() > 0:
        return True
    else:
        return False

def is_cmlistcmlist_collided(objcmlist0, objcmlist1, toggleplot = False):
    """
    detect the collision between two collision model lists
    :return: True or False
    author: weiwei
    date: 20190422osaka
    """
    oocnp = NodePath("collision nodepath")
    obj0cnplist = []
    for objcm0 in objcmlist0:
        obj0cnplist.append(objcm0.copy_cdnp_to(oocnp))
    obj1cnplist = []
    for objcm1 in objcmlist1:
        obj1cnplist.append(objcm1.copy_cdnp_to(oocnp))
    if toggleplot:
        oocnp.reparentTo(base.render)
        for obj0cnp in obj0cnplist:
            obj0cnp.show()
        for obj1cnp in obj1cnplist:
            obj1cnp.show()
    ctrav = CollisionTraverser()
    chan = CollisionHandlerQueue()
    for obj0cnp in obj0cnplist:
        obj0cnp.node().setFromCollideMask(BitMask32(0x1))
        obj0cnp.setCollideMask(BitMask32(0x2))
        ctrav.addCollider(obj0cnp, chan)
    ctrav.traverse(oocnp)
    if chan.getNumEntries() > 0:
        return True
    else:
        return False

if __name__ == '__main__':
    import numpy as np
    import modeling.collisionmodel as cm
    import visualization.panda.world as wd

    base = wd.World(camp=[.7,.7,.7], lookatpos=[0, 0, 0])
    objcm = cm.CollisionModel("./objects/bunnysim.stl")
    objcm.setcolor(np.array([.2,.5,0,1]))
    objcm.setpos(np.array([.01,-.01,.01]))
    objcm.reparent_to(base.render)
    objcm.showcn()
    objcm2 = cm.CollisionModel("./objects/housing.stl")
    objcm2.setcolor(np.array([1,.5,0,1]))
    objcm2.reparent_to(base.render)
    objcm2.showcn()

    print(is_cmcm_collided(objcm, objcm2))
    base.run()