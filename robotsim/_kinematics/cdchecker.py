import basis.robotmath as rm
from panda3d.core import NodePath, CollisionTraverser, CollisionHandlerQueue, BitMask32


class CollisionChecker(object):
    """
    This class handles collision detections for both primitives and meshes
    It also accept cd pairs from differnt JLChain Instances.
    """

    def __init__(self, name="auto_collisionchecker"):
        self.hostpdnp = NodePath(name)
        self.cdpairs = []

    def add_cdpair(self, jlobj0, idlist0=[], jlobj1, idlist1=[]):
        cdpair = {}
        cdpair['jlobj0'] = jlobj0
        cdpair['idlist0'] = idlist0
        for id in idlist0:
            if jlobj0.lnks[id]['cdnp_cache'] is None:
                pos = jlobj0.lnks[id]['gl_pos']
                rotmat = jlobj0.lnks[id]['gl_rotmat']
                this_collisionmodel = jlobj0.lnks[id]['collisionmodel']
                jlobj0.lnks[id]['cdnp_cache'] = this_collisionmodel.copy_cdnp_to(self.hospdnp,
                                                                                 rm.homomat_from_posrot(pos, rotmat))
        cdpair['jlobj1'] = jlobj1
        cdpair['idlist1'] = idlist1
        for id in idlist1:
            if jlobj1.lnks[id]['cdnp_cache'] is None:
                pos = jlobj1.lnks[id]['gl_pos']
                rotmat = jlobj1.lnks[id]['gl_rotmat']
                this_collisionmodel = jlobj1.lnks[id]['collisionmodel']
                jlobj1.lnks[id]['cdnp_cache'] = this_collisionmodel.copy_cdnp_to(self.hospdnp,
                                                                                 rm.homomat_from_posrot(pos, rotmat))
        self.cdpairs.append(cdpair)

    def is_collided_cdprimit(self):
        for cdpair in self.cdpairs:
            jlobj0 = cdpair['jlobj0']
            idlist0 = cdpair['idlist0']
            jlobj1 = cdpair['jlobj1']
            idlist1 = cdpair['idlist1']
            np0 = NodePath("collision nodepath")
            cnp0list = []
            for one_cdlnkid in idlist0:
                this_collisionmodel = jlobj0.lnks[one_cdlnkid]['collisionmodel']
                if this_collisionmodel is not None:
                    if jlobj0.lnks[one_cdlnkid]['cdprimit_cache'] is None:
                        pos = jlobj0.lnks[one_cdlnkid]['gl_pos']
                        rotmat = jlobj0.lnks[one_cdlnkid]['gl_rotmat']
                        jlobj0.lnks[one_cdlnkid]['cdprimit_cache'] = this_collisionmodel.copy_cdnd()
                        cnp0list.append(this_collisionmodel.copy_cdnp_to(np0, rm.homomat_from_posrot(pos, rotmat)))
                    else:

            np1 = NodePath("collision nodepath")
            cnp1list = []
            for onelcdp in self.linkcdpairs[1]:
                this_collisionmodel = self.lnks[onelcdp]['collisionmodel']
                if this_collisionmodel is not None:
                    pos = self.lnks[onelcdp]['gl_pos']
                    rotmat = self.lnks[onelcdp]['gl_rotmat']
                    cnp1list.append(this_collisionmodel.copy_cdnp_to(np1, rm.homomat_from_posrot(pos, rotmat)))
            if toggleplot:
                # TODO set np0 and np1 to the same random color for better examination
                np0.reparentTo(base.render)
                np1.reparentTo(base.render)
                for obj0cnp in cnp0list:
                    obj0cnp.show()
                for obj1cnp in cnp1list:
                    obj1cnp.show()
            ctrav = CollisionTraverser()  # TODO change them to member function to avoid repeated initialization
            chan = CollisionHandlerQueue()
            for one_cnp0 in cnp0list:
                ctrav.addCollider(one_cnp0, chan)  # clearColliders
            ctrav.traverse(np1)
            if chan.getNumEntries() > 0:
                return True
            else:
                return False
