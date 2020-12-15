import basis.dataadapter as da
from panda3d.core import NodePath, CollisionTraverser, CollisionHandlerQueue, BitMask32


class CollisionChecker(object):
    """
    A fast collision checker that allows maximum 32 collision pairs
    author: weiwei
    date: 20201214osaka
    """

    def __init__(self, name="auto"):
        self.ctrav = CollisionTraverser()
        self.chan = CollisionHandlerQueue()
        self.np = NodePath(name)
        self.nbitmask = 0 # 0-28

        self.lnks_inuse = [] # a list of collisionmodels for quick accessing the cd links

    def add_jlcobj(self, jlobj, lnk_idlist):
        for id in lnk_idlist:
            if jlobj.lnks[id]['cdprimit_cache'][1] is None:  # first time add
                jlobj.lnks[id]['cdprimit_cache'][1] = jlobj.lnks[id]['collisionmodel'].copy_cdnp_to(self.np)
        else:
            raise ValueError("The link is already added!")

    def add_objinhnd(self, external_cdprimit_cache, objcm):
        """
        :param external_cdprimit_cache: an empty list for receiving the [Boolean, cdprimit_cache] pair
        :param objcm:
        :return:
        """
        external_cdprimit_cache.append(False)
        external_cdprimit_cache.append(objcm.copy_cdnp_to(self.np))

    def delete_objinhnd(self, external_cdprimit_cache):
        """
        :param external_cdprimit_cache: a list that holds the [Boolean, cdprimit_cache] pair
        :param objcm:
        :return:
        """
        external_cdprimit_cache[1].removeNode()
        external_cdprimit_cache = []

    # def add_cdpair(self, jlobj0, lnk_idlist0, jlobj1, lnk_idlist1):
    #     """
    #     DEPRECATED, cannot handle links spanning multiple jlcobjects
    #     :param jlobj0:
    #     :param lnk_idlist0:
    #     :param jlobj1:
    #     :param lnk_idlist1:
    #     :return:
    #     """
    #     cdmask = BitMask32(2 ** len(self.cdpairs))
    #     for id in lnk_idlist0:
    #         if jlobj0.lnks[id]['cdprimit_cache'][1] is None: # first time add
    #             jlobj0.lnks[id]['cdprimit_cache'][1] = jlobj0.lnks[id]['collisionmodel'].copy_cdnp_to(self.np)
    #             jlobj0.lnks[id]['cdprimit_cache'][1].node().setFromCollideMask(cdmask)
    #             current_into_cdmask = jlobj0.lnks[id]['cdprimit_cache'][1].node().getIntoCollideMask()
    #             jlobj0.lnks[id]['cdprimit_cache'][1].node().setIntoCollideMask(current_into_cdmask & ~cdmask)
    #             self.ctrav.addCollider(jlobj0.lnks[id]['cdprimit_cache'][1], self.chan)
    #             self.lnks_inuse.append(jlobj0.lnks[id])
    #         else:
    #             current_from_cdmask = jlobj0.lnks[id]['cdprimit_cache'][1].node().getFromCollideMask()
    #             if current_from_cdmask == 0: # if was never added as collider
    #                 self.ctrav.addCollider(jlobj0.lnks[id]['cdprimit_cache'][1], self.chan)
    #             new_from_cdmask = current_from_cdmask | cdmask
    #             jlobj0.lnks[id]['cdprimit_cache'][1].node().setFromCollideMask(new_from_cdmask)
    #             current_into_cdmask = jlobj0.lnks[id]['cdprimit_cache'][1].node().getIntoCollideMask()
    #             jlobj0.lnks[id]['cdprimit_cache'][1].node().setIntoCollideMask(current_into_cdmask & ~new_from_cdmask)
    #     for id in lnk_idlist1:
    #         if jlobj1.lnks[id]['cdprimit_cache'][1] is None:
    #             jlobj1.lnks[id]['cdprimit_cache'][1] = jlobj1.lnks[id]['collisionmodel'].copy_cdnp_to(self.np)
    #             jlobj1.lnks[id]['cdprimit_cache'][1].node().setIntoCollideMask(cdmask)
    #             self.lnks_inuse.append(jlobj1.lnks[id])
    #         else:
    #             current_into_cdmask = jlobj1.lnks[id]['cdprimit_cache'][1].node().getIntoCollideMask()
    #             new_into_cdmask = current_into_cdmask | cdmask
    #             jlobj1.lnks[id]['cdprimit_cache'][1].node().setIntoCollideMask(new_into_cdmask)
    #     cdpair = {}
    #     cdpair['jlobj0'] = jlobj0
    #     cdpair['lnk_idlist0'] = lnk_idlist0
    #     cdpair['jlobj1'] = jlobj1
    #     cdpair['lnk_idlist1'] = lnk_idlist1
    #     self.cdpairs.append(cdpair)

    def add_self_cdpair(self, fromlist, intolist):
        """
        :param fromlist: [[jlcobj, lnk_idlist], ...]
        :param intolist: [[jlcobj, lnk_idlist], ...]
        :return:
        author: weiwei
        date: 20201214
        """
        cdmask = BitMask32(2 ** len(self.cdpairs))
        for jlc_idlist in fromlist:
            jlcobj, lnk_idlist = jlc_idlist
            for id in lnk_idlist:
                if jlcobj.lnks[id]['cdprimit_cache'][1] is None: # first time add
                    jlcobj.lnks[id]['cdprimit_cache'][1] = jlcobj.lnks[id]['collisionmodel'].copy_cdnp_to(self.np)
                    jlcobj.lnks[id]['cdprimit_cache'][1].node().setFromCollideMask(cdmask)
                    current_into_cdmask = jlcobj.lnks[id]['cdprimit_cache'][1].node().getIntoCollideMask()
                    jlcobj.lnks[id]['cdprimit_cache'][1].node().setIntoCollideMask(current_into_cdmask & ~cdmask)
                    self.ctrav.addCollider(jlcobj.lnks[id]['cdprimit_cache'][1], self.chan)
                    self.lnks_inuse.append(jlcobj.lnks[id])
                else:
                    current_from_cdmask = jlcobj.lnks[id]['cdprimit_cache'][1].node().getFromCollideMask()
                    if current_from_cdmask == 0: # if was never added as collider
                        self.ctrav.addCollider(jlcobj.lnks[id]['cdprimit_cache'][1], self.chan)
                    new_from_cdmask = current_from_cdmask | cdmask
                    jlcobj.lnks[id]['cdprimit_cache'][1].node().setFromCollideMask(new_from_cdmask)
                    current_into_cdmask = jlcobj.lnks[id]['cdprimit_cache'][1].node().getIntoCollideMask()
                    jlcobj.lnks[id]['cdprimit_cache'][1].node().setIntoCollideMask(current_into_cdmask & ~new_from_cdmask)
        for jlc_idlist in intolist:
            jlcobj, lnk_idlist = jlc_idlist
            for id in lnk_idlist:
                if jlcobj.lnks[id]['cdprimit_cache'][1] is None:
                    jlcobj.lnks[id]['cdprimit_cache'][1] = jlcobj.lnks[id]['collisionmodel'].copy_cdnp_to(self.np)
                    jlcobj.lnks[id]['cdprimit_cache'][1].node().setIntoCollideMask(cdmask)
                    self.lnks_inuse.append(jlcobj.lnks[id])
                else:
                    current_into_cdmask = jlcobj.lnks[id]['cdprimit_cache'][1].node().getIntoCollideMask()
                    new_into_cdmask = current_into_cdmask | cdmask
                    jlcobj.lnks[id]['cdprimit_cache'][1].node().setIntoCollideMask(new_into_cdmask)
        cdpair = {}
        cdpair['fromlist'] = fromlist
        cdpair['tolist'] = intolist
        self.cdpairs.append(cdpair)

    def add_cdpair(self, fromlist, intolist):
        """
        :param fromlist: [[bool, cdprimit_cache], ...]
        :param intolist: [[bool, cdprimit_cache], ...]
        :return:
        author: weiwei
        date: 20201215
        """
        cdmask = BitMask32(2 ** len(self.cdpairs))
        for cdprimit_cache in fromlist:
            current_from_cdmask = cdprimit_cache[1].node().getFromCollideMask()
            if current_from_cdmask == 0:  # if was never added as collider
                self.ctrav.addCollider(cdprimit_cache[1], self.chan)
            new_from_cdmask = current_from_cdmask | cdmask
            cdprimit_cache[1].node().setFromCollideMask(new_from_cdmask)
            current_into_cdmask = cdprimit_cache[1].node().getIntoCollideMask()
            cdprimit_cache[1].node().setIntoCollideMask(current_into_cdmask & ~new_from_cdmask)
        for cdprimit_cache in intolist:
            current_into_cdmask = cdprimit_cache[1].node().getIntoCollideMask()
            new_into_cdmask = current_into_cdmask | cdmask
            cdprimit_cache[1].node().setIntoCollideMask(new_into_cdmask)
        self.nbitmask += 1

    def is_selfcollided(self):
        for one_lnkcdmodel in self.lnks_inuse: # TODO global ik indicator
            if one_lnkcdmodel['cdprimit_cache'][0]: # need to update
                pos = one_lnkcdmodel['gl_pos']
                rotmat = one_lnkcdmodel['gl_rotmat']
                one_lnkcdmodel['cdprimit_cache'][1].setMat(da.npv3mat3_to_pdmat4(pos, rotmat))
                one_lnkcdmodel['cdprimit_cache'][0] = False # updated
                # print("From", one_lnkcdmodel['cdprimit_cache'][1].node().getFromCollideMask())
                # print("Into", one_lnkcdmodel['cdprimit_cache'][1].node().getIntoCollideMask())
        self.ctrav.traverse(self.np)
        if self.chan.getNumEntries() > 0:
            self.ctrav.showCollisions(base.render)
            return True
        else:
            return False

    def disable(self):
        """
        clear pairs and nodepath
        :return:
        """
        for one_lnkcdmodel in self.lnks_inuse:
            one_lnkcdmodel['cdprimit_cache'][1].removeNode()
            one_lnkcdmodel['cdprimit_cache'][1] = None
        pass


# if __name__ == '__main__':
#     cdmask1 = BitMask32(2**1)
#     cdmask2 = BitMask32(2**2)
#     cdmask3 = BitMask32(2**3)
#     print(cdmask1, cdmask2, cdmask1 | cdmask3)
