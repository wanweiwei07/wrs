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
        self.nbitmask = 0 # capacity 1-30
        self._bitmask_ext = BitMask32(2**31) # 31 is prepared for cd with external objects
        self.all_cdelements = [] # a list of collisionmodels for quick accessing the cd elements (cdlnks/cdobjs)

    def add_cdlnks(self, jlcobj, lnk_idlist):
        """
        The collision node of the given links will be attached to self.np, but their collision bitmask will be cleared
        When the a robot is treated as an obstacle by another robot, the IntoCollideMask of its all_cdelements will be
        set to BitMask32(2**31), so that the other robot can compare its active_cdelements with the all_cdelements.
        :param jlcobj:
        :param lnk_idlist:
        :return:
        author: weiwei
        date: 20201216toyonaka
        """
        for id in lnk_idlist:
            if jlcobj.lnks[id]['cdprimit_cache'][1] is None:  # first time add
                jlcobj.lnks[id]['cdprimit_cache'][1] = jlcobj.lnks[id]['collisionmodel'].copy_cdnp_to(self.np, clearmask=True)
                self.ctrav.addCollider(jlcobj.lnks[id]['cdprimit_cache'][1], self.chan)
                self.all_cdelements.append(jlcobj.lnks[id])
            else:
                raise ValueError("The link is already added!")

    def set_active_cdlnks(self, activelist):
        """
        The specified collistion links will be used for collision detection with external obstacles
        :param activelist: essentially a from list like [[bool, cdprimit_cache], ...],
                           the correspondent tolist will be set online in cd functions;
                           All elements in self.all_cdnlks will be used if None
        :return:
        author: weiwei
        date: 20201216toyonaka
        """
        for cdprimit_cache in activelist:
            if cdprimit_cache[1] is None:
                raise ValueError("The link needs to be added to collider using the add_cdlnks function first!")
            cdprimit_cache[1].node().setFromCollideMask(self._bitmask_ext)
            self.ctrav.addCollider(cdprimit_cache[1], self.chan)

    def set_cdpair(self, fromlist, intolist):
        """
        The given collision pair will be used for self collision detection
        :param fromlist: [[bool, cdprimit_cache], ...]
        :param intolist: [[bool, cdprimit_cache], ...]
        :return:
        author: weiwei
        date: 20201215
        """
        if self.nbitmask >= 30:
            raise ValueError("Too many collision pairs! Maximum: 29")
        cdmask = BitMask32(2 ** self.nbitmask)
        for cdprimit_cache in fromlist:
            if cdprimit_cache[1] is None:
                raise ValueError("The link needs to be added to collider using the addjlcobj function first!")
            current_from_cdmask = cdprimit_cache[1].node().getFromCollideMask()
            new_from_cdmask = current_from_cdmask | cdmask
            cdprimit_cache[1].node().setFromCollideMask(new_from_cdmask)
            # current_into_cdmask = cdprimit_cache[1].node().getIntoCollideMask()
            # cdprimit_cache[1].node().setIntoCollideMask(current_into_cdmask & ~cdmask)
        for cdprimit_cache in intolist:
            if cdprimit_cache[1] is None:
                raise ValueError("The link needs to be added to collider using the addjlcobj function first!")
            current_into_cdmask = cdprimit_cache[1].node().getIntoCollideMask()
            new_into_cdmask = current_into_cdmask | cdmask
            cdprimit_cache[1].node().setIntoCollideMask(new_into_cdmask)
        self.nbitmask += 1

    def add_cdobj(self, objcm, rel_pos, rel_rotmat, intolist):
        """
        :return: cdobj_info, a dictionary that mimics a joint link; Besides that, there is an additional 'intolist'
                 key to hold intolist to easily toggle off the bitmasks.
        """
        cdobj_info = {}
        cdobj_info['collisionmodel'] = objcm # for reversed lookup
        cdobj_info['gl_pos'] = objcm.get_pos()
        cdobj_info['gl_rotmat'] = objcm.get_rotmat()
        cdobj_info['rel_pos'] = rel_pos
        cdobj_info['rel_rotmat'] = rel_rotmat
        cdobj_info['cdprimit_cache'] = [False, objcm.copy_cdnp_to(self.np, clearmask=True)]
        cdobj_info['cdprimit_cache'][1].node().setFromCollideMask(self._bitmask_ext) # set active
        cdobj_info['intolist'] = intolist
        self.ctrav.addCollider(cdobj_info['cdprimit_cache'][1], self.chan)
        self.all_cdelements.append(cdobj_info)
        self.set_cdpair([cdobj_info['cdprimit_cache']], intolist)
        return cdobj_info

    def delete_cdobj(self, cdobj_info):
        """
        :param cdobj_info: an lnk-like object generated by self.add_objinhnd
        :param objcm:
        :return:
        """
        self.all_cdelements.remove(cdobj_info)
        for cdprimit_cache in cdobj_info['intolist']:
            current_into_cdmask = cdprimit_cache[1].node().getIntoCollideMask()
            new_into_cdmask = current_into_cdmask & ~cdobj_info['cdprimit_cache'][1].node().getFromCollideMask()
            cdprimit_cache[1].node().setIntoCollideMask(new_into_cdmask)
        self.ctrav.removeCollider(cdobj_info['cdprimit_cache'][1])

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        for cdelement in self.all_cdelements: # TODO global ik indicator, right now a bit consuming
            if cdelement['cdprimit_cache'][0]: # need to update
                pos = cdelement['gl_pos']
                rotmat = cdelement['gl_rotmat']
                cdelement['cdprimit_cache'][1].setMat(da.npv3mat3_to_pdmat4(pos, rotmat))
                cdelement['cdprimit_cache'][0] = False # updated
                # print("From", cdelement['cdprimit_cache'][1].node().getFromCollideMask())
                # print("Into", cdelement['cdprimit_cache'][1].node().getIntoCollideMask())
        # check obstacle
        for obstacle in obstacle_list:
            obstacle.pdnp.reparentTo(self.np)
        # check other robots
        for robot in otherrobot_list:
            for cdelement in robot.all_cdelements:
                current_into_cdmask = cdelement['cdprimit_cache'][1].node().getIntoCollideMask()
                cdelement['cdprimit_cache'][1].node().setIntoCollideMask(current_into_cdmask | self._bitmask_ext)
            robot.cc.np.reparentTo(self.np)
        # collision check
        self.ctrav.traverse(self.np)
        # clear other robots
        for robot in otherrobot_list:
            for cdelement in robot.all_cdelements:
                current_into_cdmask = cdlnk['cdprimit_cache'][1].node().getIntoCollideMask()
                cdelement['cdprimit_cache'][1].node().setIntoCollideMask(current_into_cdmask & ~self._bitmask_ext)
            robot.cc.np.detachNode()
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
        for cdelement in self.all_cdelements:
            cdelement['cdprimit_cache'][1].removeNode()
            cdelement['cdprimit_cache'][1] = None
        pass