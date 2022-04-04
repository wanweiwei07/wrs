import copy

import basis.data_adapter as da
import modeling.model_collection as mc
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
        self.bitmask_list = [BitMask32(2**n) for n in range(31)]
        self._bitmask_ext = BitMask32(2 ** 31)  # 31 is prepared for cd with external non-active objects
        self.all_cdelements = []  # a list of cdlnks or cdobjs for quick accessing the cd elements (cdlnks/cdobjs)

    def add_cdlnks(self, jlcobj, lnk_idlist):
        """
        The collision node of the given links will be attached to self.np, but their collision bitmask will be cleared
        When the a robot_s is treated as an obstacle by another robot_s, the IntoCollideMask of its all_cdelements will be
        set to BitMask32(2**31), so that the other robot_s can compare its active_cdelements with the all_cdelements.
        :param jlcobj:
        :param lnk_idlist:
        :return:
        author: weiwei
        date: 20201216toyonaka
        """
        for id in lnk_idlist:
            if jlcobj.lnks[id]['cdprimit_childid'] == -1:  # first time add
                cdnp = jlcobj.lnks[id]['collision_model'].copy_cdnp_to(self.np, clearmask=True)
                self.ctrav.addCollider(cdnp, self.chan)
                self.all_cdelements.append(jlcobj.lnks[id])
                jlcobj.lnks[id]['cdprimit_childid'] = len(self.all_cdelements) - 1
            else:
                raise ValueError("The link is already added!")

    def set_active_cdlnks(self, activelist):
        """
        The specified collision links will be used for collision detection with external obstacles
        :param activelist: essentially a from list like [jlchain.lnk0, jlchain.lnk1...]
                           the correspondent tolist will be set online in cd functions
                           TODO use all elements in self.all_cdnlks if None
        :return:
        author: weiwei
        date: 20201216toyonaka
        """
        for cdlnk in activelist:
            if cdlnk['cdprimit_childid'] == -1:
                raise ValueError("The link needs to be added to collider using the add_cdlnks function first!")
            cdnp = self.np.getChild(cdlnk['cdprimit_childid'])
            cdnp.node().setFromCollideMask(self._bitmask_ext)

    def set_cdpair(self, fromlist, intolist):
        """
        The given collision pair will be used for self collision detection
        :param fromlist: [[bool, cdprimit_cache], ...]
        :param intolist: [[bool, cdprimit_cache], ...]
        :return:
        author: weiwei
        date: 20201215
        """
        if len(self.bitmask_list) == 0:
            raise ValueError("Too many collision pairs! Maximum: 29")
        allocated_bitmask = self.bitmask_list.pop()
        for cdlnk in fromlist:
            if cdlnk['cdprimit_childid'] == -1:
                raise ValueError("The link needs to be added to collider using the addjlcobj function first!")
            cdnp = self.np.getChild(cdlnk['cdprimit_childid'])
            current_from_cdmask = cdnp.node().getFromCollideMask()
            new_from_cdmask = current_from_cdmask | allocated_bitmask
            cdnp.node().setFromCollideMask(new_from_cdmask)
        for cdlnk in intolist:
            if cdlnk['cdprimit_childid'] == -1:
                raise ValueError("The link needs to be added to collider using the addjlcobj function first!")
            cdnp = self.np.getChild(cdlnk['cdprimit_childid'])
            current_into_cdmask = cdnp.node().getIntoCollideMask()
            new_into_cdmask = current_into_cdmask | allocated_bitmask
            cdnp.node().setIntoCollideMask(new_into_cdmask)

    def add_cdobj(self, objcm, rel_pos, rel_rotmat, into_list):
        """
        :return: cdobj_info, a dictionary that mimics a joint link; Besides that, there is an additional 'into_list'
                 key to hold into_list to easily toggle off the bitmasks.
        """
        cdobj_info = {}
        cdobj_info['collision_model'] = objcm  # for reversed lookup
        cdobj_info['gl_pos'] = objcm.get_pos()
        cdobj_info['gl_rotmat'] = objcm.get_rotmat()
        cdobj_info['rel_pos'] = rel_pos
        cdobj_info['rel_rotmat'] = rel_rotmat
        cdobj_info['into_list'] = into_list
        cdnp = objcm.copy_cdnp_to(self.np, clearmask=True)
        cdnp.node().setFromCollideMask(self._bitmask_ext)  # set active
        self.ctrav.addCollider(cdnp, self.chan)
        self.all_cdelements.append(cdobj_info)
        cdobj_info['cdprimit_childid'] = len(self.all_cdelements) - 1
        self.set_cdpair([cdobj_info], into_list)
        return cdobj_info

    def delete_cdobj(self, cdobj_info):
        """
        :param cdobj_info: an lnk-like object generated by self.add_objinhnd
        :param objcm:
        :return:
        """
        self.all_cdelements.remove(cdobj_info)
        cdnp_to_delete = self.np.getChild(cdobj_info['cdprimit_childid'])
        self.ctrav.removeCollider(cdnp_to_delete)
        this_cdmask = cdnp_to_delete.node().getFromCollideMask()
        for cdlnk in cdobj_info['into_list']:
            cdnp = self.np.getChild(cdlnk['cdprimit_childid'])
            current_into_cdmask = cdnp.node().getIntoCollideMask()
            new_into_cdmask = current_into_cdmask & ~this_cdmask
            cdnp.node().setIntoCollideMask(new_into_cdmask)
        cdnp_to_delete.detachNode()
        self.bitmask_list.append(this_cdmask)

    def is_collided(self, obstacle_list=[], otherrobot_list=[], toggle_contact_points=False):
        """
        :param obstacle_list: staticgeometricmodel
        :param otherrobot_list:
        :return:
        """
        for cdelement in self.all_cdelements:
            pos = cdelement['gl_pos']
            rotmat = cdelement['gl_rotmat']
            cdnp = self.np.getChild(cdelement['cdprimit_childid'])
            cdnp.setPosQuat(da.npv3_to_pdv3(pos), da.npmat3_to_pdquat(rotmat))
            # print(da.npv3mat3_to_pdmat4(pos, rotmat))
            # print("From", cdnp.node().getFromCollideMask())
            # print("Into", cdnp.node().getIntoCollideMask())
        # print("xxxx colliders xxxx")
        # for collider in self.ctrav.getColliders():
        #     print(collider.getMat())
        #     print("From", collider.node().getFromCollideMask())
        #     print("Into", collider.node().getIntoCollideMask())
        # attach obstacles
        obstacle_parent_list = []
        for obstacle in obstacle_list:
            obstacle_parent_list.append(obstacle.objpdnp.getParent())
            obstacle.objpdnp.reparentTo(self.np)
        # attach other robots
        for robot in otherrobot_list:
            for cdnp in robot.cc.np.getChildren():
                current_into_cdmask = cdnp.node().getIntoCollideMask()
                new_into_cdmask = current_into_cdmask | self._bitmask_ext
                cdnp.node().setIntoCollideMask(new_into_cdmask)
            robot.cc.np.reparentTo(self.np)
        # collision check
        self.ctrav.traverse(self.np)
        # clear obstacles
        for i, obstacle in enumerate(obstacle_list):
            obstacle.objpdnp.reparentTo(obstacle_parent_list[i])
        # clear other robots
        for robot in otherrobot_list:
            for cdnp in robot.cc.np.getChildren():
                current_into_cdmask = cdnp.node().getIntoCollideMask()
                new_into_cdmask = current_into_cdmask & ~self._bitmask_ext
                cdnp.node().setIntoCollideMask(new_into_cdmask)
            robot.cc.np.detachNode()
        if self.chan.getNumEntries() > 0:
            collision_result = True
        else:
            collision_result = False
        if toggle_contact_points:
            contact_points = [da.pdv3_to_npv3(cd_entry.getSurfacePoint(base.render)) for cd_entry in
                              self.chan.getEntries()]
            return collision_result, contact_points
        else:
            return collision_result

    def show_cdprimit(self):
        """
        Copy the current nodepath to base.render to show collision states
        TODO: maintain a list to allow unshow
        :return:
        author: weiwei
        date: 20220404
        """
        # print("call show_cdprimit")
        snp_cpy = self.np.copyTo(base.render)
        for cdelement in self.all_cdelements:
            pos = cdelement['gl_pos']
            rotmat = cdelement['gl_rotmat']
            cdnp = snp_cpy.getChild(cdelement['cdprimit_childid'])
            cdnp.setPosQuat(da.npv3_to_pdv3(pos), da.npmat3_to_pdquat(rotmat))
            cdnp.show()

    def disable(self):
        """
        clear pairs and nodepath
        :return:
        """
        for cdelement in self.all_cdelements:
            cdelement['cdprimit_childid'] = -1
        self.all_cdelements = []
        for child in self.np.getChildren():
            child.removeNode()
        self.bitmask_list = list(range(31))
