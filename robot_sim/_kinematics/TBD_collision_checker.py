import copy

import basis.data_adapter as da
import modeling.model_collection as mc
from panda3d.core import NodePath, CollisionTraverser, CollisionHandlerQueue, BitMask32


class CollisionChecker(object):
    """
    A fast collision checker that allows maximum 32 collision pairs
    author: weiwei
    date: 20201214osaka, 20230811toyonaka
    """

    def __init__(self, name="auto"):
        self.cd_trav = CollisionTraverser()
        self.cd_handler = CollisionHandlerQueue()
        self.np = NodePath(name)
        self.bitmask_list = [BitMask32(2 ** n) for n in range(31)]
        self._bitmask_ext = BitMask32(2 ** 31)  # 31 is prepared for cd with external non-active objects
        self.all_cd_elements = []  # a list of cdlnks or cdobjs for quickly accessing the cd elements (cdlnks/cdobjs)

    def add_cdlnks(self, jlc, lnk_idlist):
        """
        The collision node of the given links will be attached to self.np, but their collision bitmask will be cleared
        When a robot_s is treated as an obstacle by another robot_s, the IntoCollideMask of its cce_dict will be
        set to BitMask32(2**31), so that the other robot_s can compare its active_cdelements with the cce_dict.
        :param jlc:
        :param lnk_idlist:
        :return:
        author: weiwei
        date: 20201216toyonaka, 20230811toyonaka
        """
        for id in lnk_idlist:
            if jlc.lnks[id]['cc_nodepath'] is None:
                jlc.lnks[id]['cc_nodepath'] = jlc.lnks[id]['collision_model'].copy_cdnp_to(self.np, clearmask=True)
                self.cd_trav.addCollider(collider=jlc.lnks[id]['cc_nodepath'], handler=self.cd_handler)
                self.all_cd_elements.append(jlc.lnks[id])
            else:
                raise ValueError("The link is already added!")

    def set_active_cdlnks(self, lnk_list):
        """
        The specified collision links will be used for collision detection with external obstacles
        :param lnk_list: essentially a from list like [jlchain.lnk0, jlchain.lnk1...]
                           the correspondent tolist will be set online in cd functions
                           TODO use all elements in self.all_cdnlks if None
        :return:
        author: weiwei
        date: 20201216toyonaka, 20230811toyonaka
        """
        for lnk in lnk_list:
            if lnk['cc_nodepath'] is None:
                raise ValueError("The link needs to be added to collider using the add_cdlnks function first!")
            lnk['cc_nodepath'].node().setFromCollideMask(self._bitmask_ext)

    def set_cdpair(self, from_list, into_list):
        """
        The given collision pair will be used for self collision detection
        :param from_list: each element is a dictionary including a least a [cc_nodepath] key-value pair
        :param into_list:
        :return:
        author: weiwei
        date: 20201215, 20230811toyonaka
        """
        if len(self.bitmask_list) == 0:
            raise ValueError("Too many collision pairs! Maximum: 29")
        allocated_bitmask = self.bitmask_list.pop()
        for cd_element in from_list:
            if cd_element['cc_nodepath'] is None:
                raise ValueError("The link needs to be added to collider using the addjlcobj function first!")
            current_from_cdmask = cd_element['cc_nodepath'].node().getFromCollideMask()
            new_from_cdmask = current_from_cdmask | allocated_bitmask
            cd_element['cc_nodepath'].node().setFromCollideMask(new_from_cdmask)
        for cd_element in into_list:
            if cd_element['cc_nodepath'] is None:
                raise ValueError("The link needs to be added to collider using the addjlcobj function first!")
            current_into_cdmask = cd_element['cc_nodepath'].node().getIntoCollideMask()
            new_into_cdmask = current_into_cdmask | allocated_bitmask
            cd_element['cc_nodepath'].node().setIntoCollideMask(new_into_cdmask)

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
        cdobj_info['cc_nodepath'] = objcm.copy_cdnp_to(self.np, clearmask=True)
        cdobj_info['cc_nodepath'].node().setFromCollideMask(self._bitmask_ext)  # set active
        self.cd_trav.addCollider(collider=cdobj_info['cc_nodepath'], handler=self.cd_handler)
        self.all_cd_elements.append(cdobj_info)
        self.set_cdpair([cdobj_info], into_list)
        return cdobj_info

    def delete_cdobj(self, cdobj_info):
        """
        :param cdobj_info: an lnk-like object generated by self.add_objinhnd
        :param cmodel:
        :return:
        """
        self.all_cd_elements.remove(cdobj_info)
        cdnp_to_delete = self.np.getChild(cdobj_info['cdprimit_childid'])
        self.cd_trav.removeCollider(cdnp_to_delete)
        this_cdmask = cdnp_to_delete.node().getFromCollideMask()
        this_cdmask_exclude_ext = this_cdmask & ~self._bitmask_ext
        for cdlnk in cdobj_info['into_list']:
            cdnp = self.np.getChild(cdlnk['cdprimit_childid'])
            current_into_cdmask = cdnp.node().getIntoCollideMask()
            new_into_cdmask = current_into_cdmask & ~this_cdmask_exclude_ext
            cdnp.node().setIntoCollideMask(new_into_cdmask)
        cdnp_to_delete.detachNode()
        self.bitmask_list.append(this_cdmask_exclude_ext)

    def is_collided(self, obstacle_list=[], otherrobot_list=[], toggle_contacts=False):
        """
        :param obstacle_list: staticgeometricmodel
        :param otherrobot_list:
        :return:
        """
        for cd_element in self.all_cd_elements:
            pos = cd_element['gl_pos']
            rotmat = cd_element['gl_rotmat']
            cd_element['cc_nodepath'].setPosQuat(da.npvec3_to_pdvec3(pos), da.npmat3_to_pdquat(rotmat))
            # print(da.npv3mat3_to_pdmat4(pos, rotmat))
            # print("From", cdnp.node().getFromCollideMask())
            # print("Into", cdnp.node().getIntoCollideMask())
        # print("xxxx colliders xxxx")
        # for collider in self.cd_trav.getColliders():
        #     print(collider.getMat())
        #     print("From", collider.node().getFromCollideMask())
        #     print("Into", collider.node().getIntoCollideMask())
        # attach obstacles
        obstacle_parent_list = []
        for obstacle in obstacle_list:
            obstacle_parent_list.append(obstacle.pdndp.getParent())  # save
            obstacle.pdndp.reparentTo(self.np)  # reparent
        # attach other robots
        for robot in otherrobot_list:
            for cdnp in robot.cc.np.getChildren():
                current_into_cdmask = cdnp.node().getIntoCollideMask()
                new_into_cdmask = current_into_cdmask | self._bitmask_ext
                cdnp.node().setIntoCollideMask(new_into_cdmask)
            robot.cc.np.reparentTo(self.np)
        # collision check
        self.cd_trav.traverse(self.np)
        # clear obstacles
        for i, obstacle in enumerate(obstacle_list):
            obstacle.pdndp.reparentTo(obstacle_parent_list[i])  # restore to saved values
        # clear other robots
        for robot in otherrobot_list:
            for cdnp in robot.cc.np.getChildren():
                current_into_cdmask = cdnp.node().getIntoCollideMask()
                new_into_cdmask = current_into_cdmask & ~self._bitmask_ext
                cdnp.node().setIntoCollideMask(new_into_cdmask)
            robot.cc.np.detachNode()
        if self.cd_handler.getNumEntries() > 0:
            collision_result = True
        else:
            collision_result = False
        if toggle_contacts:
            contact_points = [da.pdvec3_to_npvec3(cd_entry.getSurfacePoint(base.render)) for cd_entry in
                              self.cd_handler.getEntries()]
            return collision_result, contact_points
        else:
            return collision_result

    def show_cdprimit(self):
        """
        Copy the current pdndp to base.render to show collision states
        TODO: maintain a list to allow unshow
        :return:
        author: weiwei
        date: 20220404
        """
        # print("call show_cdprimit")
        snp_cpy = self.np.copyTo(base.render)
        for cdelement in self.all_cd_elements:
            pos = cdelement['gl_pos']
            rotmat = cdelement['gl_rotmat']
            cdnp = snp_cpy.getChild(cdelement['cdprimit_childid'])
            cdnp.setPosQuat(da.npvec3_to_pdvec3(pos), da.npmat3_to_pdquat(rotmat))
            cdnp.show()

    def disable(self):
        """
        clear pairs and pdndp
        :return:
        """
        for cdelement in self.all_cd_elements:
            cdelement['cdprimit_childid'] = -1
        self.all_cd_elements = []
        for child in self.np.getChildren():
            child.removeNode()
        self.bitmask_list = list(range(31))
