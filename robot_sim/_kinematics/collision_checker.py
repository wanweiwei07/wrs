import copy

import basis.data_adapter as da
import modeling.model_collection as mc
import modeling._panda_cdhelper as mph
from panda3d.core import NodePath, CollisionTraverser, CollisionHandlerQueue, BitMask32


class CCElement(object):
    """
    A collision detection element
    It is initialized by a pointer to the collision model and
    a pointer to the pdcndp attached to the traverse tree of
    an collision checker instance
    author: weiwei
    date: 20231116
    """

    def __init__(self, lnk, host_cc):
        self.host_cc = host_cc
        self.lnk = lnk
        # a transformed and attached copy of the reference cdprimitive (essentially pdcndp), tfd=transformed
        self.tfd_cdprimitive = mph.copy_cdprimitive_attach_to(self.lnk.cmodel,
                                                              self.host_cc.cd_pdndp,
                                                              clear_mask=True)
        self.host_cc.cd_trav.addCollider(collider=self.tfd_cdprimitive, handler=self.host_cc.cd_handler)
        # a dict with from_mask as keys and into_list (a lsit of cce) as values
        self.cce_into_dict = {}
        # toggle on collision detection with external obstacles by default
        self.enable_cd_ext(type="from")

    def enable_cd_ext(self, type="from"):
        """
        enable collision detection with external collision models
        :return:
        """
        mph.change_cdmask(self.tfd_cdprimitive, mph.BITMASK_EXT, action="add", type=type)

    def disable_cd_ext(self, type="from"):
        """
        disable collision detection with external collision models
        :return:
        """
        mph.change_cdmask(self.tfd_cdprimitive, mph.BITMASK_EXT, action="remove", type=type)

    def isolate(self):
        """
        isolate this cce from collision traversers and return the bitmasks
        from mask is not changed as it will no longer be used
        :return:
        """
        # detach from pdcndp tree
        self.tfd_cdprimitive.detachNode()
        # remove from collision traverser
        self.host_cc.cd_trav.removeCollider(collider=self.tfd_cdprimitive)
        # remove the into bitmask of all cces in the cce_into_dict
        bitmask_list_to_return = []
        for allocated_bitmask, cce_into_list in self.cce_into_dict:
            for cce_into in cce_into_list:
                cce_into.remove_into_cdmask(allocated_bitmask)
                bitmask_list_to_return.append(allocated_bitmask)
        return bitmask_list_to_return

    def add_from_cdmask(self, allocated_bitmask, cce_into_list):
        """
        Note: the bitmask of cce_into_list should be updated externally in advance
        :param allocated_bitmask:
        :param cce_into_list:
        :return:
        """
        mph.change_cdmask(self.tfd_cdprimitive, allocated_bitmask, action="add", type="from")
        self.cce_into_dict[allocated_bitmask] = cce_into_list

    def remove_from_cdmask(self, allocated_bitmask):
        """
        the into cce's cdmask will also be updated in response to this removement
        :param allocated_bitmask:
        :return:
        author: weiwei
        date: 20231117
        """
        mph.change_cdmask(self.tfd_cdprimitive, allocated_bitmask, action="remove", type="from")
        cce_into_list = self.cdpair_dict[allocated_bitmask]
        for cce_into in cce_into_list:
            cce_into.remove_into_cdmask(allocated_bitmask)

    def add_into_cdmask(self, allocated_bitmask):
        mph.change_cdmask(self.tfd_cdprimitive, allocated_bitmask, action="add", type="into")

    def remove_into_cdmask(self, allocated_bitmask):
        mph.change_cdmask(self.tfd_cdprimitive, allocated_bitmask, action="remove", type="into")


class CollisionChecker(object):
    """
    Hosts collision elements (robot links and manipulated objects),
    and checks their internal collisions and externaal collisions with other obstacles/robots
    fast and allows maximum 32 collision pairs
    author: weiwei
    date: 20201214osaka, 20230811toyonaka
    """

    def __init__(self, name="cc"):
        self.cd_trav = CollisionTraverser()
        self.cd_handler = CollisionHandlerQueue()
        self.cd_pdndp = NodePath(name)  # root of the traverse tree
        self.bitmask_pool = [BitMask32(2 ** n) for n in range(31)]
        self.bitmask_ext = BitMask32(2 ** 31)  # 31 is prepared for cd with external non-active objects
        self.cce_dict = {}  # a dict of CCElement

    def add_cce(self, lnk):
        """
        add a Link as a ccelement
        :param lnk: instance of rkjlc.Link
        :return:
        author: weiwei
        date: 20231116
        """
        self.cce_dict[lnk.uuid] = CCElement(lnk, self.cd_pdndp)

    def remove_cce(self, lnk):
        """
        remove a ccelement by using the uuid of a lnk
        :param lnk: instance of CollisionModel
        :return:
        author: weiwei
        date: 20231117
        """
        cce = self.cce_dict.pop(lnk.uuid)
        bitmask_list_to_return = cce.isolate()
        self.bitmask_pool += bitmask_list_to_return

    def set_cdpair(self, lnk_from_list, lnk_into_list):
        """
        The given two lists will be checked for collisions
        :param lnk_from_list: a list of rkjlc.Link
        :param lnk_into_list: a list of rkjlc.Link
        :return:
        author: weiwei
        date: 20201215, 20230811, 20231116
        """
        if len(self.bitmask_pool) == 0:
            raise ValueError("Too many collision pairs! Maximum: 29")
        allocated_bitmask = self.bitmask_pool.pop()
        cce_into_list = []
        for lnk_into in lnk_into_list:
            if lnk_into.uuid in self.cce_dict.keys():
                self.cce_dict[lnk_into.uuid].add_into_cdmask(allocated_bitmask)
                cce_into_list.append(self.cce_dict[lnk_into.uuid])
            else:
                raise KeyError("Into lnks do not exist in the cce_dict.")
        for lnk_from in lnk_from_list:
            if lnk_from.uuid in self.cce_dict.keys():
                self.cce_dict[lnk_from.uuid].add_from_cdmask(allocated_bitmask, cce_into_list)
            else:
                raise KeyError("From lnks do not exist in the cce_dict.")

    def is_collided(self, obstacle_list=[], otherrobot_list=[], toggle_contacts=False):
        """
        :param obstacle_list: staticgeometricmodel
        :param otherrobot_list:
        :return:
        """
        for cce in self.cce_dict.values():
            cce.tfd_cdprimitive.setPosQuat(da.npvec3_to_pdvec3(cce.cmodel.pos), da.npmat3_to_pdquat(cce.cmodel.rotmat))
            # print(da.npv3mat3_to_pdmat4(pos, rotmat))
            # print("From", cdnp.node().getFromCollideMask())
            # print("Into", cdnp.node().getIntoCollideMask())
        # print("xxxx colliders xxxx")
        # for collider in self.cd_trav.getColliders():
        #     print(collider.getMat())
        #     print("From", collider.node().getFromCollideMask())
        #     print("Into", collider.node().getIntoCollideMask())
        # attach obstacles
        obstacle_cdprimitive_list = []
        for obstacle in obstacle_list:
            obstacle.attach_cdprim_to(self.cd_pdndp)
        # attach other robots
        for robot in otherrobot_list:
            for cce in robot.cc.cce_dict.values():
                cce.enable_cd_ext(type="into")
            robot.cc.cd_pdndp.reparentTo(self.cd_pdndp)
        # collision check
        self.cd_trav.traverse(self.cd_pdndp)
        # clear obstacles
        for obstacle in obstacle_list:
            obstacle.detach_cdprim()
        # clear other robots
        for robot in otherrobot_list:
            for cce in robot.cc.cce_dict.values():
                cce.disable_cd_ext(type="into")
            robot.cc.cd_pdndp.detachNode()
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

    def show_cdprimitive(self):
        """
        Copy the current pdndp to base.render to show collision states
        :return:
        author: weiwei
        date: 20220404
        """
        for cce in self.cce_dict.values():
            tmp_tfd_cdprimitive = mph.copy_cdprimitive_attach_to(cmodel=cce.lnk,
                                                                 tgt_pdndp=base.render,
                                                                 homomat=cce.lnk.get_homomat(),
                                                                 clear_mask=True)
            tmp_tfd_cdprimitive.show()