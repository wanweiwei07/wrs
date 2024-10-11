from panda3d.core import NodePath, CollisionTraverser, CollisionHandlerQueue, BitMask32
import wrs.modeling._panda_cdhelper as mph
import wrs.basis.data_adapter as da

class CCElement(object):
    """
    A collision detection element
    It is initialized by a pointer to the collision model and
    a pointer to the pdcndp attached to the traverse tree of
    an collision checker instance
    author: weiwei
    date: 20231116
    """

    def __init__(self, lnk, host_cc, toggle_extcd=True):
        self.host_cc = host_cc
        self.lnk = lnk
        # a transformed and attached copy of the reference cdprim (essentially pdcndp), tfd=transformed
        self.tfd_cdprim = mph.copy_cdprim_attach_to(self.lnk.cmodel,
                                                    self.host_cc.cd_pdndp,
                                                    clear_mask=True)
        # print(self.tfd_cdprim.node().is_collision_node())
        for child_pdndp in self.tfd_cdprim.getChildren():
            self.host_cc.cd_trav.addCollider(collider=child_pdndp, handler=self.host_cc.cd_handler)
        # a dict with from_mask as keys and into_list (a lsit of cce) as values
        self.cce_into_dict = {}
        if toggle_extcd:
            # toggle on collision detection with external obstacles by default
            self.enable_extcd(type="from")

    def _close_bitmask(self, allocated_bitmask):
        """
        return the allocated bitmask to the host cc
        :param allocated_bitmask:
        :return:
        author: weiwei
        date: 20240630
        """
        cce_into_list = self.cce_into_dict[allocated_bitmask]
        for cce_into in cce_into_list:
            cce_into.remove_into_cdmask(allocated_bitmask)
        self.host_cc.bitmask_pool.append(allocated_bitmask)

    def enable_extcd(self, type="from"):
        """
        enable collision detection with external collision models
        :return:
        """
        mph.change_cdmask(self.tfd_cdprim, mph.BITMASK_EXT, action="add", type=type)

    def disable_extcd(self, type="from"):
        """
        disable collision detection with external collision models
        :return:
        """
        mph.change_cdmask(self.tfd_cdprim, mph.BITMASK_EXT, action="remove", type=type)

    def isolate(self):
        """
        isolate this cce from collision traversers and return the bitmasks
        from mask is not changed as it will no longer be used
        :return:
        author: weiwei
        date: 20240309
        """
        # remove from collision traverser
        for child_pdndp in self.tfd_cdprim.getChildren():
            self.host_cc.cd_trav.removeCollider(child_pdndp)
        # remove from pdcndp tree
        self.tfd_cdprim.removeNode()
        # set self.tfd_cdprim to None for delayed feletion from other cce's into list (see 63)
        self.tfd_cdprim = None
        # remove the into bitmask of all cces in the cce_into_dict
        bitmask_list_to_return = []
        for allocated_bitmask in self.cce_into_dict.keys():
            self.host_cc.bitmask_users_dict[allocated_bitmask].remove(self.lnk.uuid)
            if len(self.host_cc.bitmask_users_dict[allocated_bitmask]) == 0:
                self._close_bitmask(allocated_bitmask)
                bitmask_list_to_return.append(allocated_bitmask)
        return bitmask_list_to_return

    def get_cdmask(self, type="from"):
        """
        get the cdmask of the cce
        :param type:
        :return:
        author: weiwei
        dater: 20241009
        """
        return mph.get_cdmask(self.tfd_cdprim, type=type)

    def add_from_cdmask(self, allocated_bitmask, cce_into_list):
        """
        Note: the bitmask of cce_into_list should be updated externally in advance
        :param allocated_bitmask:
        :param cce_into_list:
        :return:
        """
        mph.change_cdmask(self.tfd_cdprim, allocated_bitmask, action="add", type="from")
        self.cce_into_dict[allocated_bitmask] = cce_into_list
        self.host_cc.bitmask_users_dict[allocated_bitmask].append(self.lnk.uuid)

    def remove_from_cdmask(self, allocated_bitmask):
        """
        the into cce's cdmask will also be updated in response to this removement
        :param allocated_bitmask:
        :return:
        author: weiwei
        date: 20231117
        """
        mph.change_cdmask(self.tfd_cdprim, allocated_bitmask, action="remove", type="from")
        self.host_cc.bitmask_users_dict[allocated_bitmask].remove(self.lnk.uuid)
        if len(self.host_cc.bitmask_users_dict[allocated_bitmask]) == 0:
            self._close_bitmask(allocated_bitmask)

    def add_into_cdmask(self, allocated_bitmask):
        mph.change_cdmask(self.tfd_cdprim, allocated_bitmask, action="add", type="into")

    def remove_into_cdmask(self, allocated_bitmask):
        mph.change_cdmask(self.tfd_cdprim, allocated_bitmask, action="remove", type="into")


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
        self.cd_pdndp = NodePath(name)  # path of the traverse tree
        self.bitmask_pool = [BitMask32(2 ** n) for n in range(31)]
        self.bitmask_users_dict = {}
        for bitmask in self.bitmask_pool:
            self.bitmask_users_dict[bitmask] = []
        self.bitmask_ext = BitMask32(2 ** 31)  # 31 is prepared for cd with external non-active objects
        self.cce_dict = {}  # a dict of CCElement
        # temporary parameter for toggling on/off show_cdprimit
        self._toggled_cdprim_list = []
        # togglable lists
        self.dynamic_into_list = [] # for oiee
        self.dynamic_ext_list = [] # for ignoring the external collision of certain components

    def add_cce(self, lnk, toggle_extcd=True):
        """
        add a Link as a ccelement
        :param lnk: instance of rkjlc.Link
        :return:
        author: weiwei
        date: 20231116
        """
        self.cce_dict[lnk.uuid] = CCElement(lnk, self, toggle_extcd=toggle_extcd)
        return lnk.uuid

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
        try:
            self.dynamic_ext_list.remove(lnk.uuid)
        except ValueError:
            pass

    def remove_cce_by_id(self, uuid):
        """
        remove a ccelement by using the uuid of a lnk
        :param lnk: instance of CollisionModel
        :return:
        author: weiwei
        date: 20240303
        """
        cce = self.cce_dict.pop(uuid)
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

    def set_cdpair_by_ids(self, uuid_from_list, uuid_into_list):
        """
        The given two lists will be checked for collisions
        :param uuid_from_list: a list of rkjlc.Link.uuid
        :param uuid_into_list: a list of rkjlc.Link.uuid
        :return:
        author: weiwei
        date: 20240303
        """
        if len(self.bitmask_pool) == 0:
            raise ValueError("Too many collision pairs! Maximum: 29")
        allocated_bitmask = self.bitmask_pool.pop()
        cce_into_list = []
        for uuid_into in uuid_into_list:
            if uuid_into in self.cce_dict.keys():
                self.cce_dict[uuid_into].add_into_cdmask(allocated_bitmask)
                cce_into_list.append(self.cce_dict[uuid_into])
            else:
                raise KeyError("Into lnks do not exist in the cce_dict.")
        for uuid_from in uuid_from_list:
            if uuid_from in self.cce_dict.keys():
                self.cce_dict[uuid_from].add_from_cdmask(allocated_bitmask, cce_into_list)
            else:
                raise KeyError("From lnks do not exist in the cce_dict.")

    def is_collided(self, obstacle_list=None, other_robot_list=None, toggle_contacts=False):
        """
        :param obstacle_list: staticgeometricmodel
        :param other_robot_list:
        :return:
        """
        for cce in self.cce_dict.values():
            cce.tfd_cdprim.setPosQuat(da.npvec3_to_pdvec3(cce.lnk.gl_pos),
                                      da.npmat3_to_pdquat(cce.lnk.gl_rotmat))
            # print(da.npv3mat3_to_pdmat4(pos, rotmat))
            # print("From", cdnp.node().getFromCollideMask())
            # print("Into", cdnp.node().getIntoCollideMask())
        # print("xxxx colliders xxxx")
        # for collider in self.cd_trav.getColliders():
        #     print(collider.getMat())
        #     print("From", collider.node().getFromCollideMask())
        #     print("Into", collider.node().getIntoCollideMask())
        # attach obstacles
        # print(len(obstacle_list))
        # base.run()
        # obstacle_cdprim_list = []
        # print("obstacles")
        if obstacle_list is not None:
            for obstacle_cmodel in obstacle_list:
                obstacle_cmodel.attach_cdprim_to(self.cd_pdndp)
                # print(mph.get_cdmask(obstacle_cmodel.cdprim, type="from"))
                # print(mph.get_cdmask(obstacle_cmodel.cdprim, type="into"))
            # obstacle_cdprim_list.append(mph.copy_cdprim_attach_to(obstacle_cmodel,
            #                                                       self.cd_pdndp,
            #                                                       homomat=obstacle_cmodel.homomat,
            #                                                       clear_mask=True))
            ## show all cdprims for debug purpose
            # self.cd_pdndp.reparentTo(base.render)
            # for cdprim in self.cd_pdndp.getChildren():
            #     mph.toggle_show_collision_node(cdprim, toggle_show_on=True)
        # attach other robots
        if other_robot_list is not None:
            for robot in other_robot_list:
                for cce in robot.cc.cce_dict.values():  # TODO: wrong, save and restore mask
                    cce.enable_extcd(type="into")
                robot.cc.cd_pdndp.reparentTo(self.cd_pdndp)
        # collision check
        self.cd_trav.traverse(self.cd_pdndp)
        # clear obstacles
        if obstacle_list is not None:
            for obstacle_cmodel in obstacle_list:
                obstacle_cmodel.detach_cdprim()
        # for obstacle_cdprim in obstacle_cdprim_list:
        #     obstacle_cdprim.removeNode()
        # clear other robots
        if other_robot_list is not None:
            for robot in other_robot_list:
                for cce in robot.cc.cce_dict.values():
                    cce.disable_extcd(type="into")
                robot.cc.cd_pdndp.detachNode()
        if self.cd_handler.getNumEntries() > 0:
            collision_result = True
        else:
            collision_result = False
        if toggle_contacts:
            contact_points = [da.pdvec3_to_npvec3(cd_entry.getSurfacePoint(base.render)) for cd_entry in
                              self.cd_handler.getEntries()]
            return (collision_result, contact_points)
        else:
            return collision_result

    def show_cdprim(self):
        """
        Copy the current pdndp to base.render to show collision states
        :return:
        author: weiwei
        date: 20220404
        """
        for cce in self.cce_dict.values():
            tmp_tfd_cdprim = mph.copy_cdprim_attach_to(cmodel=cce.lnk.cmodel,
                                                       tgt_pdndp=base.render,
                                                       homomat=cce.lnk.gl_homomat,
                                                       clear_mask=True)
            mph.toggle_show_collision_node(tmp_tfd_cdprim, toggle_show_on=True)
            self._toggled_cdprim_list.append(tmp_tfd_cdprim)

    def unshow_cdprim(self):
        for cdprim in self._toggled_cdprim_list:
            mph.toggle_show_collision_node(cdprim, toggle_show_on=False)
            cdprim.removeNode()
        self._toggled_cdprim_list = []
