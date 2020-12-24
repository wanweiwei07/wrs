import copy
import numpy as np
import modeling.modelcollection as mc
import robotsim._kinematics.jlchain as jl
import robotsim._kinematics.collisionchecker as cc


class GripperInterface(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='yumi_gripper'):
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        # joints
        # - coupling - No coupling by default
        self.coupling = jl.JLChain(pos=self.pos,
                                   rotmat=self.rotmat,
                                   homeconf=np.zeros(0),
                                   name='coupling')
        self.coupling.jnts[1]['loc_pos'] = np.array([0, 0, .0])
        self.coupling.lnks[0]['name'] = 'coupling_lnk0'
        # toggle on the following part to assign an explicit mesh model to a coupling
        # self.coupling.lnks[0]['meshfile'] = os.path.join(this_dir, "meshes", "xxx.stl")
        # self.coupling.lnks[0]['rgba'] = [.2, .2, .2, 1]
        self.coupling.reinitialize()
        self.coupling.disable_localcc()
        # collision detection
        self.cc = cc.CollisionChecker("collision_checker")
        self.cdmesh_collection = mc.ModelCollection()
        # fk tag
        self.is_fk_updated = False

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        :param obstacle_list:
        :param otherrobot_list:
        :return:
        author: weiwei
        date: 20201223
        """
        return self.cc.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list,
                                   need_update=self.is_fk_updated)

    def is_mesh_collided(self, objcm_list=[]):
        if self.is_fk_updated:
            for i, cdelement in enumerate(self.cc.all_cdelements):
                pos = cdelement['gl_pos']
                rotmat = cdelement['gl_rotmat']
                self.cdmesh_collection.cm_list[i].set_pos(pos)
                self.cdmesh_collection.cm_list[i].set_rotmat(rotmat)
                if self.cdmesh_collection.cm_list[i].is_mcdwith(objcm_list):
                    return True
        return False

    def fix_to(self, pos, rotmat):
        raise NotImplementedError

    def fk(self, motion_val):
        raise NotImplementedError

    def jaw_to(self, jawwidth):
        raise NotImplementedError

    def show_cdprimit(self):
        self.cc.show_cdprimit(need_update=self.is_fk_updated)

    def unshow_cdprimit(self):
        self.cc.unshow_cdprimit()

    def show_cdmesh(self):
        if self.is_fk_updated:
            for i, cdelement in enumerate(self.cc.all_cdelements):
                pos = cdelement['gl_pos']
                rotmat = cdelement['gl_rotmat']
                self.cdmesh_collection.cm_list[i].set_pos(pos)
                self.cdmesh_collection.cm_list[i].set_rotmat(rotmat)
        self.cdmesh_collection.show_cdmesh()

    def unshow_cdmesh(self):
        self.cdmesh_collection.unshow_cdmesh()

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='yumi_gripper_stickmodel'):
        raise NotImplementedError

    def gen_meshmodel(self,
                      tcp_jntid=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='yumi_gripper_meshmodel'):
        raise NotImplementedError

    def disable_localcc(self):
        """
        clear pairs and nodepath
        :return:
        """
        for cdelement in self.cc.all_cdelements:
            cdelement['cdprimit_childid'] = -1
        self.cc.all_cdelements = []
        for child in self.cc.np.getChildren():
            child.removeNode()
        self.cc.nbitmask = 0

    def copy(self):
        # TDO check is disabled?
        self_copy = copy.deepcopy(self)
        # deepcopying colliders are problematic, I have to update it manually
        for child in self_copy.cc.np.getChildren():
            self_copy.cc.ctrav.addCollider(child, self_copy.cc.chan)
        return self_copy

