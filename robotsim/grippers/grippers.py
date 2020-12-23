import copy
import numpy as np
import robotsim._kinematics.jlchain as jl
import robotsim._kinematics.collisionchecker as cc


class Gripper(object):

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

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        :param obstacle_list:
        :param otherrobot_list:
        :return:
        author: weiwei
        date: 20201223
        """
        raise NotImplementedError

    def is_mesh_collided(self, objcm_list=[]):
        """
        Interface for "fine collision detection", must be implemented in child class
        :param objcm_list:
        :return:
        author: weiwei
        date: 20201223
        """
        raise NotImplementedError

    def fix_to(self, pos, rotmat):
        raise NotImplementedError

    def fk(self, motion_val):
        raise NotImplementedError

    def jaw_to(self, jawwidth):
        raise NotImplementedError

    def show_cdprimit(self):
        raise NotImplementedError

    def show_cdmesh(self):
        raise NotImplementedError

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
        return copy.deepcopy(self)

