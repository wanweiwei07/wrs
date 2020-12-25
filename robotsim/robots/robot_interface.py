import copy
import numpy as np
import robotsim._kinematics.collisionchecker as cc


class ManipulatorInterface(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='yumi_gripper'):
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
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

    def fix_to(self, pos, rotmat):
        raise NotImplementedError

    def show_cdprimit(self):
        self.cc.show_cdprimit(need_update=self.is_fk_updated)

    def unshow_cdprimit(self):
        self.cc.unshow_cdprimit()

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
        self_copy = copy.deepcopy(self)
        # deepcopying colliders are problematic, I have to update it manually
        for child in self_copy.cc.np.getChildren():
            self_copy.cc.ctrav.addCollider(child, self_copy.cc.chan)
        return self_copy
