import copy
import numpy as np
import robotsim._kinematics.collisionchecker as cc


class RobotInterface(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='robot_interface'):
        # TODO self.jlcs = {}
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        # collision detection
        self.cc = None
        # component map for quick access
        self.manipulator_dict = {}
        self.hnd_dict = {}

    @property
    def is_fk_updated(self):
        raise NotImplementedError

    def change_name(self, name):
        self.name = name

    def get_hnd_on_component(self, component_name):
        raise NotImplementedError

    def get_jnt_ranges(self, component_name):
        return self.manipulator_dict[component_name].get_jnt_ranges()

    def get_jnt_values(self, component_name):
        return self.manipulator_dict[component_name].get_jnt_values()

    def get_gl_tcp(self, component_name):
        return self.manipulator_dict[component_name].get_gl_tcp()

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

    def fk(self, component_name, jnt_values):
        raise NotImplementedError

    def jaw_to(self, jaw_width, hnd_name='lft_hnd'):
        self.hnd_dict[hnd_name].jaw_to(jaw_width)

    def ik(self,
           component_name,
           tgt_pos,
           tgt_rot,
           seed_conf=None,
           tcp_jntid=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           local_minima="accept",
           toggle_debug=False):
        return self.manipulator_dict[component_name].ik(tgt_pos,
                                                        tgt_rot,
                                                        seed_conf=seed_conf,
                                                        tcp_jntid=tcp_jntid,
                                                        tcp_loc_pos=tcp_loc_pos,
                                                        tcp_loc_rotmat=tcp_loc_rotmat,
                                                        local_minima=local_minima,
                                                        toggle_debug=toggle_debug)

    def rand_conf(self, component_name):
        return self.manipulator_dict[component_name].rand_conf()

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

    def enable_cc(self):
        self.cc = cc.CollisionChecker("collision_checker")

    def disable_cc(self):
        """
        clear pairs and nodepath
        :return:
        """
        for cdelement in self.cc.all_cdelements:
            cdelement['cdprimit_childid'] = -1
        self.cc = None
        # self.cc.all_cdelements = []
        # for child in self.cc.np.getChildren():
        #     child.removeNode()
        # self.cc.nbitmask = 0

    def copy(self):
        self_copy = copy.deepcopy(self)
        # deepcopying colliders are problematic, I have to update it manually
        if self_copy.cc is not None:
            for child in self_copy.cc.np.getChildren():
                self_copy.cc.ctrav.addCollider(child, self_copy.cc.chan)
        return self_copy
