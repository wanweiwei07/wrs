import copy
import numpy as np
import robot_sim._kinematics.collision_checker as cc


class RobotInterface(object):
    """
    a robot is a combination of a manipulator and an end_type-effector
    author: weiwei
    date: 20230607
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='robot_interface', enable_cc=False):
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        # for dynamic callback in case of multiple arms
        self.userdef_is_collided_fn = None
        if enable_cc:
            self.cc = cc.CollisionChecker("collision_checker")
        else:
            self.cc = None

    def change_name(self, name):
        self.name = name

    def goto_given_conf(self, jnt_values):
        raise NotImplementedError

    def goto_home_conf(self):
        raise NotImplementedError

    def fix_to(self, pos, rotmat):
        raise NotImplementedError

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name='single_arm_robot_interface_stickmodel'):
        raise NotImplementedError

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='single_arm_robot_interface_meshmodel'):
        raise NotImplementedError

    def is_collided(self, obstacle_list=[], other_robot_list=[], toggle_contacts=False):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        :param obstacle_list:
        :param other_robot_list:
        :param toggle_contacts: debug
        :return: see CollisionChecker is_collided for details
        author: weiwei
        date: 20201223
        """
        # TODO cc assertion decorator
        if self.userdef_is_collided_fn is None:
            return self.cc.is_collided(obstacle_list=obstacle_list,
                                       other_robot_list=other_robot_list,
                                       toggle_contacts=toggle_contacts)
        else:
            return self.userdef_is_collided_fn(self.cc, obstacle_list=obstacle_list,
                                               other_robot_list=other_robot_list,
                                               toggle_contacts=toggle_contacts)

    def show_cdprim(self):
        """
        draw cdprim to base, you can use this function to double check if tf was correct
        :return:
        """
        # TODO cc assertion decorator
        self.cc.show_cdprim()

    def unshow_cdprim(self):
        # TODO cc assertion decorator
        self.cc.unshow_cdprim()
