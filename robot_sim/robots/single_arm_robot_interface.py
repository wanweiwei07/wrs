import copy
import numpy as np
import robot_sim._kinematics.TBD_collision_checker as cc


class SglArmRbtInterface(object):
    """
    a robot is a combination of a manipulator and an end_type-effector
    author: weiwei
    date: 20230607
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='robot_interface'):
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        # collision detection
        self.cc = None
        # component map for quick access
        self.manipulator = None
        self.end_effector = None

    # def _update_oih(self):
    #     """
    #     oih = object in hand
    #     :return:
    #     author: weiwei
    #     date: 20230807
    #     """
    #     for obj_info in self.oih_infos:
    #         gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
    #         obj_info['gl_pos'] = gl_pos
    #         obj_info['gl_rotmat'] = gl_rotmat
    #
    # def _update_oof(self):
    #     """
    #     oof = object on flange
    #     this function is to be implemented by subclasses for updating ft-sensors, tool changers, end_type-effectors, etc.
    #     :return:
    #     author: weiwei
    #     date: 20230807
    #     """
    #     raise NotImplementedError

    def change_name(self, name):
        self.name = name

    def fix_to(self, pos, rotmat):
        raise NotImplementedError

    def goto_given_conf(self, jnt_values):
        return self.manipulator.goto_given_conf(jnt_values=jnt_values)

    def ik(self,
           tgt_pos: np.ndarray,
           tgt_rotmat: np.ndarray,
           seed_jnt_values=None,
           toggle_dbg=False):
        return self.manipulator.ik(tgt_pos=tgt_pos,
                                   tgt_rotmat=tgt_rotmat,
                                   seed_jnt_values=seed_jnt_values,
                                   toggle_dbg=toggle_dbg)

    def manipulability_val(self):
        return self.manipulator.manipulability_val()

    def manipulability_mat(self):
        return self.manipulator.manipulability_mat()

    def jacobian(self, jnt_values=None):
        return self.manipulator.jacobian(jnt_values=jnt_values)

    def rand_conf(self):
        return self.manipulator.rand_conf()

    def fk(self, jnt_values, toggle_jacobian=True):
        """
        no update
        :param jnt_values:
        :return:
        author: weiwei
        date: 20210417
        """
        return self.manipulator.fk(jnt_values=jnt_values, toggle_jacobian=toggle_jacobian)

    def get_jnt_values(self):
        return self.manipulator.get_jnt_values()

    def get_gl_tcp(self):
        return self.manipulator.get_gl_tcp()

    def cvt_gl_pose_to_tcp(self, gl_pos, gl_rotmat):
        return self.manipulator.cvt_gl_pose_to_tcp(gl_pos=gl_pos, gl_rotmat=gl_rotmat)

    def cvt_loc_tcp_to_gl(self):
        return self.manipulator.cvt_loc_tcp_to_gl()

    # def get_oih_list(self):
    #     return_list = []
    #     for obj_info in self.oih_infos:
    #         objcm = obj_info['collision_model']
    #         objcm.set_pos(obj_info['gl_pos'])
    #         objcm.set_rotmat(obj_info['gl_rotmat'])
    #         return_list.append(objcm)
    #     return return_list

    # def release(self, objcm, jawwidth=None):
    #     """
    #     the objcm is added as a part of the robot_s to the cd checker
    #     :param jawwidth:
    #     :param objcm:
    #     :return:
    #     """
    #     if jawwidth is not None:
    #         self.end_effector.change_jaw_width(jawwidth)
    #     for obj_info in self.oih_infos:
    #         if obj_info['collision_model'] is objcm:
    #             self.cc.delete_cdobj(obj_info)
    #             self.oih_infos.remove(obj_info)
    #             break

    def is_collided(self, obstacle_list=None, otherrobot_list=None, toggle_contact_points=False):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        :param obstacle_list:
        :param otherrobot_list:
        :param toggle_contact_points: debug
        :return: see CollisionChecker is_collided for details
        author: weiwei
        date: 20201223
        """
        if obstacle_list is None:
            obstacle_list = []
        if otherrobot_list is None:
            otherrobot_list = []
        collision_info = self.cc.is_collided(obstacle_list=obstacle_list,
                                             otherrobot_list=otherrobot_list,
                                             toggle_contacts=toggle_contact_points)
        return collision_info

    # def show_cdprim(self):
    #     self.cc.show_cdprim()
    #
    # def unshow_cdprim(self):
    #     self.cc.unshow_cdprim()

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

    def enable_cc(self):
        self.cc = cc.CollisionChecker("collision_checker")

    def disable_cc(self):
        """
        clear pairs and pdndp
        :return:
        """
        for cdelement in self.cc.cce_dict:
            cdelement['cdprimit_childid'] = -1
        self.cc = None

    def copy(self):
        self_copy = copy.deepcopy(self)
        # deepcopying colliders are problematic, I have to update it manually
        if self_copy.cc is not None:
            for child in self_copy.cc.np.getChildren():
                self_copy.cc.cd_trav.addCollider(child, self_copy.cc.cd_handler)
        return self_copy
