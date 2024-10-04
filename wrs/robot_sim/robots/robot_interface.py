import numpy as np
import wrs.robot_sim._kinematics.collision_checker as cc


class RobotInterface(object):
    """
    a robot is a combination of a manipulator and an end_type-effector
    author: weiwei
    date: 20230607
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='robot_interface', enable_cc=False):
        self.name = name
        self._pos = pos
        self._rotmat = rotmat
        # for dynamic callback in case of multiple arms
        self.userdef_is_collided_fn = None  # deprecated 20240309 (originally designed for reusing cc, inflexible)
        if enable_cc:
            self.cc = cc.CollisionChecker("collision_checker")
        else:
            self.cc = None
        # delegator
        self.delegator = None  # use self.xxx in case of None

    @property
    def pos(self):
        return self._pos

    @property
    def rotmat(self):
        return self._rotmat

    @property
    def jnt_ranges(self):
        if self.delegator is None:
            raise AttributeError("Jnt ranges is not avialable.")
        else:
            return self.delegator.jnt_ranges

    @property
    def home_conf(self):
        if self.delegator is None:
            raise AttributeError("Home conf is not avialable.")
        else:
            return self.delegator.home_conf

    @property
    def end_effector(self):
        if self.delegator is None:
            raise AttributeError("End effector is not avialable.")
        else:
            return self.delegator.end_effector


    @property
    def gl_tcp_pos(self):
        if self.delegator is None:
            raise AttributeError("Global TCP pos is not avialable.")
        else:
            return self.delegator.gl_tcp_pos

    @property
    def gl_tcp_rotmat(self):
        if self.delegator is None:
            raise AttributeError("Global TCP rotmat is not avialable.")
        else:
            return self.delegator.gl_tcp_rotmat

    @property
    def oiee_list(self):
        if self.delegator is None:
            raise AttributeError("Oiee list is not avialable.")
        else:
            return self.delegator.oiee_list

    def backup_state(self):
        raise NotImplementedError

    def restore_state(self):
        raise NotImplementedError

    def clear_cc(self):
        if self.cc is None:
            print("The cc is currently unavailable. Nothing to clear.")
        else:
            # create a new cc and delete the original one
            self.cc = cc.CollisionChecker("collision_checker")

    def change_name(self, name):
        self.name = name

    def hold(self, obj_cmodel, **kwargs):
        if self.delegator is None:
            raise AttributeError("Hold is not available.")
        else:
            self.delegator.hold(obj_cmodel, **kwargs)

    def release(self, obj_cmodel, **kwargs):
        if self.delegator is None:
            raise AttributeError("Release is not available.")
        else:
            self.delegator.release(obj_cmodel, **kwargs)

    def goto_given_conf(self, jnt_values):
        raise NotImplementedError

    def goto_home_conf(self):
        raise NotImplementedError

    def ik(self, tgt_pos, tgt_rotmat, seed_jnt_values=None, toggle_dbg=False):
        if self.delegator is None:
            raise AttributeError("IK is not available.")
        else:
            return self.delegator.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=seed_jnt_values,
                                     toggle_dbg=toggle_dbg)

    def manipulability_val(self):
        if self.delegator is None:
            raise AttributeError("Manipulability value is not available.")
        else:
            return self.delegator.manipulability_val()

    def manipulability_mat(self):
        if self.delegator is None:
            raise AttributeError("Manipulability matrix is not available.")
        else:
            return self.delegator.manipulability_mat()

    def jacobian(self, jnt_values=None):
        if self.delegator is None:
            raise AttributeError("Jacobian is not available.")
        else:
            return self.delegator.jacobian(jnt_values=jnt_values)

    def get_ee_values(self):
        if self.delegator is None:
            raise AttributeError("Get ee values is not available.")
        else:
            return self.delegator.get_ee_values()

    def change_ee_values(self, ee_values):
        if self.delegator is None:
            raise AttributeError("Change jaw width is not available.")
        else:
            self.delegator.change_ee_values(ee_values=ee_values)

    def fix_to(self, pos, rotmat):
        raise NotImplementedError

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name='single_arm_robot_interface_stickmodel'):
        raise NotImplementedError

    def cvt_gl_pose_to_tcp(self, gl_pos, gl_rotmat):
        if self.delegator is None:
            raise AttributeError("Convert global pose to tcp is not available.")
        else:
            return self.delegator.cvt_gl_pose_to_tcp(gl_pos=gl_pos, gl_rotmat=gl_rotmat)

    def cvt_pose_in_tcp_to_gl(self, loc_pos=np.zeros(3), loc_rotmat=np.eye(3)):
        if self.delegator is None:
            raise AttributeError("Convert pose in tcp to global is not available.")
        else:
            return self.delegator.cvt_pose_in_tcp_to_gl(loc_pos=loc_pos, loc_rotmat=loc_rotmat)

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

    def is_collided(self, obstacle_list=None, other_robot_list=None, toggle_contacts=False):
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
