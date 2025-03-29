import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim._kinematics.collision_checker as cc
import wrs.robot_sim._kinematics.jlchain as rkjlc


class AGVInterface(object):
    """
    author: weiwei
    date: 20230607
    """

    def __init__(self, home_pos=np.zeros(3), home_rotmat=np.eye(3), name="agv_interface", enable_cc=False):
        self.name = name
        self._home_pos = home_pos
        self._home_rotmat = home_rotmat
        # jlc for fake movement
        self.movement_jlc = rkjlc.JLChain(name=name, pos=self._home_pos, rotmat=self._home_rotmat, n_dof=3)
        self.movement_jlc.home = np.zeros(3)
        self.movement_jlc.jnts[0].change_type(type=rkjlc.const.JntType.PRISMATIC)
        self.movement_jlc.jnts[0].loc_motion_ax = np.array([1, 0, 0])
        self.movement_jlc.jnts[0].loc_pos = np.zeros(3)
        self.movement_jlc.jnts[0].motion_range = np.array([-1000.0, 1000.0])
        self.movement_jlc.jnts[1].change_type(type=rkjlc.const.JntType.PRISMATIC)
        self.movement_jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
        self.movement_jlc.jnts[1].loc_pos = np.zeros(3)
        self.movement_jlc.jnts[1].motion_range = np.array([-1000.0, 1000.0])
        self.movement_jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.movement_jlc.jnts[2].loc_pos = np.zeros(3)
        self.movement_jlc.jnts[2].motion_range = [-rm.pi, rm.pi]
        self.movement_jlc.finalize()
        # cc
        if enable_cc:
            self.cc = cc.CollisionChecker("collision_checker")
        else:
            self.cc = None
        # user defined collision function
        self.userdef_collision_fn = None
        # backup
        self.jnt_values_bk = []

    @property
    def home_pos(self):
        return self._home_pos

    @property
    def home_rotmat(self):
        return self._home_rotmat

    @property
    def home_pose(self):
        return (self._home_pos, self._home_rotmat)

    def setup_cc(self):
        raise NotImplementedError

    def reset_cc(self):
        if self.cc is None:
            print("The cc is currently unavailable. Nothing to clear.")
        else:
            # create a new cc and delete the original one
            self.cc = cc.CollisionChecker("collision_checker")

    def change_name(self, name):
        self.name = name

    def backup_state(self):
        self.jnt_values_bk.append(self.movement_jlc.get_jnt_values())

    def restore_state(self):
        self.movement_jlc.goto_given_conf(jnt_values=self.jnt_values_bk.pop())

    def goto_given_conf(self, conf):
        # substract the home_pose
        conf[0] -= self._home_pos[0]
        conf[1] -= self._home_pos[1]
        # compute the rotation angle of two matrix around z axis
        conf[2] -= rm.angle_between_vectors(self._home_rotmat[:, 0], rm.const.x_ax)
        self.movement_jlc.goto_given_conf(jnt_values=conf)

    def goto_home_conf(self):
        self.movement_jlc.go_home()

    def rand_conf(self):
        return self.movement_jlc.rand_conf()

    def get_conf(self):
        mvmjlc_jnt_values = self.movement_jlc.get_jnt_values()
        # add the home_pose
        mvmjlc_jnt_values[0] += self._home_pos[0]
        mvmjlc_jnt_values[1] += self._home_pos[1]
        # compute the rotation angle of two matrix around z axis
        mvmjlc_jnt_values[2] += rm.angle_between_vectors(self._home_rotmat[:, 0], rm.const.x_ax)
        return mvmjlc_jnt_values

    def is_collided(self, obstacle_list=None, other_robot_list=None, toggle_contacts=False, toggle_dbg=False):
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
        if self.userdef_collision_fn is None:
            return self.cc.is_collided(obstacle_list=obstacle_list,
                                       other_robot_list=other_robot_list,
                                       toggle_contacts=toggle_contacts,
                                       toggle_dbg=toggle_dbg)
        else:
            result = self.cc.is_collided(obstacle_list=obstacle_list,
                                         other_robot_list=other_robot_list,
                                         toggle_contacts=toggle_contacts,
                                         toggle_dbg=toggle_dbg)
            if toggle_contacts:
                is_collided = result[0] or self.userdef_collision_fn["name"](self.userdef_collision_fn["args"])
                contacts = result[1]
                return is_collided, contacts
            else:
                return result or self.userdef_collision_fn["name"](self.userdef_collision_fn["args"])

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

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False):
        raise NotImplementedError

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False):
        raise NotImplementedError
