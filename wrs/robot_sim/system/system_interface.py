import copy
import numpy as np


class SystemInterface(object):
    """
    all functions need to be override if special target configurations like "all_robots" are needed
    author: weiwei
    date: 20230810
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='system'):
        """
        a robot system is a combination of robots, manipulators, grippers, and jlcs.
        :param pos:
        :param rotmat:
        :param name:'
        author: weiwei
        date: 20230810toyonaka
        """
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        # collision detection
        self.cc = None
        # a dictionary hosting all robots in the system.
        self.component_dict = {}
        self.current_target_configuration = None

    # TODO, do not allow specify componeent
    # TODO, access component_dict by dictionary index
    @staticmethod
    def _decorator_get(foo):
        def wrapper(self, target_configuration):
            if target_configuration is None:
                tmp_target_configuration = self.current_target_configuration
            elif self.robot_list.has_key(target_configuration):
                tmp_target_configuration = target_configuration
            else:
                # other special targets like "all_robots" can be set here
                raise ValueError(f"The given target configuration \"{target_configuration}\" is not implemented.")
            return foo(target_robot_name=tmp_target_configuration)

        return wrapper

    @_decorator_get
    def switch_target(self, target_configuration=None):
        self.current_target_configuration = target_configuration

    def change_name(self, name):
        self.name = name

    @_decorator_get
    def get_end_effector(self, target_configuration=None):
        return self.robot_list[target_configuration].end_effector

    @_decorator_get
    def get_jnt_ranges(self, target_configuration=None):
        return self.robot_list[target_configuration].get_jnt_rngs()

    @_decorator_get
    def get_jnt_values(self, target_configuration=None):
        return self.robot_list[target_configuration].get_jnt_values()

    @_decorator_get
    def get_gl_tcp(self, target_configuration=None):
        return self.robot_list[target_configuration].get_gl_tcp()

    def is_jnt_values_in_ranges(self, jnt_values):
        return self.robot_list[self.current_target_configuration].are_jnts_in_ranges(jnt_values)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        for robot in self.robot_list.values():
            robot.fix_to(pos=pos, rotmat=rotmat)

    def fk(self, jnt_values):
        """
        Define this functions case by case.
        i.e. Robots may share joints. The mimicing relation needs to be implemented in this function
        :param jnt_values:
        :return:
        """
        raise NotImplementedError()

    def ik(self,
           tgt_pos,
           tgt_rotmat,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           **kwargs):
        return self.robot_list[self.current_target_configuration].ik(tgt_pos=tgt_pos,
                                                                     tgt_rotmat=tgt_rotmat,
                                                                     tcp_loc_pos=tcp_loc_pos,
                                                                     tcp_loc_rotmat=tcp_loc_rotmat,
                                                                     **kwargs)

    @_decorator_get
    def manipulability(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       target_robot_name=None):
        return self.robot_list[self.current_target_configuration].manipulability(tcp_jnt_id=tcp_jnt_id,
                                                                                 loc_tcp_pos=tcp_loc_pos,
                                                                                 tcp_loc_rotmat=tcp_loc_rotmat)

    def manipulability_axmat(self,
                             tcp_jnt_id=None,
                             tcp_loc_pos=None,
                             tcp_loc_rotmat=None,
                             component_name='arm',
                             type="translational"):
        return self.manipulator_dict[component_name].manipulability_axmat(tcp_jnt_id=tcp_jnt_id,
                                                                          tcp_loc_pos=tcp_tloc_pos,
                                                                          tcp_loc_rotmat=tcp_loc_rotma,
                                                                          type=type)

    def jacobian(self,
                 component_name='arm',
                 tcp_jnt_id=None,
                 tcp_loc_pos=None,
                 tcp_loc_rotmat=None):
        return self.manipulator_dict[component_name].jacobian(tcp_joint_id=tcp_jnt_id,
                                                              tcp_loc_pos=tcp_loc_pos,
                                                              tcp_loc_rotmat=tcp_loc_rotmat)

    def rand_conf(self, component_name):
        return self.manipulator_dict[component_name].rand_conf()

    def cvt_conf_to_tcp(self, manipulator_name, jnt_values):
        """
        given jnt_values, this function returns the correspondent global tcp_pos, and tcp_rotmat
        :param manipulator_name:
        :param jnt_values:
        :return:
        author: weiwei
        date: 20210417
        """
        jnt_values_bk = self.get_jnt_values(manipulator_name)
        self.fk(manipulator_name, jnt_values)
        gl_tcp_pos, gl_tcp_rotmat = self.robot_s.get_gl_tcp(manipulator_name)
        self.fk(manipulator_name, jnt_values_bk)
        return gl_tcp_pos, gl_tcp_rotmat

    def cvt_gl_to_loc_tcp(self, manipulator_name, gl_obj_pos, gl_obj_rotmat):
        return self.manipulator_dict[manipulator_name].cvt_gl_pose_to_tcp(gl_obj_pos, gl_obj_rotmat)

    def cvt_loc_tcp_to_gl(self, manipulator_name, rel_obj_pos, rel_obj_rotmat):
        return self.manipulator_dict[manipulator_name].cvt_loc_tcp_to_gl(rel_obj_pos, rel_obj_rotmat)

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
                                             other_robot_list=otherrobot_list,
                                             toggle_contacts=toggle_contact_points)
        return collision_info

    def show_cdprimit(self):
        self.cc.show_cdprim()

    def unshow_cdprimit(self):
        self.cc.unshow_cdprim()

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcp_frame=False,
                       toggle_jnt_frame=False,
                       toggle_connjnt=False,
                       name='robot_interface_stickmodel'):
        raise NotImplementedError

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frame=False,
                      rgba=None,
                      name='robot_interface_meshmodel'):
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


class SystemInterface(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='robot_interface'):
        # TODO self.jlcs = {}
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        # collision detection
        self.cc = None
        # component map for quick access
        self.manipulator_dict = {}
        self.ft_sensor_dict = {}
        self.hnd_dict = {}

    def change_name(self, name):
        self.name = name

    def get_hnd_on_manipulator(self, manipulator_name):
        raise NotImplementedError

    def get_jnt_ranges(self, component_name):
        return self.manipulator_dict[component_name].get_jnt_rngs()

    def get_jnt_values(self, component_name):
        return self.manipulator_dict[component_name].get_jnt_values()

    def get_gl_tcp(self, manipulator_name):
        return self.manipulator_dict[manipulator_name].get_gl_tcp()

    def is_jnt_values_in_ranges(self, component_name, jnt_values):
        return self.manipulator_dict[component_name].are_jnts_in_ranges(jnt_values)

    def fix_to(self, pos, rotmat):
        return NotImplementedError

    def fk(self, component_name, jnt_values):
        return NotImplementedError

    def jaw_to(self, hnd_name, jaw_width):
        self.hnd_dict[hnd_name].change_jaw_width(jaw_width=jaw_width)

    def get_jawwidth(self, hand_name):
        return self.hnd_dict[hand_name].get_jaw_width()

    def ik(self,
           component_name: str = "arm",
           tgt_pos=np.zeros(3),
           tgt_rotmat=np.eye(3),
           seed_jnt_values=None,
           max_niter=200,
           tcp_jnt_id=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           local_minima: str = "end_type",
           toggle_debug=False):
        return self.manipulator_dict[component_name].ik(tgt_pos,
                                                        tgt_rotmat,
                                                        seed_jnt_values=seed_jnt_values,
                                                        max_n_iter=max_niter,
                                                        tcp_joint_id=tcp_jnt_id,
                                                        tcp_loc_pos=tcp_loc_pos,
                                                        tcp_loc_rotmat=tcp_loc_rotmat,
                                                        local_minima=local_minima,
                                                        toggle_dbg=toggle_debug)

    def manipulability(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       component_name='arm'):
        return self.manipulator_dict[component_name].manipulability(tcp_jnt_id=tcp_jnt_id,
                                                                    loc_tcp_pos=tcp_loc_pos,
                                                                    tcp_loc_rotmat=tcp_loc_rotmat)

    def manipulability_axmat(self,
                             tcp_jnt_id=None,
                             tcp_loc_pos=None,
                             tcp_loc_rotmat=None,
                             component_name='arm',
                             type="translational"):
        return self.manipulator_dict[component_name].manipulability_axmat(tcp_jnt_id=tcp_jnt_id,
                                                                          tcp_loc_pos=tcp_tloc_pos,
                                                                          tcp_loc_rotmat=tcp_loc_rotma,
                                                                          type=type)

    def jacobian(self,
                 component_name='arm',
                 tcp_jnt_id=None,
                 tcp_loc_pos=None,
                 tcp_loc_rotmat=None):
        return self.manipulator_dict[component_name].jacobian(tcp_joint_id=tcp_jnt_id,
                                                              tcp_loc_pos=tcp_loc_pos,
                                                              tcp_loc_rotmat=tcp_loc_rotmat)

    def rand_conf(self, component_name):
        return self.manipulator_dict[component_name].rand_conf()

    def cvt_conf_to_tcp(self, manipulator_name, jnt_values):
        """
        given jnt_values, this function returns the correspondent global tcp_pos, and tcp_rotmat
        :param manipulator_name:
        :param jnt_values:
        :return:
        author: weiwei
        date: 20210417
        """
        jnt_values_bk = self.get_jnt_values(manipulator_name)
        self.fk(manipulator_name, jnt_values)
        gl_tcp_pos, gl_tcp_rotmat = self.robot_s.get_gl_tcp(manipulator_name)
        self.fk(manipulator_name, jnt_values_bk)
        return gl_tcp_pos, gl_tcp_rotmat

    def cvt_gl_to_loc_tcp(self, manipulator_name, gl_obj_pos, gl_obj_rotmat):
        return self.manipulator_dict[manipulator_name].cvt_gl_pose_to_tcp(gl_obj_pos, gl_obj_rotmat)

    def cvt_loc_tcp_to_gl(self, manipulator_name, rel_obj_pos, rel_obj_rotmat):
        return self.manipulator_dict[manipulator_name].cvt_loc_tcp_to_gl(rel_obj_pos, rel_obj_rotmat)

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
                                             other_robot_list=otherrobot_list,
                                             toggle_contacts=toggle_contact_points)
        return collision_info

    def show_cdprimit(self):
        self.cc.show_cdprim()

    def unshow_cdprimit(self):
        self.cc.unshow_cdprim()

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcp_frame=False,
                       toggle_jnt_frame=False,
                       toggle_connjnt=False,
                       name='robot_interface_stickmodel'):
        raise NotImplementedError

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frame=False,
                      rgba=None,
                      name='robot_interface_meshmodel'):
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
