from functools import cached_property


class MotionData(object):

    def __init__(self, robot):
        self.robot = robot
        self._jv_list = []  # a list of jnt values
        self._ev_list = []  # a list of end effector values
        self._mesh_list = []
        self._oiee_gl_pose_list = []

    @property
    def jv_list(self):
        return self._jv_list

    @property
    def ev_list(self):
        return self._ev_list

    @property
    def oiee_gl_pose_list(self):
        return self._oiee_gl_pose_list

    @property
    def mesh_list(self):
        return self._mesh_list

    @cached_property
    def tcp_list(self):
        return [self.robot.fk(jv) for jv in self._jv_list]

    def extend(self, jv_list, toggle_mesh=True):
        """
        :param jv_list:
        :param ev_list:
        :param mesh_list: auto gen if None, fill with None if [], assign other wise
        :return:
        """
        # joint value list
        self._jv_list += jv_list
        # end effector value list
        self._ev_list += [self.robot.get_ee_values()] * len(jv_list)
        # oiee and meshxs
        if toggle_mesh:
            oiee_gl_pose_list = []
            tmp_mesh_list = []
            self.robot.backup_state()
            for i, jnt_values in enumerate(jv_list):
                # if ev_list is None:
                self.robot.goto_given_conf(jnt_values=jnt_values)
                # else:
                #     self.robot.goto_given_conf(jnt_values=jnt_values, ee_values=ev_list[i])
                tmp_mesh_list.append(self.robot.gen_meshmodel())
                if len(self.robot.oiee_list) == 0:
                    oiee_gl_pose_list.append(None)
                else:
                    oiee_gl_pose_list.append([self.robot.oiee_list[-1].gl_pos, self.robot.oiee_list[-1].gl_rotmat])
            self.robot.restore_state()
            self._mesh_list += tmp_mesh_list
            self._oiee_gl_pose_list += oiee_gl_pose_list
        else:
            self._mesh_list += [None] * len(jv_list)
            if len(self.robot.oiee_list) == 0:
                self._oiee_gl_pose_list += [None] * len(jv_list)
            else:
                oiee_gl_pose_list= []
                self.robot.backup_state()
                for i, jnt_values in enumerate(jv_list):
                    if ev_list is None:
                        self.robot.goto_given_conf(jnt_values=jnt_values)
                    else:
                        self.robot.goto_given_conf(jnt_values=jnt_values, ee_values=ev_list[i])
                    oiee_gl_pose_list.append([self.robot.oiee_list[-1].gl_pos, self.robot.oiee_list[-1].gl_rotmat])
                self.robot.restore_state()
                self._oiee_gl_pose_list += oiee_gl_pose_list

    def __len__(self):
        return len(self._jv_list)

    def __add__(self, other):
        if self.robot is other.robot:
            self._jv_list += other.jv_list
            self._ev_list += other.ev_list
            self._oiee_gl_pose_list += other.oiee_gl_pose_list
            self._mesh_list += other.mesh_list
            return self
        else:
            raise ValueError("Motion data for different robots cannot be concatenated.")

    def __str__(self):
        out_str = (f"Total: {len(self._jv_list)}\n" +
                   "Configuration List:\n-----------------------")
        for jnt_values in self._jv_list:
            out_str += ("\n" + repr(jnt_values))
        out_str += ("\n\n" + "EE Values List:\n-----------------------")
        for ee_values in self._ev_list:
            out_str += f"\n ee_values={ee_values}"
        out_str += ("\n\n" + "Obj Pose List:\n-----------------------")
        for oiee_gl_pose in self._oiee_gl_pose_list:
            out_str += f"\n obj_pose={oiee_gl_pose}"
        return out_str

    def __iter__(self):
        for item in self._jv_list:
            yield item

    def __getitem__(self, index):
        return self._jv_list[index], self._ev_list[index], self._oiee_gl_pose_list[index], self._mesh_list[index]


# class ManipulationData(MotionData):
#
#     def __init__(self, robot, obj_cmodel=None):
#         super().__init__(robot)
#         self.obj_cmodel = obj_cmodel
#
#     @staticmethod
#     def create_from_motion_data(motion_data, obj_cmodel=None, obj_pose_list=None):
#         """
#         :param motion_data:
#         :param obj_cmodel:
#         :param obj_pose_list:
#         :return:
#         """
#         manipulation_data = ManipulationData(motion_data.robot)
#         manipulation_data.extend(motion_data.jv_list, motion_data.ev_list, motion_data.mesh_list)
#         manipulation_data.obj_cmodel = obj_cmodel
#         manipulation_data._obj_pose_list = obj_pose_list
#         # update mesh_list ?
#         return manipulation_data
#
#     @property
#     def obj_pose_list(self):
#         return self._obj_pose_list
#
#     def extend(self, jv_list, ev_list=None, obj_pose_list=None, mesh_list=None):
#         """
#         :param jv_list:
#         :param ev_list:
#         :param mesh_list: auto gen if None, fill with None if [], assign other wise
#         :return:
#         """
#         super().extend(jv_list, ev_list, mesh_list)
#         # object pose list
#         if obj_pose_list is not None:
#             self._obj_pose_list += obj_pose_list
#         elif len(obj_pose_list) == 0:
#             self._obj_pose_list += [None] * len(jv_list)
#         else:
#             self._obj_pose_list += obj_pose_list
#
#     def __len__(self):
#         return len(self._jv_list)
#
#     def __add__(self, other):
#         if self.robot is other.robot:
#             self._jv_list += other.jv_list
#             self._ev_list += other.ev_list
#             self._mesh_list += other.mesh_list
#             self._obj_pose_list += other.obj_pose_list
#             return self
#         else:
#             raise ValueError("Motion data for different robots cannot be concatenated.")
#
#     def __str__(self):
#         out_str = (f"Total: {len(self._jv_list)}\n" +
#                    "Configuration List:\n-----------------------")
#         for jnt_values in self._jv_list:
#             out_str += ("\n" + repr(jnt_values))
#         out_str += ("\n" + "EE Values List:\n-----------------------")
#         for ee_values in self._ev_list:
#             out_str += f"\n ee_values={ee_values}"
#         if self.obj_cmodel is not None:
#             out_str += ("\n" + "Obj Pose List:\n-----------------------")
#             for obj_pose in self._obj_pose_list:
#                 out_str += f"\n obj_pose={obj_pose}"
#         return out_str
#
#     def __iter__(self):
#         for item in self._jv_list:
#             yield item
#
#     def __getitem__(self, index):
#         return self._jv_list[index], self._ev_list[index], self._obj_pose_list[index], self._mesh_list[index]


def keep_states_decorator(method):
    """
    decorator function for save and restore robot's joint values
    applicable to both single or multi-arm sgl_arm_robots
    :return:
    author: weiwei
    date: 20220404
    """

    def wrapper(self, *args, **kwargs):
        self.robot.backup_state()
        result = method(self, *args, **kwargs)
        self.robot.restore_state()
        return result

    return wrapper


def keep_states_objpose_decorator(method):
    """
    decorator function for save and restore robot's joints and jaw values
    only applicable to single-arm robot
    :return:
    author: weiwei
    date: 20240312
    """

    def wrapper(self, *args, **kwargs):
        self.robot.backup_state()
        obj_pose_bk = kwargs["obj_cmodel"].pose
        result = method(self, *args, **kwargs)
        self.robot.restore_state()
        kwargs["obj_cmodel"].pose = obj_pose_bk
        return result

    return wrapper
