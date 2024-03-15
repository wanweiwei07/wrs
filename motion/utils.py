class MotionData(object):

    def __init__(self, robot):
        self.robot = robot
        self._conf_list = []
        self._mesh_list = []

    @property
    def conf_list(self):
        return self._conf_list

    @property
    def mesh_list(self):
        return self._mesh_list

    def extend(self, conf_list, mesh_list=None):
        self._conf_list += conf_list
        if mesh_list is None:
            tmp_mesh_list = []
            self.robot.backup_state()
            for conf in conf_list:
                self.robot.goto_given_conf(jnt_values=conf)
                tmp_mesh_list.append(self.robot.gen_meshmodel())
            self.robot.restore_state()
            self._mesh_list += tmp_mesh_list
        else:
            self._mesh_list += mesh_list

    def __len__(self):
        return len(self._conf_list)

    def __add__(self, other):
        if self.robot is other.robot:
            self._conf_list += other.conf_list
            self._mesh_list += other.mesh_lsit
            return self
        else:
            raise ValueError("Motion data for different robots cannot be concatenated.")

    def __str__(self):
        out_str = (f"Total: {len(self._conf_list)}\n" +
                   "Configuration List:\n-----------------------")
        for conf in self._conf_list:
            out_str += ("\n" + repr(conf))
        return out_str

    def __iter__(self):
        for item in self._conf_list:
            yield item


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
