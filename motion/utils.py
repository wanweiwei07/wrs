# ==============================================
# keep jnt values decorator
# ==============================================

def keep_jnt_values_decorator(method):
    """
    decorator function for save and restore robot's joint values
    applicable to both single or multi-arm robots
    :return:
    author: weiwei
    date: 20220404
    """

    def wrapper(self, *args, **kwargs):
        jnt_values_bk = self.robot.get_jnt_values()
        result = method(self, *args, **kwargs)
        self.robot.goto_given_conf(jnt_values=jnt_values_bk)
        return result

    return wrapper

def keep_jnt_jaw_values_decorator(method):
    """
    decorator function for save and restore robot's joints and jaw values
    only applicable to single-arm robot
    :return:
    author: weiwei
    date: 20240312
    """

    def wrapper(self, *args, **kwargs):
        jnt_values_bk = self.robot.get_jnt_values()
        jaw_width_bk = self.robot.get_jaw_width()
        result = method(self, *args, **kwargs)
        self.robot.goto_given_conf(jnt_values=jnt_values_bk)
        self.robot.change_jaw_width(jaw_width=jaw_width_bk)
        return result

    return wrapper

def keep_jnt_jaw_objpose_values_decorator(method):
    """
    decorator function for save and restore robot's joints and jaw values
    only applicable to single-arm robot
    :return:
    author: weiwei
    date: 20240312
    """

    def wrapper(self, *args, **kwargs):
        jnt_values_bk = self.robot.get_jnt_values()
        jaw_width_bk = self.robot.get_jaw_width()
        obj_pose_bk = kwargs["obj_cmodel"].pose
        result = method(self, *args, **kwargs)
        self.robot.goto_given_conf(jnt_values=jnt_values_bk)
        self.robot.release(kwargs["obj_cmodel"])
        self.robot.change_jaw_width(jaw_width=jaw_width_bk)
        kwargs["obj_cmodel"].pose=obj_pose_bk
        return result

    return wrapper