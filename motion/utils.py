# ==============================================
# keep jnt values decorator
# ==============================================

def keep_jnt_values_decorator(method):
    """
    decorator function for save and restore robot's joint values
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