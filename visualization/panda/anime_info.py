class RobotInfo(object):

    def __init__(self):
        self.robot_instance = None
        self.robot_jlc_name = None
        self.robot_meshmodel = None
        self.robot_meshmodel_parameters = None
        self.robot_path = None
        self.robot_path_counter = None

    @staticmethod
    def create_robot_anime_info(robot_instance,
                                robot_jlc_name,
                                robot_meshmodel_parameters,
                                robot_path):
        robot_render_info = RobotInfo()
        robot_render_info.robot_instance = robot_instance
        robot_render_info.robot_jlc_name = robot_jlc_name
        robot_render_info.robot_meshmodel = robot_instance.gen_meshmodel(robot_meshmodel_parameters)
        robot_render_info.robot_meshmodel_parameters = robot_meshmodel_parameters
        robot_render_info.robot_path = robot_path
        robot_render_info.robot_path_counter = 0
        return robot_render_info


class ObjInfo(object):

    def __init__(self):
        self.obj = None
        self.obj_parameters = None
        self.obj_path = None
        self.obj_path_counter = None

    @staticmethod
    def create_obj_anime_info(obj,
                              obj_path=None):
        obj_render_info = ObjInfo()
        obj_render_info.obj = obj
        obj_render_info.obj_parameters = obj.get_rgba()
        if obj_path is None:
            obj_render_info.obj_path = [[obj.get_pos(), obj.get_rotmat()]]
        else:
            obj_render_info.obj_path = obj_path
        obj_render_info.obj_path_counter = 0
        return obj_render_info
