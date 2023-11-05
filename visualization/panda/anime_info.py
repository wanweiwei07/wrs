class RobotInfo(object):

    def __init__(self):
        self.robot_s = None
        self.robot_component_name = None
        self.robot_meshmodel = None
        self.robot_meshmodel_parameters = None
        self.robot_path = None
        self.robot_path_counter = None

    @staticmethod
    def create_anime_info(robot_s,
                          robot_component_name,
                          robot_meshmodel_parameters,
                          robot_path):
        anime_info = RobotInfo()
        anime_info.robot_s = robot_s
        anime_info.robot_component_name = robot_component_name
        anime_info.robot_meshmodel = robot_s.gen_mesh_model(robot_meshmodel_parameters)
        anime_info.robot_meshmodel_parameters = robot_meshmodel_parameters
        anime_info.robot_path = robot_path
        anime_info.robot_path_counter = 0
        return anime_info


class ObjInfo(object):

    def __init__(self):
        self.obj = None
        self.obj_parameters = None
        self.obj_path = None
        self.obj_path_counter = None

    @staticmethod
    def create_anime_info(obj, obj_path=None):
        anime_info = ObjInfo()
        anime_info.obj = obj
        anime_info.obj_parameters = [obj.get_rgba()]
        if obj_path is None:
            anime_info.obj_path = [[obj.get_pos(), obj.get_rotmat()]]
        else:
            anime_info.obj_path = obj_path
        anime_info.obj_path_counter = 0
        return anime_info
