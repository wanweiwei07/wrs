

class RobotInfo(object):

    def __init__(self):
        self.robot_instance = None
        self.robot_jlc_name = None
        self.robot_meshmodel = None
        self.robot_meshmodel_parameters = None
        self.robot_path = None
        self.robot_path_counter = None

class ObjInfo(object):

    def __init__(self):
        self.obj = None
        self.obj_parameters = None
        self.obj_path = None
        self.obj_path_counter = None