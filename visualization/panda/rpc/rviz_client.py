import grpc
import random
import numpy as np
import visualization.panda.rpc.rviz_pb2 as rv_msg
import visualization.panda.rpc.rviz_pb2_grpc as rv_rpc


class RVizClient(object):

    def __init__(self, host="localhost:18300"):
        channel = grpc.insecure_channel(host)
        self.stub = rv_rpc.RVizStub(channel)

    def add_obj_render_info(self, obj, path=None):
        if path != None:
            create_obj_path = "obj_path = ["
            for pose in path:
                create_obj_path += "np.array(%s)," % np.array2string(pose, separator=',')
            create_obj_path = create_obj_path[:-1] + "]\n"
        else:
            create_obj_path = "obj_path = None"
        self.run_code(create_obj_path)
        code = ("obj.set_pos(np.array(%s)\n" % np.array2string(obj.get_pos(), separator=',') +
                "obj.set_rotmat(np.array(%s))\n" % np.array2string(obj.get_rotmat(), separator=',') +
                "obj.set_rgba([%s])\n" % ','.join(map(str, obj.get_rgba()))  +
                "obj_render_info_list.append(create_obj_render_info(obj=obj, obj_path=obj_path))\n")
        self.run_code(code)

    def add_robot_render_info(self, robot_jlc_name, robot_meshmodel_parameters, path):
        create_robot_path = "robot_path = ["
        for pose in path:
            create_robot_path += "np.array(%s)," % np.array2string(pose, separator=',')
        create_robot_path = create_robot_path[:-1] + "]\n"
        self.run_code(create_robot_path)
        code = ("robot_render_info_list.append(create_robot_render_info(robot_instance,\n"+
                "                                                       '%s',\n" % robot_jlc_name+
                "                                                       %s,\n" % robot_meshmodel_parameters+
                "                                                       robot_path))\n")
        self.run_code(code)

    def clear_obj_render_info_list(self):
        code = "obj_render_info_list=[]\n"
        self.run_code(code)

    def clear_robot_render_info_list(self):
        code = "robot_render_info_list=[]\n"
        self.run_code(code)

    def load_common_definition(self, file):
        with open(file, 'r') as cdfile:
            self.common_definition = cdfile.read()
        self.run_code(self.common_definition)

    def change_campos(self, campos):
        code = "base.change_campos(np.array(%s))" % np.array2string(campos, separator=',')
        self.run_code(code)

    def change_lookatpos(self, lookatpos):
        code = "base.change_lookatpos(np.array(%s))" % np.array2string(lookatpos, separator=',')
        self.run_code(code)

    def change_campos_and_lookatpos(self, campos, lookatpos):
        code = ("base.change_campos_and_lookatpos(np.array(%s), np.array(%s))" %
                (np.array2string(campos, separator=','), np.array2string(lookatpos, separator=',')))
        self.run_code(code)

    def run_code(self, code):
        """
        :param code: string
        :return:
        author: weiwei
        date: 20201229
        """
        code_bytes = code.encode('utf-8')
        return_val = self.stub.run_code(rv_msg.CodeRequest(code=code_bytes)).value
        if return_val == rv_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            return
            print("The given code is succesfully executed.")

    def clear_task(self, name="all"):
        """
        :param name:
        :return:
        """
        code = ("task_list = taskMgr.getAllTasks()\n" +
                "if '%s' == 'all':\n" % name +
                "    taskMgr.removeTasksMatching('rviz_*')" +
                "else:\n" +
                "    taskMgr.removeTasksMatching('%s')" % name)
        self.run_code(code)
