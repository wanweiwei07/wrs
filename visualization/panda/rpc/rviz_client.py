import grpc
import random
import numpy as np
import visualization.panda.rpc.rviz_pb2 as rv_msg
import visualization.panda.rpc.rviz_pb2_grpc as rv_rpc


class RVizClient(object):

    def __init__(self, host="localhost:18300"):
        channel = grpc.insecure_channel(host)
        self.stub = rv_rpc.RVizStub(channel)

    def _gen_random_name(self):
        return 'rmt_'+str(random.randint(0,1e6))

    def add_anime_obj(self, rmt_obj, loc_obj, loc_obj_path):
        create_obj_path = "obj_path = ["
        for pose in loc_obj_path:
            create_obj_path += "np.array(%s)," % np.array2string(pose, separator=',')
        create_obj_path = create_obj_path[:-1] + "]\n"
        self.run_code(create_obj_path)
        code = ("%s.set_pos(np.array(%s)\n" % (rmt_obj, np.array2string(loc_obj.get_pos(), separator=',')) +
                "%s.set_rotmat(np.array(%s))\n" % (rmt_obj, np.array2string(loc_obj.get_rotmat(), separator=',')) +
                "%s.set_rgba([%s])\n" % (rmt_obj, ','.join(map(str, loc_obj.get_rgba()))) +
                "base.attach_manualupdate_obj(wd.ani.ObjInfo.create_obj_anime_info(obj=%s, obj_path=obj_path))\n" % rmt_obj)
        self.run_code(code)

    def add_anime_robot(self, rmt_robot_instance, loc_robot_jlc_name, loc_robot_meshmodel_parameters, loc_robot_path):
        create_robot_path = "robot_path = ["
        for pose in loc_robot_path:
            create_robot_path += "np.array(%s)," % np.array2string(pose, separator=',')
        create_robot_path = create_robot_path[:-1] + "]\n"
        self.run_code(create_robot_path)
        code = ("base.attach_manualupdate_robot(wd.ani.RobotInfo.create_robot_anime_info(%s,\n" % rmt_robot_instance +
                "'%s',\n" % loc_robot_jlc_name + "%s,\n" % loc_robot_meshmodel_parameters + "robot_path))\n")
        self.run_code(code)

    def delete_anime_obj(self, rmt_obj):
        code = "base.detach_manualupdate_obj(%s)\n" % rmt_obj
        self.run_code(code)

    def delete_anime_robot(self, rmt_robot_instance):
        code = "base.detach_manualupdate_robot(%s)\n" % rmt_robot_instance
        self.run_code(code)

    # def add_stationary_obj(self, rmt_obj, loc_obj):
    #     code = ("%s.set_pos(np.array(%s)\n" % (rmt_obj, np.array2string(loc_obj.get_pos(), separator=',')) +
    #             "%s.set_rotmat(np.array(%s))\n" % (rmt_obj, np.array2string(loc_obj.get_rotmat(), separator=',')) +
    #             "%s.set_rgba([%s])\n" % (rmt_obj, ','.join(map(str, loc_obj.get_rgba()))) +
    #             "base.obj_anime_info_list.append(wd.ani.ObjInfo.create_obj_anime_info(obj=%s, obj_path=obj_path))\n" % rmt_obj)
    #     code = "base."
    #
    def add_stationary_robot(self, rmt_robot_instance, loc_robot_instance):
        """
        :param rmt_robot_instance:
        :param loc_robot_instance:
        :return: The name of the robot_meshmodel created in the remote end
        """
        random_rmt_robot_meshmodel_name = self._gen_random_name()
        jnt_angles_str = np.array2string(loc_robot_instance.get_jntvalues(jlc_name='all'), separator=',')
        code = ("%s.fk(jnt_values=np.array(%s), jlc_name='all')\n" % (rmt_robot_instance, jnt_angles_str) +
                "%s = %s.gen_meshmodel()\n" % (random_rmt_robot_meshmodel_name, rmt_robot_instance) +
                "base.attach_autoupdate_robot(%s)\n" % random_rmt_robot_meshmodel_name)
        self.run_code(code)
        return random_rmt_robot_meshmodel_name

    def delete_stationary_robot(self, rmt_robot_meshmodel):
        code = "base.detach_autoupdate_robot(%s)" % rmt_robot_meshmodel
        self.run_code(code)

    def load_common_definition(self, file):
        with open(file, 'r') as cdfile:
            self.common_definition = cdfile.read()
        # exec at remote
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
