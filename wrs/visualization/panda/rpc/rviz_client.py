import pickle
import grpc
import random
import numpy as np
import wrs.visualization.panda.rpc.rviz_pb2 as rv_msg
import wrs.visualization.panda.rpc.rviz_pb2_grpc as rv_rpc
from wrs import robot_sim as ri, modeling as gm


class RVizClient(object):

    def __init__(self, host="localhost:18300"):
        channel = grpc.insecure_channel(host)
        self.stub = rv_rpc.RVizStub(channel)
        # self.rmt_mesh_list = [] # TODO move to server side

    def _gen_random_name(self, prefix):
        return prefix + str(random.randint(100000, 1e6))  # 6 digits

    def reset(self):
        # code = "base.clear_internal_update_obj()\n"
        # code += "base.clear_internal_update_robot()\n"
        code = "base.clear_external_update_obj()\n"
        code += "base.clear_external_update_robot()\n"
        code += "base.clear_noupdate_model()\n"
        # code += "for item in [%s]:\n" % ', '.join(self.rmt_mesh_list)
        # code += "    item.detach()"
        # self.rmt_mesh_list = []
        self.run_code(code)

    def run_code(self, code):
        """
        :param code: string
        :return:
        author: weiwei
        date: 20201229
        """
        print(code)
        code_bytes = code.encode('utf-8')
        return_val = self.stub.run_code(rv_msg.CodeRequest(code=code_bytes)).value
        if return_val == rv_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            return

    def load_common_definition(self, file, line_ids=None):
        """
        :param file:
        :param line_ids: [1:3], load lines before main if None, else load specified line_ids
        :return:
        """
        with open(file, 'r') as cdfile:
            if line_ids is None:
                tmp_text = cdfile.read()
                main_idx = tmp_text.find('if __name__')
                self.common_definition = tmp_text[:main_idx]
            else:
                self.common_definition = ""
                tmp_text_lines = cdfile.readlines()
                for line_id in line_ids:
                    self.common_definition += tmp_text_lines[line_id - 1]
        # exec at remote
        print(self.common_definition)
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

    def copy_to_remote(self, loc_instance, given_rmt_robot_s_name=None):
        if given_rmt_robot_s_name is None:
            given_rmt_robot_s_name = self._gen_random_name(prefix='rmt_robot_s_')
        if isinstance(loc_instance, ri.RobotInterface):
            loc_instance.disable_cc()
            self.stub.create_instance(rv_msg.CreateInstanceRequest(name=given_rmt_robot_s_name,
                                                                   data=pickle.dumps(loc_instance)))
            loc_instance.enable_cc()
        else:
            self.stub.create_instance(rv_msg.CreateInstanceRequest(name=given_rmt_robot_s_name,
                                                                   data=pickle.dumps(loc_instance)))
        return given_rmt_robot_s_name

    def update_remote(self, rmt_instance, loc_instance):
        if isinstance(loc_instance, ri.RobotInterface):
            code = ("%s.fk(jnt_values=np.array(%s), hnd_name='all')\n" %
                    (rmt_instance, np.array2string(loc_instance.get_jnt_values(), separator=',')))
        elif isinstance(loc_instance, gm.GeometricModel):
            code = ("%s.set_pos(np.array(%s))\n" % (
            rmt_instance, np.array2string(loc_instance.get_pos(), separator=',')) +
                    "%s.set_rotmat(np.array(%s))\n" % (
                    rmt_instance, np.array2string(loc_instance.get_rotmat(), separator=',')) +
                    "%s.set_rgba([%s])\n" % (rmt_instance, ','.join(map(str, loc_instance.get_rgba()))))
        elif isinstance(loc_instance, gm.StaticGeometricModel):
            code = "%s.set_rgba([%s])\n" % (rmt_instance, ','.join(map(str, loc_instance.get_rgba())))
        else:
            raise ValueError
        self.run_code(code)

    def show_model(self, rmt_mesh):
        code = "base.attach_noupdate_model(%s)\n" % rmt_mesh
        # code = "%s.attach_to(base)\n" % rmt_mesh
        # self.rmt_mesh_list.append(rmt_mesh)
        self.run_code(code)

    def unshow_model(self, rmt_mesh):
        code = "base.detach_noupdate_model(%s)\n" % rmt_mesh
        # code = "%s.detach()\n" % rmt_mesh
        # self.rmt_mesh_list.remove(rmt_mesh)
        self.run_code(code)

    def showmodel_to_remote(self, loc_mesh, given_rmt_mesh_name=None):
        """
        helper function that merges copy_to_remote, show_instance, and unshow_instance
        :param loc_mesh:
        :param given_rmt_mesh_name:
        :return:
        author: weiwei
        date: 20201231
        """
        rmt_mesh = self.copy_to_remote(loc_instance=loc_mesh, given_rmt_robot_s_name=given_rmt_mesh_name)
        self.show_model(rmt_mesh)
        return rmt_mesh

    def unshowmodel_from_remote(self, rmt_mesh):
        """
        for symmetry purpose
        :param rmt_mesh:
        :return:
        """
        self.unshow_model(rmt_mesh)

    def add_anime_obj(self,
                      rmt_obj,
                      loc_obj,
                      loc_obj_path,
                      given_rmt_anime_objinfo_name=None):
        """
        add wd.ani.ObjInfo to base._external_update_obj_list
        :param rmt_obj: str
        :param loc_obj: CollisionModel, Static/GeometricModel, ModelCollection
        :param loc_obj_path:
        :param given_rmt_anime_objinfo: str
        :return: rmt_anime_objinfo
        author: weiwei
        date: 20201231
        """
        if given_rmt_anime_objinfo_name is None:
            given_rmt_anime_objinfo_name = self._gen_random_name(prefix='rmt_anime_objinfo_')
        code = "obj_path = ["
        for pose in loc_obj_path:
            pos, rotmat = pose
            code += "[np.array(%s), np.array(%s)]," % (
            np.array2string(pos, separator=','), np.array2string(rotmat, separator=','))
        code = code[:-1] + "]\n"
        code += ("%s.set_pos(np.array(%s))\n" % (rmt_obj, np.array2string(loc_obj.get_pos(), separator=',')) +
                 "%s.set_rotmat(np.array(%s))\n" % (rmt_obj, np.array2string(loc_obj.get_rotmat(), separator=',')) +
                 "%s.set_rgba([%s])\n" % (rmt_obj, ','.join(map(str, loc_obj.get_rgba()))) +
                 "%s = wd.ani.ObjInfo.create_anime_info(obj=%s, obj_path=obj_path)\n" %
                 (given_rmt_anime_objinfo_name, rmt_obj))
        code += "base.attach_external_update_obj(%s)\n" % given_rmt_anime_objinfo_name
        self.run_code(code)
        return given_rmt_anime_objinfo_name

    def add_anime_robot(self,
                        rmt_robot_s,
                        loc_robot_component_name,
                        loc_robot_meshmodel_parameters,
                        loc_robot_motion_path,
                        given_rmt_anime_robotinfo_name=None):
        """
        :param rmt_robot_s:
        :param loc_robot_component_name:
        :param loc_robot_meshmodel_parameters:
        :param loc_robot_motion_path:
        :param given_rmt_anime_robotinfo_name:
        :return: remote anime_robotinfo
        author: weiwei
        date: 20201231
        """
        if given_rmt_anime_robotinfo_name is None:
            given_rmt_anime_robotinfo_name = self._gen_random_name(prefix='rmt_anime_robotinfo_')
        code = "robot_path = ["
        for pose in loc_robot_motion_path:
            code += "np.array(%s)," % np.array2string(pose, separator=',')
        code = code[:-1] + "]\n"
        code += ("%s = wd.ani.RobotInfo.create_anime_info(%s, " %
                 (given_rmt_anime_robotinfo_name, rmt_robot_s) +
                 "'%s', " % loc_robot_component_name +
                 "%s, " % loc_robot_meshmodel_parameters + "robot_path)\n")
        code += "base.attach_external_update_robot(%s)\n" % given_rmt_anime_robotinfo_name
        self.run_code(code)
        return given_rmt_anime_robotinfo_name

    def delete_anime_obj(self, rmt_anime_objinfo):
        code = "base.detach_external_update_obj(%s)\n" % rmt_anime_objinfo
        self.run_code(code)

    def delete_anime_robot(self, rmt_anime_robotinfo):
        code = "base.detach_external_update_robot(%s)\n" % rmt_anime_robotinfo
        self.run_code(code)

    def add_stationary_obj(self,
                           rmt_obj,
                           loc_obj):
        code = ("%s.set_pos(np.array(%s)\n" % (rmt_obj, np.array2string(loc_obj.get_pos(), separator=',')) +
                "%s.set_rotmat(np.array(%s))\n" % (rmt_obj, np.array2string(loc_obj.get_rotmat(), separator=',')) +
                "%s.set_rgba([%s])\n" % (rmt_obj, ','.join(map(str, loc_obj.get_rgba()))) +
                "base.attach_noupdate_model(%s)\n" % (rmt_obj))
        self.run_code(code)

    def delete_stationary_obj(self, rmt_obj):
        code = "base.delete_noupdate_model(%s)" % rmt_obj
        self.run_code(code)

    def add_stationary_robot(self,
                             rmt_robot_s,
                             loc_robot_s,
                             given_rmt_robot_meshmodel_name=None):
        """
        :param rmt_robot_s:
        :param loc_robot_s:
        :param given_rmt_robot_meshmodel_name: str, a random name will be generated if None
        :return: The name of the robot_meshmodel created in the remote end_type
        """
        if given_rmt_robot_meshmodel_name is None:
            given_rmt_robot_meshmodel_name = self._gen_random_name(prefix='rmt_robot_meshmodel_')
        jnt_angles_str = np.array2string(loc_robot_s.get_jnt_values(component_name='all'), separator=',')
        code = ("%s.fk(hnd_name='all', jnt_values=np.array(%s))\n" % (rmt_robot_s, jnt_angles_str) +
                "%s = %s.gen_meshmodel()\n" % (given_rmt_robot_meshmodel_name, rmt_robot_s) +
                "base.attach_noupdate_model(%s)\n" % given_rmt_robot_meshmodel_name)
        self.run_code(code)
        return given_rmt_robot_meshmodel_name

    def delete_stationary_robot(self, rmt_robot_meshmodel):
        code = "base.delete_noupdate_model(%s)" % rmt_robot_meshmodel
        self.run_code(code)
