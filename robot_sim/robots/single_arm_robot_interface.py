import numpy as np
import robot_sim._kinematics.collision_checker as cc
import robot_sim.robots.robot_interface as ri

class SglArmRobotInterface(ri.RobotInterface):
    """
    a robot is a combination of a manipulator and an end_type-effector
    author: weiwei
    date: 20230607
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='robot_interface', enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        self.manipulator = None
        self.end_effector = None

    @property
    def gl_tcp_pos(self):
        return self.manipulator.gl_tcp_pos

    @property
    def gl_tcp_rotmat(self):
        return self.manipulator.gl_tcp_rotmat

    def _update_end_effector(self):
        self.end_effector.fix_to(pos=self.manipulator.gl_flange_pos, rotmat=self.manipulator.gl_flange_rotmat)

    def goto_given_conf(self, jnt_values):
        result = self.manipulator.goto_given_conf(jnt_values=jnt_values)
        self._update_end_effector()
        return result

    def ik(self,
           tgt_pos: np.ndarray,
           tgt_rotmat: np.ndarray,
           seed_jnt_values=None,
           toggle_dbg=False):
        return self.manipulator.ik(tgt_pos=tgt_pos,
                                   tgt_rotmat=tgt_rotmat,
                                   seed_jnt_values=seed_jnt_values,
                                   toggle_dbg=toggle_dbg)

    def manipulability_val(self):
        return self.manipulator.manipulability_val()

    def manipulability_mat(self):
        return self.manipulator.manipulability_mat()

    def jacobian(self, jnt_values=None):
        return self.manipulator.jacobian(jnt_values=jnt_values)

    def rand_conf(self):
        return self.manipulator.rand_conf()

    def fk(self, jnt_values, toggle_jacobian=True):
        """
        no update
        :param jnt_values:
        :return:
        author: weiwei
        date: 20210417
        """
        return self.manipulator.fk(jnt_values=jnt_values, toggle_jacobian=toggle_jacobian)

    def get_jnt_values(self):
        return self.manipulator.get_jnt_values()

    def cvt_gl_pose_to_tcp(self, gl_pos, gl_rotmat):
        return self.manipulator.cvt_gl_pose_to_tcp(gl_pos=gl_pos, gl_rotmat=gl_rotmat)

    def cvt_pose_in_tcp_to_gl(self, loc_pos=np.zeros(3), loc_rotmat=np.eye(3)):
        return self.manipulator.cvt_pose_in_tcp_to_gl(loc_pos=loc_pos, loc_rotmat=loc_rotmat)

    # def get_oih_list(self):
    #     return_list = []
    #     for obj_info in self.oih_infos:
    #         obj_cmodel = obj_info['collision_model']
    #         obj_cmodel.set_pos(obj_info['gl_pos'])
    #         obj_cmodel.set_rotmat(obj_info['gl_rotmat'])
    #         return_list.append(obj_cmodel)
    #     return return_list

    # def release(self, obj_cmodel, jawwidth=None):
    #     """
    #     the obj_cmodel is added as a part of the robot_s to the cd checker
    #     :param jawwidth:
    #     :param obj_cmodel:
    #     :return:
    #     """
    #     if jawwidth is not None:
    #         self.end_effector.change_jaw_width(jawwidth)
    #     for obj_info in self.oih_infos:
    #         if obj_info['collision_model'] is obj_cmodel:
    #             self.cc.delete_cdobj(obj_info)
    #             self.oih_infos.remove(obj_info)
    #             break
