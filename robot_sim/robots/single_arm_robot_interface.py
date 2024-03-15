import numpy as np
import robot_sim._kinematics.collision_checker as cc
import robot_sim.robots.robot_interface as ri
import modeling.model_collection as mmc


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
        self.jnt_values_bk = []

    @property
    def home_conf(self):
        return self.manipulator.home_conf

    @home_conf.setter
    def home_conf(self, conf):
        self.manipulator.home_conf = conf

    @property
    def gl_tcp_pos(self):
        return self.manipulator.gl_tcp_pos

    @property
    def gl_tcp_rotmat(self):
        return self.manipulator.gl_tcp_rotmat

    def _update_end_effector(self):
        self.end_effector.fix_to(pos=self.manipulator.gl_flange_pos, rotmat=self.manipulator.gl_flange_rotmat)

    def backup_state(self):
        self.jnt_values_bk.append(self.manipulator.get_jnt_values())
        self.end_effector.backup_state()

    def restore_state(self):
        self.manipulator.goto_given_conf(jnt_values=self.jnt_values_bk.pop())
        self.end_effector.restore_state()

    def hold(self, obj_cmodel, **kwargs):
        self.end_effector.hold(obj_cmodel, **kwargs)

    def release(self, obj_cmodel, **kwargs):
        self.end_effector.release(obj_cmodel, **kwargs)

    def goto_given_conf(self, jnt_values):
        result = self.manipulator.goto_given_conf(jnt_values=jnt_values)
        self._update_end_effector()
        return result

    def goto_home_conf(self):
        self.manipulator.goto_home_conf()
        self._update_end_effector()

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

    def fk(self, jnt_values, toggle_jacobian=False):
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

    def are_jnts_in_ranges(self, jnt_values):
        return self.manipulator.are_jnts_in_ranges(jnt_values=jnt_values)

    def cvt_gl_pose_to_tcp(self, gl_pos, gl_rotmat):
        return self.manipulator.cvt_gl_pose_to_tcp(gl_pos=gl_pos, gl_rotmat=gl_rotmat)

    def cvt_pose_in_tcp_to_gl(self, loc_pos=np.zeros(3), loc_rotmat=np.eye(3)):
        return self.manipulator.cvt_pose_in_tcp_to_gl(loc_pos=loc_pos, loc_rotmat=loc_rotmat)

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name='single_arm_robot_interface_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.manipulator.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                        toggle_jnt_frames=toggle_jnt_frames,
                                        toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        self.end_effector.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                         toggle_jnt_frames=toggle_jnt_frames).attach_to(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='single_arm_robot_interface_meshmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.manipulator.gen_meshmodel(rgb=rgb,
                                       alpha=alpha,
                                       toggle_tcp_frame=toggle_tcp_frame,
                                       toggle_jnt_frames=toggle_jnt_frames,
                                       toggle_flange_frame=toggle_flange_frame,
                                       toggle_cdprim=toggle_cdprim,
                                       toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        self.end_effector.gen_meshmodel(rgb=rgb,
                                        alpha=alpha,
                                        toggle_tcp_frame=toggle_tcp_frame,
                                        toggle_jnt_frames=toggle_jnt_frames,
                                        toggle_cdprim=toggle_cdprim,
                                        toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        return m_col

    # def get_oih_list(self):
    #     return_list = []
    #         obj_cmodel = obj_info['collision_model']
    #         obj_cmodel.set_pos(obj_info['gl_pos'])
    #         obj_cmodel.set_rotmat(obj_info['gl_rotmat'])
    #         return_list.append(obj_cmodel)
    #     return return_list

    # def release(self, obj_cmodel, jaw_width=None):
    #     """
    #     the obj_cmodel is added as a part of the robot_s to the cd checker
    #     :param jaw_width:
    #     :param obj_cmodel:
    #     :return:
    #     """
    #     if jaw_width is not None:
    #         self.end_effector.change_jaw_width(jaw_width)
    #     for obj_info in self.oih_infos:
    #         if obj_info['collision_model'] is obj_cmodel:
    #             self.cc.delete_cdobj(obj_info)
    #             self.oih_infos.remove(obj_info)
    #             break
