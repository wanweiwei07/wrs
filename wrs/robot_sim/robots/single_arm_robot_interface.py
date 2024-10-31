import numpy as np
import wrs.modeling.model_collection as mmc
import wrs.robot_sim.robots.robot_interface as ri


class SglArmRobotInterface(ri.RobotInterface):
    """
    a robot is a combination of a manipulator and an end_type-effector
    author: weiwei
    date: 20230607
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='robot_interface', enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        self._manipulator = None
        self._end_effector = None

    @property
    def end_effector(self):
        return self._end_effector

    @end_effector.setter
    def end_effector(self, end_effector):
        self._end_effector = end_effector

    @property
    def manipulator(self):
        return self._manipulator

    @manipulator.setter
    def manipulator(self, manipulator):
        self._manipulator = manipulator

    @property
    def n_dof(self):
        return self._manipulator.n_dof

    @property
    def jnt_ranges(self):
        return self._manipulator.jnt_ranges

    @property
    def home_conf(self):
        return self._manipulator.home_conf

    @home_conf.setter
    def home_conf(self, conf):
        self._manipulator.home_conf = conf

    @property
    def gl_tcp_pos(self):
        return self._manipulator.gl_tcp_pos

    @property
    def gl_tcp_rotmat(self):
        return self._manipulator.gl_tcp_rotmat

    @property
    def oiee_list(self):
        if self._end_effector is not None:
            return self.end_effector.oiee_list
        else:
            return []

    def update_end_effector(self, ee_values=None):
        if self.end_effector is not None:
            if ee_values is not None:
                self.end_effector.change_ee_values(ee_values=ee_values)
            self.end_effector.fix_to(pos=self._manipulator.gl_flange_pos, rotmat=self._manipulator.gl_flange_rotmat)

    def backup_state(self):
        self._manipulator.backup_state()
        if self._end_effector is not None:
            self._end_effector.backup_state()

    def restore_state(self):
        self._manipulator.restore_state()
        for oiee in self.oiee_list:
            if self.cc is not None:
                self.cc.remove_cce(oiee)
        self.update_end_effector()
        if self._end_effector is not None:
            self._end_effector.restore_state()
        if self.cc is not None:
            uuid_list = []
            for oiee in self.oiee_list:
                uuid_list.append(self.cc.add_cce(oiee))
            if len(uuid_list) > 0:
                self.cc.enable_extcd_by_id_list(id_list=uuid_list, type="from")
                self.cc.enable_innercd_by_id_list(id_list=uuid_list, type="from")
                self.cc.dynamic_ext_list.extend(uuid_list)

    def get_ee_values(self):
        if self._end_effector is not None:
            return self.end_effector.get_ee_values()
        else:
            print("No end effector is attached to the robot!")
            return None

    def change_ee_values(self, ee_values):
        if self._end_effector is not None:
            self.end_effector.change_ee_values(ee_values=ee_values)
        else:
            print("No end effector is attached to the robot!")
            return

    def hold(self, obj_cmodel, **kwargs):
        if self._end_effector is None:
            print("No end effector is attached to the robot!")
            return
        oiee = self.end_effector.hold(obj_cmodel, **kwargs)
        if self.cc is not None:
            uuid = self.cc.add_cce(oiee)
            self.cc.enable_extcd_by_id_list(id_list=[uuid], type="from")
            self.cc.enable_innercd_by_id_list(id_list=[uuid], type="from")
            self.cc.dynamic_ext_list.append(uuid)

    def release(self, obj_cmodel, **kwargs):
        if self._end_effector is None:
            print("No end effector is attached to the robot!")
            return
        oiee = self.end_effector.release(obj_cmodel, **kwargs)
        if oiee is not None and self.cc is not None:
            self.cc.remove_cce(oiee)

    def toggle_off_eecd(self):
        if self.cc is not None:
            self.cc.disable_extcd_by_id_list(id_list=self.cc.dynamic_ext_list, type="from")
            self.cc.disable_innercd_by_id_list(id_list=self.cc.dynamic_into_list, type="into")

    def toggle_on_eecd(self):
        if self.cc is not None:
            self.cc.enable_extcd_by_id_list(id_list=self.cc.dynamic_ext_list, type="from")
            self.cc.enable_innercd_by_id_list(id_list=self.cc.dynamic_into_list, type="into")

    def goto_given_conf(self, jnt_values, ee_values=None):
        result = self._manipulator.goto_given_conf(jnt_values=jnt_values)
        self.update_end_effector(ee_values=ee_values)
        return result

    def goto_home_conf(self):
        self._manipulator.goto_home_conf()
        self.update_end_effector(ee_values=0)

    def ik(self,
           tgt_pos: np.ndarray,
           tgt_rotmat: np.ndarray,
           seed_jnt_values=None,
           option="single",
           toggle_dbg=False):
        return self._manipulator.ik(tgt_pos=tgt_pos,
                                    tgt_rotmat=tgt_rotmat,
                                    seed_jnt_values=seed_jnt_values,
                                    option=option,
                                    toggle_dbg=toggle_dbg)

    def manipulability_val(self):
        return self._manipulator.manipulability_val()

    def manipulability_mat(self):
        return self._manipulator.manipulability_mat()

    def jacobian(self, jnt_values=None):
        return self._manipulator.jacobian(jnt_values=jnt_values)

    def rand_conf(self):
        return self._manipulator.rand_conf()

    def fk(self, jnt_values, toggle_jacobian=False, update=False):
        """
        no update
        :param jnt_values:
        :return:
        author: weiwei
        date: 20210417
        """
        return self._manipulator.fk(jnt_values=jnt_values, toggle_jacobian=toggle_jacobian, update=update)

    def get_jnt_values(self):
        return self._manipulator.get_jnt_values()

    def are_jnts_in_ranges(self, jnt_values):
        return self._manipulator.are_jnts_in_ranges(jnt_values=jnt_values)

    def cvt_gl_pose_to_tcp(self, gl_pos, gl_rotmat):
        return self._manipulator.cvt_gl_pose_to_tcp(gl_pos=gl_pos, gl_rotmat=gl_rotmat)

    def cvt_pose_in_tcp_to_gl(self, loc_pos=np.zeros(3), loc_rotmat=np.eye(3)):
        return self._manipulator.cvt_pose_in_tcp_to_gl(loc_pos=loc_pos, loc_rotmat=loc_rotmat)

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False):
        m_col = mmc.ModelCollection(name=self.name+'_stickmodel')
        if self._manipulator is not None:
            self._manipulator.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                             toggle_jnt_frames=toggle_jnt_frames,
                                             toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        if self.end_effector is not None:
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
                      toggle_cdmesh=False):
        m_col = mmc.ModelCollection(name=self.name+'_meshmodel')
        if self._manipulator is not None:
            self._manipulator.gen_meshmodel(rgb=rgb,
                                            alpha=alpha,
                                            toggle_tcp_frame=toggle_tcp_frame,
                                            toggle_jnt_frames=toggle_jnt_frames,
                                            toggle_flange_frame=toggle_flange_frame,
                                            toggle_cdprim=toggle_cdprim,
                                            toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        if self.end_effector is not None:
            self.end_effector.gen_meshmodel(rgb=rgb,
                                            alpha=alpha,
                                            toggle_tcp_frame=toggle_tcp_frame,
                                            toggle_jnt_frames=toggle_jnt_frames,
                                            toggle_cdprim=toggle_cdprim,
                                            toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        return m_col
