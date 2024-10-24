import numpy as np
import wrs.robot_sim.end_effectors.ee_interface as ei
import wrs.modeling.constant as mc
import wrs.grasping.grasp as gg


class GripperInterface(ei.EEInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type=mc.CDMeshType.AABB, name='grippers'):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        # jaw width
        self.jaw_range = np.array([0.0, 0.05])  # 0~0.05m by default
        self.jaw_width = self.jaw_range[0]
        # fgr0 opening vec
        self.loc_fgr0_opening_vec = np.array([0, 1, 0])  # y as opening vec by default
        # backup
        self.jaw_width_bk = []

    def backup_state(self):
        self.oiee_list_bk.append(self.oiee_list.copy())
        self.oiee_pose_list_bk.append([oiee.loc_pose for oiee in self.oiee_list])
        self.jaw_width_bk.append(self.get_jaw_width())

    def restore_state(self):
        self.oiee_list = self.oiee_list_bk.pop()
        oiee_pose_list = self.oiee_pose_list_bk.pop()
        for i, oiee in enumerate(self.oiee_list):
            oiee.loc_pose = oiee_pose_list[i]
        self.update_oiee()
        jaw_width = self.jaw_width_bk.pop()
        if jaw_width != self.get_jaw_width():
            self.change_jaw_width(jaw_width=jaw_width)

    def get_jaw_width(self):
        raise NotImplementedError

    @ei.EEInterface.assert_oiee_decorator
    def change_jaw_width(self, jaw_width):
        raise NotImplementedError

    def get_ee_values(self):
        return self.get_jaw_width()

    @ei.EEInterface.assert_oiee_decorator
    def change_ee_values(self, ee_values):
        """
        interface
        :param ee_values:
        :return:
        """
        self.change_jaw_width(jaw_width=ee_values)

    def hold(self, obj_cmodel, jaw_width=None):
        if jaw_width is None:
            jaw_width = self.jaw_range[0]
        self.change_jaw_width(jaw_width=jaw_width)
        return super().hold(obj_cmodel=obj_cmodel)

    def release(self, obj_cmodel, jaw_width=None):
        obj_lnk = super().release(obj_cmodel=obj_cmodel)
        if jaw_width is None:
            jaw_width = self.jaw_range[1]
        self.change_jaw_width(jaw_width=jaw_width)
        return obj_lnk

    def grip_at_by_twovecs(self,
                           jaw_center_pos,
                           approaching_direction,
                           thumb_opening_direction,
                           jaw_width):
        """
        specifying the gripping pose using two axes -- approaching vector and opening vector
        the thumb is the finger at the y+ direction
        :param jaw_center_pos:
        :param approaching_direction: jaw_center's approaching motion_vec
        :param thumb_opening_direction: jaw_center's opening motion_vec (thumb is the left finger, or finger 0)
        :param jaw_width: [ee_values, jaw_center_pos, jaw_center_rotmat, eef_root_pos, eef_root_rotmat]
        :return:
        """
        self.change_jaw_width(jaw_width)
        param_list = self.align_acting_center_by_twovecs(acting_center_pos=jaw_center_pos,
                                                         approaching_vec=approaching_direction,
                                                         side_vec=thumb_opening_direction)
        return gg.Grasp(ee_values=jaw_width, ac_pos=param_list[0], ac_rotmat=param_list[1])

    def grip_at_by_pose(self, jaw_center_pos, jaw_center_rotmat, jaw_width):
        """
        :param jaw_center_pos:
        :param jaw_center_rotmat:
        :param jaw_width:
        :return: [ee_values, jaw_center_pos, gl_jaw_center_rotmat, eef_root_pos, eef_root_rotmat]
        """
        self.change_jaw_width(jaw_width)
        param_list = self.align_acting_center_by_pose(acting_center_pos=jaw_center_pos,
                                                      acting_center_rotmat=jaw_center_rotmat)
        return gg.Grasp(ee_values=jaw_width, ac_pos=param_list[0], ac_rotmat=param_list[1])