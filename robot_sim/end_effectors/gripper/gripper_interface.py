import numpy as np
import modeling.constant as mc
import robot_sim.end_effectors.ee_interface as ei


class GripperInterface(ei.EEInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type=mc.CDMType.AABB, name='gripper'):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        # jaw width
        self.jaw_range = np.array([0.0, 0.05])  # 0~0.05m by default
        # fgr0 opening vec
        self.loc_fgr0_opening_vec = np.array([0, 1, 0])  # y as opening vec by default

    @ei.EEInterface.assert_oiee_decorator
    def change_jaw_width(self, jaw_width):
        pass

    def get_jaw_width(self):
        raise NotImplementedError

    def hold(self, obj_cmodel, jaw_width=None):
        super().hold(obj_cmodel=obj_cmodel)
        if jaw_width is None:
            jaw_width = self.jaw_range[0]
        self.change_jaw_width(jaw_width=jaw_width)

    def release(self, obj_cmodel, jaw_width=None):
        super().release(obj_cmodel=obj_cmodel)
        if jaw_width is None:
            jaw_width = self.jaw_range[1]
        self.change_jaw_width(jaw_width=jaw_width)

    def grip_at_by_twovecs(self,
                           jaw_center_pos,
                           approaching_vec,
                           fgr0_opening_vec,
                           jaw_width):
        """
        specifying the gripping pose using two axes -- approaching vector and opening vector
        :param jaw_center_pos:
        :param approaching_vec: jaw_center's approaching motion_vec
        :param fgr0_opening_vec: jaw_center's opening motion_vec
        :param jaw_width: [jaw_width, jaw_center_pos, jaw_center_rotmat, eef_root_pos, eef_root_rotmat]
        :return:
        """
        self.change_jaw_width(jaw_width)
        param_list = self.align_acting_center_by_twovecs(acting_center_pos=jaw_center_pos,
                                                         approaching_vec=approaching_vec,
                                                         side_vec=fgr0_opening_vec)
        return [jaw_width] + param_list

    def grip_at_by_pose(self, jaw_center_pos, jaw_center_rotmat, jaw_width):
        """
        :param jaw_center_pos:
        :param jaw_center_rotmat:
        :param jaw_width:
        :return: [jaw_width, jaw_center_pos, gl_jaw_center_rotmat, eef_root_pos, eef_root_rotmat]
        """
        self.change_jaw_width(jaw_width)
        param_list = self.align_acting_center_by_pose(acting_center_pos=jaw_center_pos,
                                                      acting_center_rotmat=jaw_center_rotmat)
        return [jaw_width] + param_list
