import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.end_effectors.ee_interface as ei
import wrs.grasping.grasp as gg


class SCTInterface(ei.EEInterface):
    """
    single contact tool interface
    examples: screwdriver, suction cup, etc.
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='aabb', name='suction'):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)

    def act_to_by_pose(self, acting_center_pos, acting_center_rotmat, ee_values=None):
        """
        :param acting_center_pos:
        :param acting_center_rotmat:
        :return:
        """
        sct_root_rotmat = acting_center_rotmat.dot(self.loc_acting_center_rotmat.T)
        sct_root_pos = acting_center_pos - sct_root_rotmat.dot(self.loc_acting_center_pos)
        self.fix_to(sct_root_pos, sct_root_rotmat)
        return gg.Grasp(ee_values=ee_values, ac_pos=acting_center_pos, ac_rotmat=acting_center_rotmat)

    def act_to_by_twovecs(self, action_center_pos, approaching_direction, heading_direction, ee_values=None):
        """
        :param action_center_pos:
        :param approaching_direction:
        :param heading_direction: arbitrary if the tool is symmetric
        :param ee_values:
        :return:
        author: weiwei
        date: 20220127
        """
        action_center_rotmat = np.eye(3)
        action_center_rotmat[:3, 2] = rm.unit_vector(approaching_direction)
        action_center_rotmat[:3, 1] = rm.unit_vector(heading_direction)
        action_center_rotmat[:3, 0] = np.cross(action_center_rotmat[:3, 1], action_center_rotmat[:3, 2])
        return self.act_to_by_pose(action_center_pos, action_center_rotmat, ee_values)

    def change_ee_values(self, ee_values):
        pass