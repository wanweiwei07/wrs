import numpy as np
import basis.robot_math as rm
import robot_sim.end_effectors.ee_interface as eei


class SCTInterface(eei.EEInterface):
    """
    single contact tool interface
    examples: screwdriver, suction cup, etc.
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='aabb', name='suction'):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)

    def act_to_with_cpose(self, gl_action_center_pos, gl_action_center_rotmat):
        """
        :param gl_action_center_posm:
        :param gl_action_center_rotmat: jaw_center's rotmat
        :param jaw_width:
        :return:
        """
        sct_root_rotmat = gl_action_center_rotmat.dot(self.action_center_rotmat.T)
        sct_root_pos = gl_action_center_pos - sct_root_rotmat.dot(self.action_center_pos)
        self.fix_to(sct_root_pos, sct_root_rotmat)
        return [gl_action_center_pos, gl_action_center_rotmat, sct_root_pos, sct_root_rotmat]

    def act_to_with_czy(self, gl_action_center_pos, gl_action_center_z, gl_action_center_y):
        """
        :param gl_action_center_pos:
        :param gl_action_center_z: jaw_center's approaching direction
        :param gl_action_center_y: jaw_center's opening direction
        :param jaw_width:
        :return:
        author: weiwei
        date: 20220127
        """
        gl_action_center_rotmat = np.eye(3)
        gl_action_center_rotmat[:, 2] = rm.unit_vector(gl_action_center_z)
        gl_action_center_rotmat[:, 1] = rm.unit_vector(gl_action_center_y)
        gl_action_center_rotmat[:, 0] = np.cross(gl_action_center_rotmat[:3, 1], gl_action_center_rotmat[:3, 2])
        return self.act_to_with_cpose(gl_action_center_pos, gl_action_center_rotmat)