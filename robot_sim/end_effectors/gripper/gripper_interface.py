import numpy as np
import modeling.model_collection as mmc
import modeling.constant as mc
import basis.robot_math as rm
import robot_sim.end_effectors.ee_interface as ei


class GripperInterface(ei.EEInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type=mc.CDMType.AABB, name='gripper'):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        # jaw width
        self.jaw_rng = [0.0, 0.05]  # 0~0.05m by default
        # collision detection
        self.cc = None
        # cd mesh collection for precise collision checking
        self.cdmesh_collection = mmc.ModelCollection()

    # jaw center pos and rotmat are defined for back compatibility reasons. to be replaced by action_xxx; 20230807
    @property
    def action_center_pos(self):
        return self.action_center_pos

    @action_center_pos.setter
    def action_center_pos(self, pos: np.array):
        self.action_center_pos = pos

    @property
    def action_center_rotmat(self):
        return self.action_center_rotmat

    @action_center_rotmat.setter
    def action_center_rotmat(self, rotmat: np.array):
        self.action_center_rotmat = rotmat

    def fk(self, motion_val):
        raise NotImplementedError

    def change_jaw_width(self, jaw_width):
        raise NotImplementedError

    def get_jaw_width(self):
        raise NotImplementedError

    def grip_at_with_acao(self, gl_action_center_pos, gl_approaching_direction, gl_opening_direction, jaw_width):
        """
        specifying the gripping pose using ACAO, where AC=Acution Center, AO=Approaching and Opening directions
        :param gl_action_center_pos:
        :param gl_approaching_direction: jaw_center's approaching direction
        :param gl_opening_direction: jaw_center's opening direction
        :param jaw_width:
        :return:
        """
        gl_action_center_rotmat = np.eye(3)
        gl_action_center_rotmat[:, 2] = rm.unit_vector(gl_approaching_direction)
        gl_action_center_rotmat[:, 1] = rm.unit_vector(gl_opening_direction)
        gl_action_center_rotmat[:, 0] = np.cross(gl_action_center_rotmat[:3, 1], gl_action_center_rotmat[:3, 2])
        return self.grip_at_with_jcpose(gl_action_center_pos, gl_action_center_rotmat, jaw_width)

    def grip_at_with_jcpose(self, gl_jaw_center_pos, gl_jaw_center_rotmat, jaw_width):
        """
        :param gl_jaw_center_pos:
        :param gl_jaw_center_rotmat: jaw_center's rotmat
        :param jaw_width:
        :return:
        """
        self.change_jaw_width(jaw_width)
        eef_root_rotmat = gl_jaw_center_rotmat.dot(self.action_center_rotmat.T)
        eef_root_pos = gl_jaw_center_pos - eef_root_rotmat.dot(self.action_center_pos)
        self.fix_to(eef_root_pos, eef_root_rotmat)
        return [jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, eef_root_pos, eef_root_rotmat]
