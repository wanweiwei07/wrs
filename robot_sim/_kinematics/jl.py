import warnings
import numpy as np
import basis.constant as bc
import basis.robot_math as rm
import modeling.collision_model as cm
import modeling.geometric_model as gm
import modeling.model_collection as mc
import robot_sim._kinematics.constant as rkc


class Link(object):
    """
    author: weiwei
    date: 20230822
    """

    def __init__(self,
                 name="auto",
                 loc_pos=np.zeros(3),
                 loc_rotmat=np.eye(3),
                 com=np.zeros(3),
                 inertia=np.eye(3),
                 mass=0,
                 collision_model=None,
                 rgba=bc.link_stick_rgba):
        self.name = name
        self.loc_pos = loc_pos
        self.loc_rotmat = loc_rotmat
        self.com = com
        self.inertia = inertia
        self.mass = mass
        self.rgba = rgba
        self.collision_model = collision_model
        # the following values will be updated automatically
        self._gl_pos = self.loc_pos
        self._gl_rotmat = self.loc_rotmat

    @property
    def gl_pos(self):
        return self._gl_pos

    @property
    def gl_rotmat(self):
        return self._gl_rotmat

    def update_globals(self, pos=np.zeros(3), rotmat=np.eye(3)):
        """
        update the global parameters against give reference pos, reference rotmat
        :param pos:
        :param rotmat:
        :return:
        """
        self._gl_pos = pos + rotmat @ self.loc_pos
        self._gl_rotmat = rotmat @ self.loc_rotmat


class Anchor(object):
    """
    author: weiwei
    date: 20230926
    """

    def __init__(self,
                 name="auto",
                 pos=np.zeros(3),
                 rotmat=np.eye(3)):
        self.name = name
        self.pos = pos
        self.rotmat = rotmat

    @property
    def homomat(self):
        return rm.homomat_from_posrot(self.pos, self.rotmat)


class Joint(object):
    """
    author: weiwei
    date: 20230822
    """

    def __init__(self,
                 name="auto",
                 type=rkc.JointType.REVOLUTE,
                 loc_pos=np.zeros(3),
                 loc_rotmat=np.eye(3),
                 loc_motion_ax=np.array([0, 1, 0]),
                 motion_rng=np.array([-np.pi, np.pi])):
        self.name = name
        self.loc_pos = loc_pos
        self.loc_rotmat = loc_rotmat
        self.loc_motion_axis = loc_motion_ax
        self.motion_rng = motion_rng
        # the following parameters will be updated automatically
        self._motion_val = 0
        self._gl_pos_0 = self.loc_pos
        self._gl_rotmat_0 = self.loc_rotmat
        self._gl_motion_ax = self.loc_motion_axis
        self._gl_pos_q = self._gl_pos_0
        self._gl_rotmat_q = self._gl_rotmat_0
        # the following parameter has a setter function
        self._link = None
        # the following parameter should not be changed
        self._type = type

    @property
    def motion_val(self):
        return self._motion_val

    @property
    def loc_homomat(self):
        return rm.homomat_from_posrot(pos=self.loc_pos, rotmat=self.loc_rotmat)

    @property
    def gl_pos_0(self):
        return self._gl_pos_0

    @property
    def gl_rotmat_0(self):
        return self._gl_rotmat_0

    @property
    def gl_homomat_0(self):
        return rm.homomat_from_posrot(pos=self._gl_pos_0, rotmat=self._gl_rotmat_0)

    @property
    def gl_motion_ax(self):
        return self._gl_motion_ax

    @property
    def gl_pos_q(self):
        return self._gl_pos_q

    @property
    def gl_rotmat_q(self):
        return self._gl_rotmat_q

    @property
    def gl_homomat_q(self):
        return rm.homomat_from_posrot(pos=self._gl_pos_q, rotmat=self._gl_rotmat_q)

    @property
    def type(self):
        return self._type

    @property
    def link(self):
        return self._link

    @link.setter
    def link(self, value):
        self._link = value
        self._link.update_globals(self.gl_pos_q, self.gl_rotmat_q)

    def change_type(self, type: rkc.JointType, motion_rng: np.ndarray = None):
        if motion_rng is None:
            if type == rkc.JointType.PRISMATIC:
                motion_rng = np.array([-.1, .1])
            elif type == rkc.JointType.REVOLUTE:
                motion_rng = np.array([-np.pi, np.pi])
        self._type = type
        self.motion_rng = motion_rng

    def assert_motion_val(self, val):
        return
        if val < self.motion_rng[0] or val > self.motion_rng[1]:
            raise ValueError("Motion value is out of range!")

    def set_motion_value(self, motion_value):
        self._motion_val = motion_value
        if self.type == rkc.JointType.REVOLUTE:
            self._gl_pos_q = self._gl_pos_0
            self._gl_rotmat_q = rm.rotmat_from_axangle(self._gl_motion_ax, self._motion_val) @ self._gl_rotmat_0
        elif self.type == rkc.JointType.PRISMATIC:
            self._gl_pos_q = self._gl_pos_0 + self._gl_motion_ax * self._motion_val
            self._gl_rotmat_q = self._gl_rotmat_0

    def update_globals(self, pos=np.zeros(3), rotmat=np.eye(3), motion_val=0):
        """
        update the global parameters against give reference pos, reference rotmat, and motion_value
        :param pos:
        :param rotmat:
        :param motion_val:
        :return:
        """
        self._gl_pos_0 = pos + rotmat @ self.loc_pos
        self._gl_rotmat_0 = rotmat @ self.loc_rotmat
        self._gl_motion_ax = self._gl_rotmat_0 @ self.loc_motion_axis
        self.set_motion_value(motion_value=motion_val)
        if self._link is not None:
            self._link.update_globals(self.gl_pos_q, self.gl_rotmat_q)

    def get_motion_homomat(self, motion_val=0):
        self.assert_motion_val(val=motion_val)
        if self.type == rkc.JointType.REVOLUTE:
            rotmat_by_motion = rm.rotmat_from_axangle(self.loc_motion_axis, motion_val)
            return self.loc_homomat @ rm.homomat_from_posrot(pos=np.zeros(3), rotmat=rotmat_by_motion)
        elif self.type == rkc.JointType.PRISMATIC:
            pos_by_motion = self.loc_motion_axis * motion_val
            return self.loc_homomat @ rm.homomat_from_posrot(pos=pos_by_motion, rotmat=np.eye(3))


def create_link(mesh_file: str,
                name="auto",
                loc_pos=np.zeros(3),
                loc_rotmat=np.eye(3),
                com=np.zeros(3),
                inertia=np.eye(3),
                mass=0,
                rgba=bc.link_stick_rgba):
    objcm = cm.CollisionModel(initor=mesh_file)
    return Link(name=name,
                loc_pos=loc_pos,
                loc_rotmat=loc_rotmat,
                com=com,
                inertia=inertia,
                mass=mass,
                collision_model=objcm,
                rgba=rgba)


def create_joint_with_link(joint_name="auto",
                           joint_type=rkc.JointType.REVOLUTE,
                           link_name="auto"):
    jnt = Joint(joint_name, type=joint_type)
    jnt.link = Link(link_name)
    return jnt


if __name__ == '__main__':
    import visualization.panda.world as wd
    import robot_sim._kinematics.model_generator as rkmg

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    jnt = Joint()
    #
    ref_pos = np.array([0, .1, 0])
    ref_rotmat = rm.rotmat_from_euler(np.pi / 6, np.pi / 3, np.pi / 4)
    # gm.gen_dashed_frame(pos=pos, rotmat=rotmat).attach_to(base)
    #
    jnt.update_globals(pos=ref_pos, rotmat=ref_rotmat, motion_val=np.pi / 2)
    # gm.gen_frame(pos=joint.gl_pos_q, rotmat=joint.gl_rotmat_q).attach_to(base)
    # print(joint.gl_pos_q, joint.gl_rotmat_q)
    #
    # pos = joint.get_transform_homomat(motion_value=np.pi / 2)
    # ref_homomat = rm.homomat_from_posrot(pos=pos, rotmat=rotmat)
    # result_homomat = ref_homomat @ pos
    # print(result_homomat)
    # gm.gen_myc_frame(pos=result_homomat[:3, 3], rotmat=result_homomat[:3, :3]).attach_to(base)

    jnt.link = create_link("../../basis/objects/or2fg7_base.stl")
    rkmg.gen_joint(jnt, toggle_link_mesh=True).attach_to(base)
    base.run()
