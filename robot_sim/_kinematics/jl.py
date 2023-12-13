import warnings
import numpy as np
import basis.constant as bc
import basis.robot_math as rm
import modeling.collision_model as mcm
import modeling.geometric_model as mgm
import modeling.constant as mc
import robot_sim._kinematics.constant as rkc
import uuid


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
                 cmodel=None):
        self.uuid = uuid.uuid4()
        self.name = name
        self._loc_pos = loc_pos
        self._loc_rotmat = loc_rotmat
        self.com = com
        self.inertia = inertia
        self.mass = mass
        self._cmodel = cmodel
        # the following values will be updated automatically
        self._gl_pos = self._loc_pos
        self._gl_rotmat = self._loc_rotmat

    @property
    def loc_pos(self):
        return self._loc_pos

    @loc_pos.setter
    def loc_pos(self, pos):
        self._loc_pos = pos
        self.update_globals(pos=self.gl_pos, rotmat=self.gl_rotmat)

    @property
    def loc_rotmat(self):
        return self._loc_rotmat

    @loc_rotmat.setter
    def loc_rotmat(self, rotmat):
        self._loc_rotmat = rotmat
        self.update_globals(pos=self.gl_pos, rotmat=self.gl_rotmat)

    @property
    def gl_pos(self):
        return self._gl_pos

    @property
    def gl_rotmat(self):
        return self._gl_rotmat

    @property
    def cmodel(self):
        return self._cmodel

    @cmodel.setter
    def cmodel(self, cmodel):
        self._cmodel = cmodel
        self._cmodel.pose = (self._gl_pos, self._gl_rotmat)

    def update_globals(self, pos=np.zeros(3), rotmat=np.eye(3)):
        """
        update the global parameters against give reference pos, reference rotmat
        :param pos:
        :param rotmat:
        :return:
        """
        self._gl_pos = pos + rotmat @ self._loc_pos
        self._gl_rotmat = rotmat @ self._loc_rotmat
        if self._cmodel is not None:
            self._cmodel.pose = (self._gl_pos, self._gl_rotmat)


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
        self._pos = pos
        self._rotmat = rotmat
        self._lnk = Link()

    @property
    def pos(self):
        return self._pos

    @property
    def rotmat(self):
        return self._rotmat

    @property
    def homomat(self):
        return rm.homomat_from_posrot(self.pos, self.rotmat)

    @property
    def lnk(self):
        return self._lnk

    @lnk.setter
    def lnk(self, value):
        self._lnk = value

    def update_pose(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        if self._lnk is not None:
            self._lnk.update_globals(self._pos, self._rotmat)


class Joint(object):
    """
    author: weiwei
    date: 20230822
    """

    def __init__(self,
                 name="auto",
                 type=rkc.JntType.REVOLUTE,
                 loc_pos=np.zeros(3),
                 loc_rotmat=np.eye(3),
                 loc_motion_ax=np.array([0, 1, 0]),
                 motion_rng=np.array([-np.pi, np.pi])):
        self.name = name
        self.loc_pos = loc_pos
        self.loc_rotmat = loc_rotmat
        self.loc_motion_ax = loc_motion_ax
        self.motion_rng = motion_rng
        # the following parameters will be updated automatically
        self._motion_val = 0
        self._gl_pos_0 = self.loc_pos
        self._gl_rotmat_0 = self.loc_rotmat
        self._gl_motion_ax = self.loc_motion_ax
        self._gl_pos_q = self._gl_pos_0
        self._gl_rotmat_q = self._gl_rotmat_0
        # the following parameter has a setter function
        self._lnk = Link()
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
    def lnk(self):
        return self._lnk

    @lnk.setter
    def lnk(self, value):
        self._lnk = value

    def change_type(self, type: rkc.JntType, motion_rng: np.ndarray = None):
        if motion_rng is None:
            if type == rkc.JntType.PRISMATIC:
                motion_rng = np.array([-.1, .1])
            elif type == rkc.JntType.REVOLUTE:
                motion_rng = np.array([-np.pi, np.pi])
        self._type = type
        self.motion_rng = motion_rng

    def assert_motion_val(self, val):
        return
        # if val < self.motion_rng[0] or val > self.motion_rng[1]:
        #     raise ValueError("Motion value is out of range!")

    def set_motion_value(self, motion_value):
        self._motion_val = motion_value
        if self.type == rkc.JntType.REVOLUTE:
            self._gl_pos_q = self._gl_pos_0
            self._gl_rotmat_q = rm.rotmat_from_axangle(self._gl_motion_ax, self._motion_val) @ self._gl_rotmat_0
        elif self.type == rkc.JntType.PRISMATIC:
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
        self._gl_motion_ax = self._gl_rotmat_0 @ self.loc_motion_ax
        self.set_motion_value(motion_value=motion_val)
        if self._lnk is not None:
            self._lnk.update_globals(self.gl_pos_q, self.gl_rotmat_q)

    def get_motion_homomat(self, motion_val=0):
        self.assert_motion_val(val=motion_val)
        if self.type == rkc.JntType.REVOLUTE:
            rotmat_by_motion = rm.rotmat_from_axangle(self.loc_motion_ax, motion_val)
            return self.loc_homomat @ rm.homomat_from_posrot(pos=np.zeros(3), rotmat=rotmat_by_motion)
        elif self.type == rkc.JntType.PRISMATIC:
            pos_by_motion = self.loc_motion_ax * motion_val
            return self.loc_homomat @ rm.homomat_from_posrot(pos=pos_by_motion, rotmat=np.eye(3))


if __name__ == '__main__':
    import visualization.panda.world as wd
    import robot_sim._kinematics.model_generator as rkmg

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    jnt = Joint()
    #
    ref_pos = np.array([0, .1, 0])
    ref_rotmat = rm.rotmat_from_euler(np.pi / 6, np.pi / 3, np.pi / 4)
    # mgm.gen_dashed_frame(pos=pos, rotmat=rotmat).attach_to(base)
    #
    jnt.update_globals(pos=ref_pos, rotmat=ref_rotmat, motion_val=np.pi / 2)
    # mgm.gen_frame(pos=joint.gl_pos_q, rotmat=joint.gl_rotmat_q).attach_to(base)
    # print(joint.gl_pos_q, joint.gl_rotmat_q)
    #
    # pos = joint.get_transform_homomat(motion_value=np.pi / 2)
    # ref_homomat = rm.homomat_from_posrot(pos=pos, rotmat=rotmat)
    # result_homomat = ref_homomat @ pos
    # print(result_homomat)
    # mgm.gen_myc_frame(pos=result_homomat[:3, 3], rotmat=result_homomat[:3, :3]).attach_to(base)

    jnt.lnk.cmodel = mcm.CollisionModel("../../basis/objects/or2fg7_base.stl")
    rkmg.gen_jnt(jnt, tgl_lnk_mesh=True).attach_to(base)
    jnt.lnk.cmodel.show_cdprimitive()
    base.run()
