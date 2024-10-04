import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.model_generator as rkmg
import wrs.robot_sim._kinematics.constant as const


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
        # grafting target
        self._root_pos = np.zeros(3)
        self._root_rotmat = np.eye(3)
        # delay
        self._is_gl_pose_delayed = False

    @staticmethod
    def delay_gl_pose_decorator(method):
        def wrapper(self, *args, **kwargs):
            self._is_gl_pose_delayed = True
            return method(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def update_gl_pose_decorator(method):
        def wrapper(self, *args, **kwargs):
            if self._is_gl_pose_delayed:
                self._gl_pos = self._root_pos + self._root_rotmat @ self._loc_pos
                self._gl_rotmat = self._root_rotmat @ self._loc_rotmat
                self._is_gl_pose_delayed = False
            return method(self, *args, **kwargs)

        return wrapper

    @property
    def uuid(self):
        if self.cmodel is not None:
            return self.cmodel.uuid
        else:
            raise ValueError("uuid will not be available until cmodel is assigned.")

    @property
    def root_pose(self):
        return (self._root_pos, self._root_rotmat)

    @root_pose.setter
    @delay_gl_pose_decorator
    def root_pose(self, pose):
        self._root_pos = pose[0]
        self._root_rotmat = pose[1]

    @property
    def loc_pos(self):
        return self._loc_pos

    @loc_pos.setter
    @delay_gl_pose_decorator
    def loc_pos(self, pos):
        self._loc_pos = pos

    @property
    def loc_rotmat(self):
        return self._loc_rotmat

    @loc_rotmat.setter
    @delay_gl_pose_decorator
    def loc_rotmat(self, rotmat):
        self._loc_rotmat = rotmat

    @property
    def loc_homomat(self):
        return rm.homomat_from_posrot(self._loc_pos, self._loc_rotmat)

    @property
    def loc_pose(self):
        return (self._loc_pos, self._loc_rotmat)

    @loc_pose.setter
    @delay_gl_pose_decorator
    def loc_pose(self, pose):
        self._loc_pos = pose[0]
        self._loc_rotmat = pose[1]

    @property
    @update_gl_pose_decorator
    def gl_pos(self):
        return self._gl_pos

    @property
    @update_gl_pose_decorator
    def gl_rotmat(self):
        return self._gl_rotmat

    @property
    def gl_homomat(self):
        return rm.homomat_from_posrot(self._gl_pos, self._gl_rotmat)

    @property
    @update_gl_pose_decorator
    def cmodel(self):
        if self._cmodel is not None:
            self._cmodel.pose = (self._gl_pos, self._gl_rotmat)
        return self._cmodel

    @cmodel.setter
    def cmodel(self, cmodel):
        self._cmodel = cmodel
        self._cmodel.pose = (self._gl_pos, self._gl_rotmat)

    def install_onto(self, pos=np.zeros(3), rotmat=np.eye(3)):
        """
        update the global parameters with given reference pos, reference rotmat
        :param pos:
        :param rotmat:
        :return:
        """
        self._root_pos = pos
        self._root_rotmat = rotmat
        self._gl_pos = self._root_pos + self._root_rotmat @ self._loc_pos
        self._gl_rotmat = self._root_rotmat @ self._loc_rotmat
        if self._cmodel is not None:
            self._cmodel.pose = (self._gl_pos, self._gl_rotmat)

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_cdprim=False,
                      toggle_cdmesh=False):
        return rkmg.gen_lnk_mesh(lnk=self,
                                 rgb=rgb,
                                 alpha=alpha,
                                 toggle_cdprim=toggle_cdprim,
                                 toggle_cdmesh=toggle_cdmesh)

    def copy(self):
        lnk_copy = Link(name=self.name,
                        loc_pos=self.loc_pos,
                        loc_rotmat=self.loc_rotmat,
                        com=self.com,
                        inertia=self.inertia,
                        mass=self.mass,
                        cmodel=self.cmodel.copy())
        lnk_copy.install_onto(pos=self.root_pose[0], rotmat=self.root_pose[1])
        return lnk_copy


class Anchor(object):
    """
    author: weiwei
    date: 20230926
    """

    def __init__(self,
                 name="auto",
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 n_flange=1,
                 n_lnk=1):
        """
        :param name:
        :param pos: pos for parent
        :param rotmat: rotmat for parent
        :param loc_flange_pos: pos for mounting (local in the pos/rotmat frame)
        :param loc_flange_rotmat: rotmat for mounting (local in the pos/rotmat frame)
        """
        self.name = name
        self._pos = pos
        self._rotmat = rotmat
        self._n_flange = n_flange
        self._n_lnk = n_lnk
        self._loc_flange_pose_list = [[np.zeros(3), np.eye(3)] for _ in range(self._n_flange)]
        self._gl_flange_pose_list = self.compute_gl_flange()
        self._lnk_list = [Link(name=name) for _ in range(self._n_lnk)]
        self._is_gl_flange_delayed = True
        self._is_lnk_delayed = True

    # decorator for delayed update of gl_flanges and lnk
    @staticmethod
    def delay_gl_flange_decorator(method):
        def wrapper(self, *args, **kwargs):
            self._is_gl_flange_delayed = True
            return method(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def update_gl_flange_decorator(method):
        def wrapper(self, *args, **kwargs):
            if self._is_gl_flange_delayed:
                self._gl_flange_pose_list = self.compute_gl_flange()
                self._is_gl_flange_delayed = False
            return method(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def delay_gl_lnk_decorator(method):
        def wrapper(self, *args, **kwargs):
            self._is_lnk_delayed = True
            return method(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def update_gl_lnk_decorator(method):
        def wrapper(self, *args, **kwargs):
            if self._is_lnk_delayed:
                for lnk in self._lnk_list:
                    lnk.install_onto(self.pos, self.rotmat)
                self._is_lnk_delayed = False
            return method(self, *args, **kwargs)

        return wrapper

    @property
    def n_flange(self):
        return self._n_flange

    @property
    def n_lnk(self):
        return self._n_lnk

    @property
    def pos(self):
        return self._pos

    @pos.setter
    @delay_gl_flange_decorator
    @delay_gl_lnk_decorator
    def pos(self, pos):
        self._pos = pos

    @property
    def rotmat(self):
        return self._rotmat

    @rotmat.setter
    @delay_gl_flange_decorator
    @delay_gl_lnk_decorator
    def rotmat(self, rotmat):
        self._rotmat = rotmat

    @property
    def homomat(self):
        return rm.homomat_from_posrot(self.pos, self.rotmat)

    @property
    @update_gl_flange_decorator
    @update_gl_lnk_decorator
    def gl_flange_pose_list(self):
        return self._gl_flange_pose_list

    @property
    @update_gl_flange_decorator
    @update_gl_lnk_decorator
    def gl_flange_pose(self):
        """
        return the first flange pose if gl_flange_pose is accesssed
        :return:
        author: weiwei
        date: 20240422
        """
        return self._gl_flange_pose_list[0]

    @property
    @update_gl_flange_decorator
    @update_gl_lnk_decorator
    def gl_flange_homomat_list(self):
        return [rm.homomat_from_posrot(gl_flange_pose[0], gl_flange_pose[1]) for gl_flange_pose in
                self._gl_flange_pose_list]

    @property
    @delay_gl_flange_decorator  # for reassigning individual ones
    def loc_flange_pose_list(self):
        return self._loc_flange_pose_list

    @loc_flange_pose_list.setter
    @delay_gl_flange_decorator
    def loc_flange_pose_list(self, list):
        self._loc_flange_pose_list = list
        self._n_flange = len(self._loc_flange_pose_list)

    @property
    @update_gl_lnk_decorator
    def lnk_list(self):
        return self._lnk_list

    @lnk_list.setter
    @delay_gl_lnk_decorator
    def lnk_list(self, list):
        self._lnk_list = list
        self._n_lnk = len(self._lnk_list)

    @delay_gl_flange_decorator
    @delay_gl_lnk_decorator
    def fix_to(self, pos, rotmat):
        """
        fix the anchor to a given pos and rotmat
        :param pos:
        :param rotmat:
        :return:
        author: weiwei
        date: 20240421, for uniform interface
        """
        self._pos = pos
        self._rotmat = rotmat

    def compute_gl_flange(self):
        gl_flange_list = []
        for loc_flange_pos, loc_flange_rotmat in self._loc_flange_pose_list:
            gl_flange_list.append([self.pos + self.rotmat @ loc_flange_pos, self.rotmat @ loc_flange_rotmat])
        return gl_flange_list

    def gen_stickmodel(self,
                       toggle_root_frame=True,
                       toggle_flange_frame=True,
                       radius=const.JNT_RADIUS,
                       frame_stick_radius=const.FRAME_STICK_RADIUS,
                       frame_stick_length=const.FRAME_STICK_LENGTH_MEDIUM):
        return rkmg.gen_anchor(anchor=self,
                               toggle_root_frame=toggle_root_frame,
                               toggle_flange_frame=toggle_flange_frame,
                               toggle_lnk_mesh=False,
                               radius=radius,
                               frame_stick_radius=frame_stick_radius,
                               frame_stick_length=frame_stick_length)

    def gen_meshmodel(self, name="anchor_lnk_mesh", rgb=None,
                      alpha=None, toggle_cdmesh=False, toggle_cdprim=False,
                      toggle_root_frame=False, toggle_flange_frame=False,
                      frame_stick_radius=const.FRAME_STICK_RADIUS,
                      frame_stick_length=const.FRAME_STICK_LENGTH_MEDIUM):
        m_col = rkmg.mmc.ModelCollection(name=name)
        for i in range(self._n_lnk):
            rkmg.gen_lnk_mesh(self.lnk_list[i], rgb=rgb, alpha=alpha, toggle_cdmesh=toggle_cdmesh,
                              toggle_cdprim=toggle_cdprim).attach_to(m_col)
        if toggle_root_frame:
            mgm.gen_frame(pos=self.pos, rotmat=self.rotmat, ax_radius=frame_stick_radius,
                          ax_length=frame_stick_length, alpha=alpha).attach_to(m_col)
        if toggle_flange_frame:
            for gl_flange_pos, gl_flange_rotmat in self.gl_flange_pose_list:
                rkmg.gen_indicated_frame(spos=self.pos, gl_pos=gl_flange_pos, gl_rotmat=gl_flange_rotmat,
                                         indicator_rgba=rm.const.cyan, frame_alpha=alpha).attach_to(m_col)
        return m_col


class Joint(object):
    """
    TODO: loc pos / rotmat identical to zero pos / rotmat? 20240309
    author: weiwei
    date: 20230822
    """

    def __init__(self,
                 name="auto",
                 type=const.JntType.REVOLUTE,
                 loc_pos=np.zeros(3),
                 loc_rotmat=np.eye(3),
                 loc_motion_ax=np.array([0, 1, 0]),
                 motion_range=np.array([-np.pi, np.pi])):
        self.name = name
        self.loc_pos = loc_pos
        self.loc_rotmat = loc_rotmat
        self.loc_motion_ax = loc_motion_ax
        self.motion_range = motion_range
        # the following parameters will be updated automatically
        self._motion_value = 0
        self._gl_pos_0 = self.loc_pos
        self._gl_rotmat_0 = self.loc_rotmat
        self._gl_motion_ax = self.loc_motion_ax
        self._gl_pos_q = self._gl_pos_0
        self._gl_rotmat_q = self._gl_rotmat_0
        # the following parameter has a setter function
        self._lnk = Link(name=name)
        # the following parameter should not be changed
        self._type = type

    @property
    def motion_value(self):
        return self._motion_value

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

    def change_type(self, type: const.JntType, motion_range: np.ndarray = None):
        if motion_range is None:
            if type == const.JntType.PRISMATIC:
                motion_range = np.array([-.1, .1])
            elif type == const.JntType.REVOLUTE:
                motion_range = np.array([-np.pi, np.pi])
        self._type = type
        self.motion_range = motion_range

    def assert_motion_value(self, value):
        return
        # if value < self.motion_range[0] or value > self.motion_range[1]:
        #     raise ValueError("Motion value is out of range!")

    def set_motion_value(self, motion_value):
        self._motion_value = motion_value
        if self.type == const.JntType.REVOLUTE:
            self._gl_pos_q = self._gl_pos_0
            self._gl_rotmat_q = rm.rotmat_from_axangle(self._gl_motion_ax, self._motion_value) @ self._gl_rotmat_0
        elif self.type == const.JntType.PRISMATIC:
            self._gl_pos_q = self._gl_pos_0 + self._gl_motion_ax * self._motion_value
            self._gl_rotmat_q = self._gl_rotmat_0

    def update_globals(self, pos=np.zeros(3), rotmat=np.eye(3), motion_value=0):
        """
        update the global parameters against give reference pos, reference rotmat, and motion_value
        :param pos:
        :param rotmat:
        :param motion_value:
        :return:
        """
        self._gl_pos_0 = pos + rotmat @ self.loc_pos  # TODO offset to loc
        self._gl_rotmat_0 = rotmat @ self.loc_rotmat
        self._gl_motion_ax = self._gl_rotmat_0 @ self.loc_motion_ax
        self.set_motion_value(motion_value=motion_value)
        if self._lnk is not None:
            self._lnk.install_onto(self.gl_pos_q, self.gl_rotmat_q)

    def get_motion_homomat(self, motion_value=0):
        """
        result of loc_homomat multiplied by motion transformation
        :param motion_value:
        :return:
        """
        self.assert_motion_value(value=motion_value)
        if self.type == const.JntType.REVOLUTE:
            rotmat_by_motion = rm.rotmat_from_axangle(self.loc_motion_ax, motion_value)
            return self.loc_homomat @ rm.homomat_from_posrot(pos=np.zeros(3), rotmat=rotmat_by_motion)
        elif self.type == const.JntType.PRISMATIC:
            pos_by_motion = self.loc_motion_ax * motion_value
            return self.loc_homomat @ rm.homomat_from_posrot(pos=pos_by_motion, rotmat=np.eye(3))

    def gen_model(self,
                  toggle_frame_0=True,
                  toggle_frame_q=True,
                  toggle_lnk_mesh=False,
                  radius=const.JNT_RADIUS,
                  frame_stick_radius=const.FRAME_STICK_RADIUS,
                  frame_stick_length=const.FRAME_STICK_LENGTH_MEDIUM):
        return rkmg.gen_jnt(jnt=self,
                            toggle_frame_0=toggle_frame_0,
                            toggle_frame_q=toggle_frame_q,
                            toggle_lnk_mesh=toggle_lnk_mesh,
                            radius=radius,
                            frame_stick_radius=frame_stick_radius,
                            frame_stick_length=frame_stick_length)


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    jnt = Joint(loc_motion_ax=np.array([0, 0, 1]))
    #
    ref_pos = np.array([0, .1, 0])
    ref_rotmat = rm.rotmat_from_euler(np.pi / 6, np.pi / 3, np.pi / 4)
    # mgm.gen_dashed_frame(pos=pos, rotmat=rotmat).attach_to(base)
    #
    jnt.update_globals(pos=ref_pos, rotmat=ref_rotmat, motion_value=np.pi / 2)
    # mgm.gen_frame(pos=joint.gl_pos_q, rotmat=joint.gl_rotmat_q).attach_to(base)
    # print(joint.gl_pos_q, joint.gl_rotmat_q)
    #
    # pos = joint.get_transform_homomat(motion_value=np.pi / 2)
    # ref_homomat = rm.homomat_from_posrot(pos=pos, rotmat=rotmat)
    # result_homomat = ref_homomat @ pos
    # print(result_homomat)
    # mgm.gen_myc_frame(pos=result_homomat[:3, 3], rotmat=result_homomat[:3, :3]).attach_to(base)

    jnt.lnk.cmodel = mcm.CollisionModel(initor="../../basis/objects/or2fg7_base.stl")
    jnt.gen_model(toggle_lnk_mesh=True).attach_to(base)
    base.run()
