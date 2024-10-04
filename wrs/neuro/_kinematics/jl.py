import torch
from wrs import basis as bc, robot_sim as rkc, robot_sim as rkmg, modeling as mcm, modeling as mgm
import wrs.neuro._kinematics.math_utils as nkm


class Link(object):
    """
    author: weiwei
    date: 20230822
    """

    def __init__(self,
                 name="auto",
                 loc_pos=torch.zeros(3),
                 loc_rotmat=torch.eye(3),
                 com=torch.zeros(3),
                 inertia=torch.eye(3),
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
        self._root_pos = torch.zeros(3)
        self._root_rotmat = torch.eye(3)
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

    @loc_rotmat.setter
    @delay_gl_pose_decorator
    def loc_rotmat(self, rotmat):
        self._loc_rotmat = rotmat

    @property
    def loc_homomat(self):
        return nkm.homomat_from_posrot(self._loc_pos, self._loc_rotmat)

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
        return nkm.homomat_from_posrot(self._gl_pos, self._gl_rotmat)

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

    def install_onto(self, pos=torch.zeros(3), rotmat=torch.eye(3)):
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
                 pos=torch.zeros(3),
                 rotmat=torch.eye(3),
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
        self._loc_flange_pose_list = [[torch.zeros(3), torch.eye(3)] for _ in range(self._n_flange)]
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
        return nkm.homomat_from_posrot(self.pos, self.rotmat)

    @property
    @update_gl_flange_decorator
    @update_gl_lnk_decorator
    def gl_flange_pose_list(self):
        return self._gl_flange_pose_list

    @property
    @update_gl_flange_decorator
    @update_gl_lnk_decorator
    def gl_flange_homomat_list(self):
        return [nkm.homomat_from_posrot(gl_flange_pose[0], gl_flange_pose[1]) for gl_flange_pose in
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

    def compute_gl_flange(self):
        gl_flange_list = []
        for loc_flange_pos, loc_flange_rotmat in self._loc_flange_pose_list:
            gl_flange_list.append([self.pos + self.rotmat @ loc_flange_pos, self.rotmat @ loc_flange_rotmat])
        return gl_flange_list

    def gen_stickmodel(self,
                       toggle_root_frame=True,
                       toggle_flange_frame=True,
                       radius=rkc.JNT_RADIUS,
                       frame_stick_radius=rkc.FRAME_STICK_RADIUS,
                       frame_stick_length=rkc.FRAME_STICK_LENGTH_MEDIUM):
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
                      frame_stick_radius=rkc.FRAME_STICK_RADIUS,
                      frame_stick_length=rkc.FRAME_STICK_LENGTH_MEDIUM):
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
                                         indicator_rgba=bc.cyan, frame_alpha=alpha).attach_to(m_col)
        return m_col


class Joint(object):
    """
    TODO: loc pos / rotmat identical to zero pos / rotmat? 20240309
    author: weiwei
    date: 20230822
    """

    def __init__(self,
                 name="auto",
                 type=rkc.JntType.REVOLUTE,
                 loc_pos=torch.zeros(3),
                 loc_rotmat=torch.eye(3),
                 loc_motion_ax=torch.tensor([0, 1, 0]),
                 motion_range=torch.tensor([-torch.pi, torch.pi])):
        self.name = name
        self.loc_pos = loc_pos
        self.loc_rotmat = loc_rotmat
        self.loc_motion_ax = loc_motion_ax
        self.motion_range = motion_range
        # the following parameters will be updated automatically
        self._motion_value = torch.tensor(.0, requires_grad=True)
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
        return nkm.homomat_from_posrot(pos=self.loc_pos, rotmat=self.loc_rotmat)

    @property
    def gl_pos_0(self):
        return self._gl_pos_0

    @property
    def gl_rotmat_0(self):
        return self._gl_rotmat_0

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
        return nkm.homomat_from_posrot(pos=self._gl_pos_q, rotmat=self._gl_rotmat_q)

    @property
    def type(self):
        return self._type

    @property
    def lnk(self):
        return self._lnk

    @lnk.setter
    def lnk(self, value):
        self._lnk = value

    def change_type(self, type, motion_range):
        if motion_range is None:
            if type == rkc.JntType.PRISMATIC:
                motion_range = torch.tensor([-.1, .1])
            elif type == rkc.JntType.REVOLUTE:
                motion_range = torch.tensor([-torch.pi, torch.pi])
        self._type = type
        self.motion_range = motion_range

    def assert_motion_value(self, value):
        return
        # if value < self.motion_range[0] or value > self.motion_range[1]:
        #     raise ValueError("Motion value is out of range!")

    def set_motion_value(self, motion_value):
        self._motion_value = motion_value
        if self.type == rkc.JntType.REVOLUTE:
            self._gl_pos_q = self._gl_pos_0
            self._gl_rotmat_q = nkm.rotmat_from_axangle(self._gl_motion_ax, self._motion_value) @ self._gl_rotmat_0
        elif self.type == rkc.JntType.PRISMATIC:
            self._gl_pos_q = self._gl_pos_0 + self._gl_motion_ax * self._motion_value
            self._gl_rotmat_q = self._gl_rotmat_0

    def update_globals(self, pos=torch.zeros(3), rotmat=torch.eye(3),
                       motion_value=torch.tensor(.0, requires_grad=True)):
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
        if self.type == rkc.JntType.REVOLUTE:
            rotmat_by_motion = nkm.rotmat_from_axangle(self.loc_motion_ax, motion_value)
            return self.loc_homomat @ nkm.homomat_from_posrot(pos=torch.zeros(3), rotmat=rotmat_by_motion)
        elif self.type == rkc.JntType.PRISMATIC:
            pos_by_motion = self.loc_motion_ax * motion_value
            return self.loc_homomat @ nkm.homomat_from_posrot(pos=pos_by_motion, rotmat=torch.eye(3))

    def gen_model(self,
                  toggle_frame_0=True,
                  toggle_frame_q=True,
                  toggle_lnk_mesh=False,
                  radius=rkc.JNT_RADIUS,
                  frame_stick_radius=rkc.FRAME_STICK_RADIUS,
                  frame_stick_length=rkc.FRAME_STICK_LENGTH_MEDIUM):
        with torch.no_grad():
            m_col = rkmg.gen_jnt(jnt=self,
                                 toggle_frame_0=toggle_frame_0,
                                 toggle_frame_q=toggle_frame_q,
                                 toggle_lnk_mesh=toggle_lnk_mesh,
                                 radius=radius,
                                 frame_stick_radius=frame_stick_radius,
                                 frame_stick_length=frame_stick_length)
        return m_col


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    from torch.optim import Adam

    base = wd.World(cam_pos=[1., 1., 1.], lookat_pos=[.0, .0, .0])
    mgm.gen_frame().attach_to(base)
    jnt = Joint(loc_motion_ax=torch.tensor([.0, .0, 1.]))
    jnt.lnk.cmodel = mcm.CollisionModel(initor="../../basis/objects/or2fg7_base.stl")
    #
    ref_pos = torch.tensor([.0, .1, .0])
    ref_rotmat = nkm.rotmat_from_euler(torch.pi / 6, torch.pi / 3, torch.pi / 4)
    motion_value = torch.tensor(torch.pi / 2, requires_grad=True, dtype=torch.float)
    jnt.update_globals(pos=ref_pos, rotmat=ref_rotmat, motion_value=motion_value)
    init_model = jnt.gen_model(toggle_lnk_mesh=True)
    init_model.rgb=bc.red
    init_model.attach_to(base)
    # base.run()
    #
    optimizer = Adam([motion_value], lr=0.01)
    for i in range(300):
        jnt.update_globals(pos=ref_pos, rotmat=ref_rotmat, motion_value=motion_value)
        deltap = (jnt.gl_pos_q**2).sum()
        deltaw = (nkm.delta_w_between_rotmat(jnt.gl_rotmat_q, torch.eye(3))**2).sum()
        loss = deltap+deltaw
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(f'Iteration {i}, Loss: {loss.item()}, Params: {motion_value.data.numpy()}')
    jnt.gen_model(toggle_lnk_mesh=True).attach_to(base)

    #
    # print(jnt.gl_pos_q)
    # deltaw = nkm.delta_w_between_rotmat(torch.eye(3), jnt.gl_rotmat_q)
    # deltap = jnt.gl_pos_q.sum()
    # delta = torch.norm(deltap) + torch.norm(deltaw)
    # make_dot(deltap, params={'motion_value': motion_value}).render("jnt_graph", format="png")
    #
    # print(jnt.motion_value.grad)
    #
    base.run()
