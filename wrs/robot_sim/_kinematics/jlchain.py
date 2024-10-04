import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim._kinematics.jl as rkjl
import wrs.robot_sim._kinematics.model_generator as rkmg
import wrs.robot_sim._kinematics.ik_dd as ikdd
import wrs.robot_sim._kinematics.ik_num as ikn
import wrs.robot_sim._kinematics.ik_opt as iko
import wrs.robot_sim._kinematics.constant as const


# TODO delay finalize
# TODO joint gl -> flange

class JLChain(object):
    """
    Joint Link Chain, no branches allowed
    Usage:
    1. Create a JLChain instance with a given n_dof and update its parameters for particular definition
    2. Define multiple instances of this class to compose a complicated structure
    3. Use mimic for underactuated or coupled mechanism
    """

    def __init__(self,
                 name="auto",
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 n_dof=6):
        """
        conf -- configuration: target joint values
        :param name:
        :param pos:
        :param rotmat:
        :param home: number of joints
        :param name:
        """
        self.name = name
        self.n_dof = n_dof
        self.home = np.zeros(self.n_dof)
        # initialize anchor
        self.anchor = rkjl.Anchor(name=f"{name}_anchor", pos=pos, rotmat=rotmat)
        # initialize joints and links
        self.jnts = [rkjl.Joint(name=f"{name}_j{i}") for i in range(self.n_dof)]
        self._jnt_ranges = self._get_jnt_ranges()
        # default flange joint id, loc_xxx are considered described in it
        self._flange_jnt_id = self.n_dof - 1
        # default flange for cascade connection
        self._loc_flange_pos = np.zeros(3)
        self._loc_flange_rotmat = np.eye(3)
        self._gl_flange_pos = np.zeros(3)
        self._gl_flange_rotmat = np.zeros(3)
        # finalizing tag
        self._is_finalized = False
        # iksolver
        self._ik_solver = None

    @staticmethod
    def assert_finalize_decorator(method):
        def wrapper(self, *args, **kwargs):
            if self._is_finalized:
                return method(self, *args, **kwargs)
            else:
                raise ValueError("JLChain is not finalized.")

        return wrapper

    @property
    def jnt_ranges(self):
        return self._jnt_ranges

    @property
    def flange_jnt_id(self):
        return self._flange_jnt_id

    @flange_jnt_id.setter
    def flange_jnt_id(self, value):
        self._flange_jnt_id = value

    @property
    def loc_flange_pos(self):
        return self._loc_flange_pos

    @property
    def loc_flange_rotmat(self):
        return self._loc_flange_rotmat

    @property
    def loc_flange_homomat(self):
        return rm.homomat_from_posrot(pos=self._loc_flange_pos, rotmat=self._loc_flange_rotmat)

    @property
    @assert_finalize_decorator
    def gl_flange_pos(self):
        return self._gl_flange_pos

    @property
    @assert_finalize_decorator
    def gl_flange_rotmat(self):
        return self._gl_flange_rotmat

    @property
    @assert_finalize_decorator
    def gl_flange_pose(self):
        return (self._gl_flange_pos, self._gl_flange_rotmat)

    @property
    def pos(self):
        return self.anchor.pos

    @property
    def rotmat(self):
        return self.anchor.rotmat

    def _get_jnt_ranges(self):
        """
        get jnt ranges
        :return: [[jnt1min, jnt1max], [jnt2min, jnt2max], ...]
        date: 20180602, 20200704osaka
        author: weiwei
        """
        jnt_limits = []
        for i in range(self.n_dof):
            jnt_limits.append(self.jnts[i].motion_range)
        return np.asarray(jnt_limits)

    def fk(self, jnt_values, toggle_jacobian=False, update=False):
        """
        :param jnt_values: a 1xn ndarray where each element indicates the value of a joint (in radian or meter)
        :param toggle_jacobian: return jacobian matrix if true
        :param update if True, update internal values
        :return: True (succ), False (failure)
        author: weiwei
        date: 20161202, 20201009osaka, 20230823
        """
        if not update:
            homomat = self.anchor.gl_flange_homomat_list[0]
            jnt_pos = np.zeros((self.n_dof, 3))
            jnt_motion_ax = np.zeros((self.n_dof, 3))
            for i in range(self.flange_jnt_id + 1):
                jnt_pos[i, :] = homomat[:3, 3] + homomat[:3, :3] @ self.jnts[i].loc_pos
                homomat = homomat @ self.jnts[i].get_motion_homomat(motion_value=jnt_values[i])
                jnt_motion_ax[i, :] = homomat[:3, :3] @ self.jnts[i].loc_motion_ax
            gl_flange_homomat = homomat @ self.loc_flange_homomat
            gl_flange_pos = gl_flange_homomat[:3, 3]
            gl_flange_rotmat = gl_flange_homomat[:3, :3]
            if toggle_jacobian:
                j_mat = np.zeros((6, self.n_dof))
                for i in range(self.flange_jnt_id + 1):
                    if self.jnts[i].type == const.JntType.REVOLUTE:
                        j2t_vec = gl_flange_pos - jnt_pos[i, :]
                        j_mat[:3, i] = np.cross(jnt_motion_ax[i, :], j2t_vec)
                        j_mat[3:6, i] = jnt_motion_ax[i, :]
                    if self.jnts[i].type == const.JntType.PRISMATIC:
                        j_mat[:3, i] = jnt_motion_ax[i, :]
                return gl_flange_pos, gl_flange_rotmat, j_mat
            else:
                return gl_flange_pos, gl_flange_rotmat
        else:
            # self.anchor.update_pose()
            pos = self.anchor.gl_flange_pose_list[0][0]
            rotmat = self.anchor.gl_flange_pose_list[0][1]
            for i in range(self.n_dof):
                motion_value = jnt_values[i]
                self.jnts[i].update_globals(pos=pos, rotmat=rotmat, motion_value=motion_value)
                pos = self.jnts[i].gl_pos_q
                rotmat = self.jnts[i].gl_rotmat_q
            self._gl_flange_pos, self._gl_flange_rotmat = self._compute_gl_flange()
            if toggle_jacobian:
                j_mat = np.zeros((6, self.n_dof))
                for i in range(self.flange_jnt_id + 1):
                    if self.jnts[i].type == const.JntType.REVOLUTE:
                        j2t_vec = self._gl_flange_pos - self.jnts[i].gl_pos_q
                        j_mat[:3, i] = np.cross(self.jnts[i].gl_motion_ax, j2t_vec)
                        j_mat[3:6, i] = self.jnts[i].gl_motion_ax
                    if self.jnts[i].type == const.JntType.PRISMATIC:
                        j_mat[:3, i] = self.jnts[i].gl_motion_ax
                return self._gl_flange_pos, self._gl_flange_rotmat, j_mat
            else:
                return self._gl_flange_pos, self._gl_flange_rotmat

    def jacobian(self, jnt_values=None):
        """
        compute the jacobian matrix; use internal values if jnt_values is None
        :param jnt_values:
        :return:
        author :weiwei
        date: 20230829
        """
        if jnt_values is None:  # use internal, ignore update
            j_mat = np.zeros((6, self.n_dof))
            for i in range(self.flange_jnt_id + 1):
                if self.jnts[i].type == const.JntType.REVOLUTE:
                    j2t_vec = self._gl_flange_pos - self.jnts[i].gl_pos_q
                    j_mat[:3, i] = np.cross(self.jnts[i].gl_motion_ax, j2t_vec)
                    j_mat[3:6, i] = self.jnts[i].gl_motion_ax
                if self.jnts[i].type == const.JntType.PRISMATIC:
                    j_mat[:3, i] = self.jnts[i].gl_motion_ax
        else:
            _, _, j_mat = self.fk(jnt_values=jnt_values,
                                  toggle_jacobian=True,
                                  update=False)
        return j_mat

    def manipulability_val(self, jnt_values=None):
        """
        compute the yoshikawa manipulability
        :param jnt_values:
        :return:
        author: weiwei
        date: 20200331
        """
        j_mat = self.jacobian(jnt_values=jnt_values)
        return np.sqrt(np.linalg.det(j_mat @ j_mat.T))

    def manipulability_mat(self, jnt_values=None):
        """
        compute the axes of the manipulability ellipsoid
        :param jnt_values:
        :return: (linear ellipsoid matrix, angular ellipsoid matrix)
        """
        j_mat = self.jacobian(jnt_values=jnt_values)
        # linear ellipsoid
        linear_j_dot_jt = j_mat[:3, :] @ j_mat[:3, :].T
        eig_values, eig_vecs = np.linalg.eig(linear_j_dot_jt)
        linear_ellipsoid_mat = np.eye(3)
        linear_ellipsoid_mat[:, 0] = np.sqrt(eig_values[0]) * eig_vecs[:, 0]
        linear_ellipsoid_mat[:, 1] = np.sqrt(eig_values[1]) * eig_vecs[:, 1]
        linear_ellipsoid_mat[:, 2] = np.sqrt(eig_values[2]) * eig_vecs[:, 2]
        # angular ellipsoid
        angular_j_dot_jt = j_mat[3:, :] @ j_mat[3:, :].T
        eig_values, eig_vecs = np.linalg.eig(angular_j_dot_jt)
        angular_ellipsoid_mat = np.eye(3)
        angular_ellipsoid_mat[:, 0] = np.sqrt(eig_values[0]) * eig_vecs[:, 0]
        angular_ellipsoid_mat[:, 1] = np.sqrt(eig_values[1]) * eig_vecs[:, 1]
        angular_ellipsoid_mat[:, 2] = np.sqrt(eig_values[2]) * eig_vecs[:, 2]
        return (linear_ellipsoid_mat, angular_ellipsoid_mat)

    def fix_to(self, pos, rotmat, jnt_values=None):
        self.anchor.fix_to(pos, rotmat)
        if jnt_values is None:
            return self.goto_given_conf(jnt_values=self.get_jnt_values())
        else:
            return self.goto_given_conf(jnt_values=jnt_values)

    def finalize(self, ik_solver=None, identifier_str="test", **kwargs):
        """
        ddik is both fast and has high success rate, but it required prebuilding a data file.
        tracik is also fast and reliable, but it is a bit slower and energe-intensive.
        pinv_wc is fast but has low success rate. it is used as a backbone for ddik.
        sqpss has high success rate but is very slow.
        :param ik_solver: 'd' for ddik; 'n' for numik.pinv_wc; 'o' for optik.sqpss; 't' for tracik; default: None
        :param identifier_str: a string identifier for data (using a robot name is more preferred than the default)
        :**kwargs: path for DDIKSolver
        :return:
        author: weiwei
        date: 20201126, 20231111
        """
        self._jnt_ranges = self._get_jnt_ranges()
        self.go_home()
        self._is_finalized = True
        if ik_solver == 'd':
            self._ik_solver = ikdd.DDIKSolver(self, identifier_str=identifier_str)
        elif ik_solver == 'n':
            self._ik_solver = ikn.NumIKSolver(self)
        elif ik_solver == 'o':
            self._ik_solver = iko.OptIKSolver(self)
        elif ik_solver == 'a': # analytical ik, user defined
            self._ik_solver = None

    def set_flange(self, loc_flange_pos=None, loc_flange_rotmat=None):
        if loc_flange_pos is not None:
            self._loc_flange_pos = loc_flange_pos
        if loc_flange_rotmat is not None:
            self._loc_flange_rotmat = loc_flange_rotmat
        self._is_finalized = False

    def _compute_gl_flange(self):
        if self.n_dof >= 1:
            gl_pos = self.jnts[self.flange_jnt_id].gl_pos_q + self.jnts[
                self.flange_jnt_id].gl_rotmat_q @ self._loc_flange_pos
            gl_rotmat = self.jnts[self.flange_jnt_id].gl_rotmat_q @ self._loc_flange_rotmat
        else:
            pos = self.anchor.gl_flange_pose_list[0][0]
            rotmat = self.anchor.gl_flange_pose_list[0][1]
            gl_pos = pos + rotmat @ self._loc_flange_pos
            gl_rotmat = rotmat @ self._loc_flange_rotmat
        return (gl_pos, gl_rotmat)

    @assert_finalize_decorator
    def cvt_pose_in_flange_to_gl(self, loc_pos=np.zeros(3), loc_rotmat=np.eye(3)):
        """
        given a loc pos and rotmat in the flange frame, convert it to global frame
        :param loc_pos: nparray 1x3 in the flange frame
        :param loc_rotmat: nparray 3x3 in the flange frame
        :param
        :return:
        author: weiwei
        date: 202403032
        """
        tmp_gl_pos = self._gl_flange_pos + self._gl_flange_rotmat @ loc_pos
        tmp_gl_rotmat = self._gl_flange_rotmat @ loc_rotmat
        return (tmp_gl_pos, tmp_gl_rotmat)

    @assert_finalize_decorator
    def cvt_gl_to_flange(self, gl_pos, gl_rotmat):
        """
        given a global pos and rotmat, get its relative pos and rotmat to the flange frame
        :param gl_pos: 1x3 nparray
        :param gl_rotmat: 3x3 nparray
        :return:
        author: weiwei
        date: 20190312
        """
        return rm.rel_pose(self.gl_flange_pos, self.gl_flange_rotmat, gl_pos, gl_rotmat)

    def cvt_pose_in_flange_to_functional(self, pos_in_flange, rotmat_in_flange):
        """
        convert a pose in the flange frame to the functional joint frame
        :param pos_in_flange:
        :param rotmat_in_flange:
        :return:
        author: weiwei
        date: 20240301
        """
        tmp_loc_pos = self._loc_flange_pos + self._loc_flange_rotmat @ pos_in_flange
        tmp_loc_rotmat = self._loc_flange_rotmat @ rotmat_in_flange
        return (tmp_loc_pos, tmp_loc_rotmat)

    def are_jnts_in_ranges(self, jnt_values):
        """
        check if the given jnt_values are in range
        :param jnt_values:
        :return:
        author: weiwei
        date: 20220326toyonaka
        """
        if len(jnt_values) != self.n_dof:
            raise ValueError(f"The given joint values do not match n_dof: {len(jnt_values)} vs. {self.n_dof}")
        jnt_values = np.asarray(jnt_values)
        if np.any(jnt_values < self.jnt_ranges[:, 0]) or np.any(jnt_values > self.jnt_ranges[:, 1]):
            print("Joints are out of ranges!")
            return False
        else:
            return True

    def goto_given_conf(self, jnt_values):
        """
        move to the given configuration
        :param jnt_values
        :return: null
        author: weiwei
        date: 20230927osaka
        """
        return self.fk(jnt_values=jnt_values, toggle_jacobian=False, update=True)

    def go_home(self):
        """
        move to home configuration
        :return: null
        author: weiwei
        date: 20161211osaka
        """
        return self.goto_given_conf(jnt_values=self.home)

    def go_zero(self):
        """
        move to zero configuration
        :return: null
        author: weiwei
        date: 20161211osaka
        """
        return self.goto_given_conf(jnt_values=np.zeros(self.n_dof))

    def get_jnt_values(self):
        """
        get the current joint values
        :return: jnt_values: a 1xn ndarray
        author: weiwei
        date: 20161205tsukuba
        """
        jnt_values = np.zeros(self.n_dof)
        for i in range(self.n_dof):
            jnt_values[i] = self.jnts[i].motion_value
        return jnt_values

    def rand_conf(self):
        """
        generate a random configuration
        author: weiwei
        date: 20200326
        """
        return np.random.rand(self.n_dof) * (self.jnt_ranges[:, 1] - self.jnt_ranges[:, 0]) + self.jnt_ranges[:, 0]

    @assert_finalize_decorator
    def ik(self,
           tgt_pos,
           tgt_rotmat,
           seed_jnt_values=None,
           toggle_dbg=False):
        """
        :param tgt_pos: 1x3 nparray
        :param tgt_rotmat: 3x3 nparray
        :param seed_jnt_values: the starting configuration used in the numerical iteration
        :return:
        """
        if self._ik_solver is None:
            raise Exception("IK solver undefined. Use JLChain.finalize to define it.")
        jnt_values = self._ik_solver(tgt_pos=tgt_pos,
                                     tgt_rotmat=tgt_rotmat,
                                     seed_jnt_values=seed_jnt_values,
                                     toggle_dbg=toggle_dbg)
        return jnt_values

    def gen_stickmodel(self,
                       stick_rgba=rm.const.lnk_stick_rgba,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=True,
                       toggle_actuation=False,
                       name='jlc_stick_model',
                       jnt_radius=const.JNT_RADIUS,
                       lnk_radius=const.LNK_STICK_RADIUS,
                       jnt_alpha=1,
                       lnk_alpha=1):
        return rkmg.gen_jlc_stick(jlc=self,
                                  stick_rgba=stick_rgba,
                                  toggle_jnt_frames=toggle_jnt_frames,
                                  toggle_flange_frame=toggle_flange_frame,
                                  toggle_actuation=toggle_actuation,
                                  name=name,
                                  jnt_radius=jnt_radius,
                                  lnk_radius=lnk_radius,
                                  jnt_alpha=jnt_alpha,
                                  lnk_alpha=lnk_alpha)

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_flange_frame=False,
                      toggle_jnt_frames=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='jlc_mesh_model'):
        return rkmg.gen_jlc_mesh(jlc=self,
                                 rgb=rgb,
                                 alpha=alpha,
                                 toggle_flange_frame=toggle_flange_frame,
                                 toggle_jnt_frames=toggle_jnt_frames,
                                 toggle_cdprim=toggle_cdprim,
                                 toggle_cdmesh=toggle_cdmesh,
                                 name=name)

    def test_ik_success_rate(self, n_times=100):
        success = 0
        time_list = []
        tgt_list = []
        for i in tqdm(range(n_times), desc="ik"):
            random_jnts = self.rand_conf()
            flange_pos, flange_rotmat = self.fk(jnt_values=random_jnts, update=False, toggle_jacobian=False)
            tic = time.time()
            solved_jnt_values = self.ik(tgt_pos=flange_pos,
                                        tgt_rotmat=flange_rotmat,
                                        # seed_jnt_values=seed_jnt_values,
                                        toggle_dbg=False)
            toc = time.time()
            time_list.append(toc - tic)
            if solved_jnt_values is not None:
                success += 1
            else:
                tgt_list.append((flange_pos, flange_rotmat))
        print("------------------testing results------------------")
        print(f"The current success rate is: {success / n_times * 100}%")
        print('average time cost', np.mean(time_list))
        print('max', np.max(time_list))
        print('min', np.min(time_list))
        print('std', np.std(time_list))
        return success / n_times


if __name__ == "__main__":
    import time
    from tqdm import tqdm
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1.25, .75, .75], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)

    jlc = JLChain(n_dof=6)
    jlc.jnts[0].loc_pos = np.array([0, 0, 0])
    jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[0].motion_range = np.array([-np.pi / 2, np.pi / 2])
    # jlc.jnts[1].change_type(const.JntType.PRISMATIC)
    jlc.jnts[1].loc_pos = np.array([0, 0, .05])
    jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
    jlc.jnts[1].motion_range = np.array([-np.pi / 2, np.pi / 2])
    jlc.jnts[2].loc_pos = np.array([0, 0, .2])
    jlc.jnts[2].loc_motion_ax = np.array([0, 1, 0])
    jlc.jnts[2].motion_range = np.array([-np.pi, np.pi])
    jlc.jnts[3].loc_pos = np.array([0, 0, .2])
    jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[3].motion_range = np.array([-np.pi / 2, np.pi / 2])
    jlc.jnts[4].loc_pos = np.array([0, 0, .1])
    jlc.jnts[4].loc_motion_ax = np.array([0, 1, 0])
    jlc.jnts[4].motion_range = np.array([-np.pi / 2, np.pi / 2])
    jlc.jnts[5].loc_pos = np.array([0, 0, .05])
    jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[5].motion_range = np.array([-np.pi / 2, np.pi / 2])
    jlc._loc_flange_pos = np.array([0, 0, .01])
    jlc._loc_flange_pos = np.array([0.1, 0.1, 0.1])
    jlc.finalize(ik_solver=None)
    # jlc.finalize(ik_solver='d')
    jlc.gen_stickmodel(toggle_jnt_frames=True, toggle_flange_frame=True).attach_to(base)
    # seed_jnt_values = jlc.get_jnt_values()
    seed_jnt_values = None
    base.run()

    success = 0
    num_win = 0
    opt_win = 0
    time_list = []
    tgt_list = []
    for i in tqdm(range(100), desc="ik"):
        random_jnts = jlc.rand_conf()
        tgt_pos, tgt_rotmat = jlc.fk(jnt_values=random_jnts, update=False, toggle_jacobian=False)
        mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat, ax_length=.2).attach_to(base)
        tic = time.time()
        joint_values_with_dbg_info = jlc.ik(tgt_pos=tgt_pos,
                                            tgt_rotmat=tgt_rotmat,
                                            seed_jnt_values=seed_jnt_values,
                                            toggle_dbg=False)
        # print("success!")
        # base.run()
        toc = time.time()
        time_list.append(toc - tic)
        print(time_list[-1])
        if joint_values_with_dbg_info is not None:
            success += 1
            if joint_values_with_dbg_info[0] == 'o':
                opt_win += 1
            elif joint_values_with_dbg_info[0] == 'n':
                num_win += 1
            mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat, ax_length=.2).attach_to(base)
            jlc.fk(jnt_values=joint_values_with_dbg_info, update=True, toggle_jacobian=False)
            jlc.gen_stickmodel(stick_rgba=rm.const.navy_blue, toggle_flange_frame=True,
                               toggle_jnt_frames=True).attach_to(base)
            base.run()
        else:
            tgt_list.append((tgt_pos, tgt_rotmat))
    print(f'success: {success}')
    print(f'num_win: {num_win}, opt_win: {opt_win}')
    print('average', np.mean(time_list))
    print('max', np.max(time_list))
    print('min', np.min(time_list))
    base.run()
