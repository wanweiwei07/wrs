import copy
import numpy as np

from wrs import basis as bc, basis as rm, robot_sim as rkn, robot_sim as rko, robot_sim as rkc, modeling as cm, \
    modeling as gm
import wrs.robot_sim._kinematics.jl as rkjl


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
                 bgn_anchor_pos=np.zeros(3),
                 bgn_anchor_rotmat=np.eye(3),
                 end_anchor_loc_pos=np.zeros(3),
                 end_anchor_loc_rotmat=np.eye(3),
                 n_dof=6,
                 cdprimitive_type=cm.CDPrimitiveType.BOX,
                 cdmesh_type=cm.CDMeshType.DEFAULT):
        """
        conf -- configuration: target joint values
        :param name:
        :param bgn_anchor_pos:
        :param bgn_anchor_rotmat:
        :param end_anchor_pos:
        :param
        :param home: number of joints
        :param cdprimitive_type:
        :param cdmesh_type:
        :param name:
        """
        self.name = name
        self.n_dof = n_dof
        self.home = np.zeros(self.n_dof)  # self.n_dof+1 joints in total, the first joint is a anchor joint
        # create anchors
        self.bgn_anchor = rkjl.Anchor(name, pos=bgn_anchor_pos, rotmat=bgn_anchor_rotmat)
        self.end_anchor = rkjl.Anchor(name, pos=end_anchor_loc_pos, rotmat=end_anchor_loc_rotmat)
        # initialize joints and links
        self.jnts = [rkjl.Joint(joint_name=f"j{i}") for i in range(self.n_dof)]
        self._jnt_rngs = self._get_jnt_rngs()
        # default tcp
        self._tcp_jnt_id = self.n_dof - 1
        self.end_anchor_loc_pos = end_anchor_loc_pos
        self.end_anchor_loc_rotmat = end_anchor_loc_rotmat
        # initialize
        self.go_home()
        # collision primitives
        # mesh generator
        self.cdprimitive_type = cdprimitive_type
        self.cdmesh_type = cdmesh_type
        self._nik_solver = rkn.NumIKSolver(self)
        self._oik_solver = rko.OptIKSolver(self)

    @property
    def jnt_rngs(self):
        return self._jnt_rngs

    @property
    def tcp_jnt_id(self):
        return self._tcp_jnt_id

    @tcp_jnt_id.setter
    def tcp_jnt_id(self, value):
        self._tcp_jnt_id = value

    @property
    def tcp_loc_homomat(self):
        return rm.homomat_from_posrot(pos=self.tcp_loc_pos, rotmat=self.tcp_loc_rotmat)

    @property
    def pos(self):
        return self.anchor.pos

    @property
    def rotmat(self):
        return self.anchor.rotmat

    def _get_jnt_rngs(self):
        """
        get jntsrnage
        :return: [[jnt1min, jnt1max], [jnt2min, jnt2max], ...]
        date: 20180602, 20200704osaka
        author: weiwei
        """
        jnt_limits = []
        for i in range(self.n_dof):
            jnt_limits.append(self.jnts[i].motion_range)
        return np.asarray(jnt_limits)

    def forward_kinematics(self,
                           jnt_vals=None,
                           tcp_loc_pos=None,
                           tcp_loc_rotmat=None,
                           toggle_jac=True,
                           update=False,
                           toggle_dbg=False):
        """
        This function will update the global parameters
        :param jnt_vals: a 1xn ndarray where each element indicates the value of a joint (in radian or meter)
        :param update if True, update internal values
        :return: True (succ), False (failure)
        author: weiwei
        date: 20161202, 20201009osaka, 20230823
        """
        tcp_loc_homomat = rm.homomat_from_posrot(pos=tcp_loc_pos, rotmat=tcp_loc_rotmat)
        if not update and jnt_vals is not None:
            homomat = self.bgn_anchor.homomat
            j_pos = np.zeros((self.n_dof, 3))
            j_axis = np.zeros((self.n_dof, 3))
            for i in range(self.tcp_jnt_id + 1):
                j_pos[i, :] = homomat[:3, 3] + homomat[:3, :3] @ self.jnts[i].loc_pos
                homomat = homomat @ self.jnts[i].get_motion_homomat(motion_value=jnt_vals[i])
                j_axis[i, :] = homomat[:3, :3] @ self.jnts[i].loc_motion_ax
            tcp_gl_homomat = homomat @ self.tcp_loc_homomat
            tcp_gl_pos = tcp_gl_homomat[:3, 3]
            tcp_gl_rotmat = tcp_gl_homomat[:3, :3]
            if toggle_jac:
                j_mat = np.zeros((6, self.n_dof))
                for i in range(self.tcp_jnt_id + 1):
                    if self.jnts[i].type == rkc.JntType.REVOLUTE:
                        vec_jnt2tcp = tcp_gl_pos - j_pos[i, :]
                        j_mat[:3, i] = np.cross(j_axis[i, :], vec_jnt2tcp)
                        j_mat[3:6, i] = j_axis[i, :]
                        if toggle_dbg:
                            gm.gen_arrow(spos=j_pos[i, :],
                                         epos=j_pos[i, :] + .2 * j_axis[i, :],
                                         rgba=bc.black).attach_to(base)
                    if self.jnts[i].type == rkc.JntType.PRISMATIC:
                        j_mat[:3, i] = j_axis[i, :]
                return tcp_gl_pos, tcp_gl_rotmat, j_mat
            else:
                return tcp_gl_pos, tcp_gl_rotmat
        if update and jnt_vals is not None:
            pos = self.bgn_anchor.pos
            rotmat = self.bgn_anchor.rotmat
            for i in range(self.n_dof):
                motion_value = jnt_vals[i]
                self.jnts[i].update_pose_considering_refd(pos=pos, rotmat=rotmat, motion_val=motion_value)
                pos = self.jnts[i].gl_pos_q
                rotmat = self.jnts[i].gl_rotmat_q
        tcp_gl_pos, tcp_gl_rotmat = self.cvt_tcp_loc_to_gl()
        if toggle_jac:
            j_mat = np.zeros((6, self.n_dof))
            for i in range(self.tcp_jnt_id + 1):
                if self.jnts[i].type == rkc.JntType.REVOLUTE:
                    vec_jnt2tcp = tcp_gl_pos - self.jnts[i].gl_pos_q
                    j_mat[:3, i] = np.cross(self.jnts[i].gl_motion_ax, vec_jnt2tcp)
                    j_mat[3:6, i] = self.jnts[i].gl_motion_ax
                    if toggle_dbg:
                        gm.gen_arrow(spos=self.jnts[i].gl_pos_q,
                                     epos=self.jnts[i].gl_pos_q + .3 * self.jnts[i].gl_motion_ax,
                                     rgba=bc.black).attach_to(base)
                if self.jnts[i].type == rkc.JntType.PRISMATIC:
                    j_mat[:3, i] = self.jnts[i].gl_motion_ax
            return tcp_gl_rotmat, tcp_gl_rotmat, j_mat
        else:
            return tcp_gl_rotmat, tcp_gl_rotmat

    def jacobian(self, joint_values=None):
        """
        compute the jacobian matrix; use internal values if jnt_values is None
        :param joint_values:
        :param update:
        :return:
        author :weiwei
        date: 20230829
        """
        if joint_values is None:  # use internal, ignore update
            _, _, j_mat = self.forward_kinematics(jnt_vals=self.get_joint_values(),
                                                  toggle_jac=True,
                                                  update=False)
        else:
            _, _, j_mat = self.forward_kinematics(jnt_vals=joint_values,
                                                  toggle_jac=True,
                                                  update=False)
        return j_mat

    def manipulability_val(self, joint_values=None):
        """
        compute the yoshikawa manipulability
        :param tcp_joint_id:
        :param _loc_flange_pos:
        :param _loc_flange_rotmat:
        :return:
        author: weiwei
        date: 20200331
        """
        j_mat = self.jacobian(joint_values=joint_values)
        return np.sqrt(np.linalg.det(j_mat @ j_mat.T))

    def manipulability_mat(self, joint_values=None):
        """
        compute the axes of the manipulability ellipsoid
        :param tcp_joint_id:
        :param _loc_flange_pos:
        :param _loc_flange_rotmat:
        :return: (linear ellipsoid matrix, angular ellipsoid matrix)
        """
        j_mat = self.jacobian(joint_values=joint_values)
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

    def fix_to(self, pos, rotmat):
        self.anchor.pos = pos
        self.anchor.rotmat = rotmat
        return self.go_given_conf(joint_values=self.get_joint_values())

    def reinitialize(self):
        """
        :return:
        author: weiwei
        date: 20201126
        """
        self._jnt_rngs = self._get_jnt_rngs()
        self.go_home()
        self._nik_solver = rkn.NumIKSolver(self)
        self._oik_solver = rko.OptIKSolver(self)

    def set_tcp(self, tcp_joint_id=None, tcp_loc_pos=None, tcp_loc_rotmat=None):
        if tcp_joint_id is not None:
            self.tcp_jnt_id = tcp_joint_id
        if tcp_loc_pos is not None:
            self.tcp_loc_pos = tcp_loc_pos
        if tcp_loc_rotmat is not None:
            self.tcp_loc_rotmat = tcp_loc_rotmat

    def cvt_tcp_loc_to_gl(self):
        gl_pos = self.jnts[self.tcp_jnt_id].gl_pos_q + self.jnts[self.tcp_jnt_id].gl_rotmat_q @ self.tcp_loc_pos
        gl_rotmat = self.jnts[self.tcp_jnt_id].gl_rotmat_q @ self.tcp_loc_rotmat
        return (gl_pos, gl_rotmat)

    def cvt_posrot_in_tcp_to_gl(self,
                                pos_in_loc_tcp=np.zeros(3),
                                rotmat_in_loc_tcp=np.eye(3)):
        """
        given a loc pos and rotmat in loc_tcp, convert it to global frame
        if the last three parameters are given, the code will use them as loc_tcp instead of the internal member vars.
        :param pos_in_loc_tcp: nparray 1x3
        :param rotmat_in_loc_tcp: nparray 3x3
        :param
        :return:
        author: weiwei
        date: 20190312, 20210609
        """
        tcp_gl_pos, tcp_gl_rotmat = self.cvt_tcp_loc_to_gl()
        cvted_gl_pos = tcp_gl_pos + tcp_gl_rotmat.dot(pos_in_loc_tcp)
        cvted_gl_rotmat = tcp_gl_rotmat.dot(rotmat_in_loc_tcp)
        return (cvted_gl_pos, cvted_gl_rotmat)

    def cvt_gl_posrot_to_tcp(self, gl_pos, gl_rotmat):
        """
        given a world pos and world rotmat
        get the relative pos and relative rotmat with respective to the ith jntlnk
        :param gl_pos: 1x3 nparray
        :param gl_rotmat: 3x3 nparray
        :return:
        author: weiwei
        date: 20190312
        """
        tcp_gl_pos, tcp_gl_rotmat = self.cvt_tcp_loc_to_gl()
        return rm.rel_pose(tcp_gl_pos, tcp_gl_rotmat, gl_pos, gl_rotmat)

    def are_joint_values_in_ranges(self, joint_values):
        """
        check if the given jnt_values
        :param joint_values:
        :return:
        author: weiwei
        date: 20220326toyonaka
        """
        if len(joint_values) == self.n_dof:
            raise ValueError('The given joint values do not match n_dof')
        joint_values = np.asarray(joint_values)
        if np.any(joint_values < self.jnt_rngs[:, 0]) or np.any(joint_values > self.jnt_rngs[:, 1]):
            return False
        else:
            return True

    def go_given_conf(self, joint_values):
        """
        move the robot_s to the given pose
        :return: null
        author: weiwei
        date: 20230927osaka
        """
        return self.forward_kinematics(jnt_vals=joint_values, toggle_jac=False, update=True)

    def go_home(self):
        """
        move the robot_s to initial pose
        :return: null
        author: weiwei
        date: 20161211osaka
        """
        return self.go_given_conf(joint_values=self.home)

    def go_zero(self):
        """
        move the robot_s to initial pose
        :return: null
        author: weiwei
        date: 20161211osaka
        """
        return self.go_given_conf(joint_values=np.zeros(self.n_dof))

    def get_joint_values(self):
        """
        get the current joint values
        :return: jnt_values: a 1xn ndarray
        author: weiwei
        date: 20161205tsukuba
        """
        jnt_vals = np.zeros(self.n_dof)
        for i in range(self.n_dof):
            jnt_vals[i] = self.jnts[i].motion_value
        return jnt_vals

    def rand_conf(self):
        """
        generate a random configuration
        author: weiwei
        date: 20200326
        """
        return np.multiply(np.random.rand(self.n_dof),
                           (self.jnt_rngs[:, 1] - self.jnt_rngs[:, 0])) + self.jnt_rngs[:, 0]

    def ik(self,
           tgt_pos: np.ndarray,
           tgt_rotmat: np.ndarray,
           seed_jnt_vals=None,
           max_n_iter=10000,
           toggle_dbg=False):
        """
        Numerical IK by default
        NOTE1: in the numik function of rotjntlinksik,
        in case None is provided, the self.tcp_joint_id, self._loc_flange_pos, self._loc_flange_rotmat will be used
        NOTE2: if list, len(tgtpos)=len(tgtrot) < len(tcp_joint_id)=len(_loc_flange_pos)=len(_loc_flange_rotmat)
        :param tgt_pos: 1x3 nparray, single value or list
        :param tgt_rotmat: 3x3 nparray, single value or list
        :param seed_jnt_vals: the starting configuration used in the numerical iteration
        :param max_n_iter
        :return:
        """
        # tic = time.time()
        # jnt_values = self._nik_solver.dls_rr(tgt_pos=tgt_pos,
        #                                               tgt_rotmat=tgt_rotmat,
        #                                               seed_jnt_values=seed_jnt_values,
        #                                               max_n_iter=max_n_iter,
        #                                               toggle_dbg=toggle_dbg)
        # toc = time.time()
        # print("DLS RR time ", toc - tic)
        # tic = time.time()
        # jnt_values = self._nik_solver.cwln(tgt_pos=tgt_pos,
        #                                             tgt_rotmat=tgt_rotmat,
        #                                             seed_jnt_values=seed_jnt_values,
        #                                             max_n_iter=max_n_iter,
        #                                             toggle_dbg=toggle_dbg)
        # toc = time.time()
        # print("CWLN time ", toc - tic)
        tic = time.time()
        jnt_vals = self._nik_solver.pinv_cw(tgt_pos=tgt_pos,
                                            tgt_rotmat=tgt_rotmat,
                                            seed_jnt_vals=seed_jnt_vals,
                                            max_n_iter=max_n_iter,
                                            toggle_dbg=toggle_dbg)
        toc = time.time()
        print("PINV WC time ", toc - tic)
        # tic = time.time()
        # jnt_values = self._nik_solver.pinv_rr(tgt_pos=tgt_pos,
        #                                             tgt_rotmat=tgt_rotmat,
        #                                             seed_jnt_values=seed_jnt_values,
        #                                             max_n_iter=max_n_iter,
        #                                             toggle_dbg=toggle_dbg)
        # toc = time.time()
        # print("PINV time ", toc - tic)
        # tic = time.time()
        # jnt_values = self._nik_solver.jt_rr(tgt_pos=tgt_pos,
        #                                             tgt_rotmat=tgt_rotmat,
        #                                             seed_jnt_values=seed_jnt_values,
        #                                             max_n_iter=max_n_iter,
        #                                             toggle_dbg=toggle_dbg)
        # toc = time.time()
        # print("JT time ", toc - tic)
        # tic = time.time()
        # jnt_values = self._oik_solver.sqpss(tgt_pos=tgt_pos,
        #                                     tgt_rotmat=tgt_rotmat,
        #                                     seed_jnt_values=seed_jnt_values,
        #                                     max_n_iter=max_n_iter,
        #                                     toggle_dbg=toggle_dbg)
        # toc = time.time()
        # print("SQP-SS time ", toc - tic)
        # tic = time.time()
        # jnt_values = self._oik_solver.sqp(tgt_pos=tgt_pos,
        #                                            tgt_rotmat=tgt_rotmat,
        #                                            seed_jnt_values=seed_jnt_values,
        #                                            max_n_iter=max_n_iter,
        #                                            toggle_dbg=toggle_dbg)
        # toc = time.time()
        # print("SQP time ", toc - tic)
        return jnt_vals

    def copy(self):
        return copy.deepcopy(self)


if __name__ == "__main__":
    import time
    import wrs.visualization.panda.world as wd
    from tqdm import tqdm

    base = wd.World(cam_pos=[1.25, .75, .75], lookat_pos=[0, 0, .3])
    gm.gen_frame().attach_to(base)

    jlc = JLChain(n_dof=6)
    jlc.jnts[0].loc_pos = np.array([0, 0, 0])
    jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[0].motion_range = np.array([-np.pi / 2, np.pi / 2])
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
    jlc.tcp_loc_pos = np.array([0, 0, .01])
    jlc.reinitialize()
    seed_jnt_vals = jlc.get_joint_values()

    success = 0
    time_list = []
    tgt_list = []
    for i in tqdm(range(100), desc="ik"):
        jnts = jlc.rand_conf()
        tgt_pos, tgt_rotmat = jlc.forward_kinematics(jnt_vals=jnts, update=False, toggle_jac=False)
        tic = time.time()
        joint_values_with_dbg_info = jlc.ik(tgt_pos=tgt_pos,
                                            tgt_rotmat=tgt_rotmat,
                                            seed_jnt_vals=seed_jnt_vals,
                                            max_n_iter=30)
        toc = time.time()
        time_list.append(toc - tic)
        if joint_values_with_dbg_info is not None:
            success += 1
        else:
            print(repr(jnts), repr(tgt_pos), repr(tgt_rotmat))
            tgt_list.append((tgt_pos, tgt_rotmat))
    print(success)
    print('average', np.mean(time_list))
    print('max', np.max(time_list))
    print('min', np.min(time_list))
    base.run()
