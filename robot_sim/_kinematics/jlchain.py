import math
import os
import copy
import random

import numpy as np
import time
import basis.constant as bc
import basis.utils as bu
import basis.robot_math as rm
import modeling.constant as mc
import modeling.collision_model as mcm
import robot_sim._kinematics.constant as rkc
import robot_sim._kinematics.jl as rkjl
import robot_sim._kinematics.ik_num as rkn
import robot_sim._kinematics.ik_opt as rko
import robot_sim._kinematics.ik_dd as rkd
import robot_sim._kinematics.ik_trac as rkt
import basis.constant as cst


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
                 n_dof=6,
                 cdprimitive_type=mc.CDPType.BOX,
                 cdmesh_type=mc.CDMType.DEFAULT):
        """
        conf -- configuration: target joint values
        :param name:
        :param pos:
        :param rotmat:
        :param home: number of joints
        :param cdprimitive_type:
        :param cdmesh_type:
        :param name:
        """
        self.name = name
        self.n_dof = n_dof
        self.home = np.zeros(self.n_dof)  # self.n_dof joints plus one anchor
        # initialize joints and links
        self.anchor = rkjl.Anchor(name, pos=pos, rotmat=rotmat)
        self.jnts = [rkjl.Joint(name=f"j{i}") for i in range(self.n_dof)]
        self._jnt_ranges = self._get_jnt_ranges()
        # default functional joint id, loc_xxx are considered described in it
        self._functional_jnt_id = self.n_dof - 1
        # default tcp for inverse kinematics
        self.loc_tcp_pos = np.zeros(3)
        self.loc_tcp_rotmat = np.eye(3)
        # default flange for cascade connection
        self.loc_flange_pos = np.zeros(3)
        self.loc_flange_rotmat = np.eye(3)
        # initialize
        self.go_home()
        # collision primitives
        # mesh generator
        self.cdprimitive_type = cdprimitive_type
        self.cdmesh_type = cdmesh_type
        # iksolver
        self._ik_solver = None

    @property
    def jnt_ranges(self):
        return self._jnt_ranges

    @property
    def functional_jnt_id(self):
        return self._functional_jnt_id

    @functional_jnt_id.setter
    def functional_jnt_id(self, value):
        self._functional_jnt_id = value

    @property
    def loc_tcp_homomat(self):
        return rm.homomat_from_posrot(pos=self.loc_tcp_pos, rotmat=self.loc_tcp_rotmat)

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

    def fk(self, jnt_values, toggle_jacobian=True, update=False):
        """
        :param jnt_values: a 1xn ndarray where each element indicates the value of a joint (in radian or meter)
        :param toggle_jacobian: return jacobian matrix if true
        :param update if True, update internal values
        :return: True (succ), False (failure)
        author: weiwei
        date: 20161202, 20201009osaka, 20230823
        """
        if not update:
            homomat = self.anchor.gl_flange_homomat
            jnt_pos = np.zeros((self.n_dof, 3))
            jnt_motion_ax = np.zeros((self.n_dof, 3))
            for i in range(self.functional_jnt_id + 1):
                jnt_motion_ax[i, :] = homomat[:3, :3] @ self.jnts[i].loc_motion_ax
                if self.jnts[i].type == rkc.JntType.REVOLUTE:
                    jnt_pos[i, :] = homomat[:3, 3] + homomat[:3, :3] @ self.jnts[i].loc_pos
                homomat = homomat @ self.jnts[i].get_motion_homomat(motion_value=jnt_values[i])
            gl_tcp_homomat = homomat @ self.loc_tcp_homomat
            gl_tcp_pos = gl_tcp_homomat[:3, 3]
            gl_tcp_rotmat = gl_tcp_homomat[:3, :3]
            if toggle_jacobian:
                j_mat = np.zeros((6, self.n_dof))
                for i in range(self.functional_jnt_id + 1):
                    if self.jnts[i].type == rkc.JntType.REVOLUTE:
                        vec_jnt2tcp = gl_tcp_pos - jnt_pos[i, :]
                        j_mat[:3, i] = np.cross(jnt_motion_ax[i, :], vec_jnt2tcp)
                        j_mat[3:6, i] = jnt_motion_ax[i, :]
                    if self.jnts[i].type == rkc.JntType.PRISMATIC:
                        j_mat[:3, i] = jnt_motion_ax[i, :]
                return gl_tcp_pos, gl_tcp_rotmat, j_mat
            else:
                return gl_tcp_pos, gl_tcp_rotmat
        else:
            self.anchor.update_pose()
            pos = self.anchor.gl_flange_pos
            rotmat = self.anchor.gl_flange_rotmat
            for i in range(self.n_dof):
                motion_value = jnt_values[i]
                self.jnts[i].update_globals(pos=pos, rotmat=rotmat, motion_value=motion_value)
                pos = self.jnts[i].gl_pos_q
                rotmat = self.jnts[i].gl_rotmat_q
            gl_tcp_pos, gl_tcp_rotmat = self.get_gl_tcp()
            if toggle_jacobian:
                j_mat = np.zeros((6, self.n_dof))
                for i in range(self.functional_jnt_id + 1):
                    if self.jnts[i].type == rkc.JntType.REVOLUTE:
                        vec_jnt2tcp = gl_tcp_pos - self.jnts[i].gl_pos_q
                        j_mat[:3, i] = np.cross(self.jnts[i].gl_motion_ax, vec_jnt2tcp)
                        j_mat[3:6, i] = self.jnts[i].gl_motion_ax
                    if self.jnts[i].type == rkc.JntType.PRISMATIC:
                        j_mat[:3, i] = self.jnts[i].gl_motion_ax
                return gl_tcp_pos, gl_tcp_rotmat, j_mat
            else:
                return gl_tcp_pos, gl_tcp_rotmat

    def jacobian(self, jnt_values=None):
        """
        compute the jacobian matrix; use internal values if jnt_values is None
        :param jnt_values:
        :param update:
        :return:
        author :weiwei
        date: 20230829
        """
        if jnt_values is None:  # use internal, ignore update
            _, _, j_mat = self.fk(jnt_values=self.get_jnt_values(),
                                  toggle_jacobian=True,
                                  update=False)
        else:
            _, _, j_mat = self.fk(jnt_values=jnt_values,
                                  toggle_jacobian=True,
                                  update=False)
        return j_mat

    def manipulability_val(self, jnt_values=None):
        """
        compute the yoshikawa manipulability
        :param tcp_joint_id:
        :param loc_tcp_pos:
        :param loc_tcp_rotmat:
        :return:
        author: weiwei
        date: 20200331
        """
        j_mat = self.jacobian(jnt_values=jnt_values)
        return np.sqrt(np.linalg.det(j_mat @ j_mat.T))

    def manipulability_mat(self, jnt_values=None):
        """
        compute the axes of the manipulability ellipsoid
        :param tcp_joint_id:
        :param loc_tcp_pos:
        :param loc_tcp_rotmat:
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
        self.anchor.pos = pos
        self.anchor.rotmat = rotmat
        if jnt_values is None:
            return self.go_given_conf(jnt_values=self.get_jnt_values())
        else:
            return self.go_given_conf(jnt_values=jnt_values)

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
        if ik_solver == 'd':
            self._ik_solver = rkd.DDIKSolver(self, identifier_str=identifier_str)

    def set_tcp(self, loc_tcp_pos=None, loc_tcp_rotmat=None):
        if loc_tcp_pos is not None:
            self.loc_tcp_pos = loc_tcp_pos
        if loc_tcp_rotmat is not None:
            self.loc_tcp_rotmat = loc_tcp_rotmat

    def cvt_loc_to_gl(self, loc_pos, loc_rotmat):
        if self.n_dof >= 1:
            gl_pos = self.jnts[self.functional_jnt_id].gl_pos_q + self.jnts[
                self.functional_jnt_id].gl_rotmat_q @ loc_pos
            gl_rotmat = self.jnts[self.functional_jnt_id].gl_rotmat_q @ loc_rotmat
        else:
            gl_pos = self.anchor.gl_flange_pos + self.anchor.gl_flange_rotmat @ loc_pos
            gl_rotmat = self.anchor.gl_flange_rotmat @ self.loc_rotmat
        return (gl_pos, gl_rotmat)

    def get_gl_tcp(self):
        gl_tcp_pos, gl_tcp_rotmat = self.cvt_loc_to_gl(loc_pos=self.loc_tcp_pos, loc_rotmat=self.loc_tcp_rotmat)
        return (gl_tcp_pos, gl_tcp_rotmat)

    def cvt_pose_in_tcp_to_gl(self,
                              pos_in_tcp=np.zeros(3),
                              rotmat_in_tcp=np.eye(3)):
        """
        given a loc pos and rotmat in tcp, convert it to global frame
        :param pos_in_tcp: nparray 1x3 in the frame defined by loc_tcp_pos, loc_tcp_rotmat
        :param rotmat_in_tcp: nparray 3x3 in the frame defined by loc_tcp_pos, loc_tcp_rotmat
        :param
        :return:
        author: weiwei
        date: 20190312, 20210609
        """
        gl_tcp_pos, gl_tcp_rotmat = self.get_gl_tcp()
        tmp_gl_pos = gl_tcp_pos + gl_tcp_rotmat.dot(pos_in_tcp)
        tmp_gl_rotmat = gl_tcp_rotmat.dot(rotmat_in_tcp)
        return (tmp_gl_pos, tmp_gl_rotmat)

    def cvt_gl_pose_to_tcp(self, gl_pos, gl_rotmat):
        """
        given a world pos and world rotmat
        get the relative pos and relative rotmat with respective to the tcp
        :param gl_pos: 1x3 nparray
        :param gl_rotmat: 3x3 nparray
        :return:
        author: weiwei
        date: 20190312
        """
        gl_tcp_pos, gl_tcp_rotmat = self.get_gl_tcp()
        return rm.rel_pose(gl_tcp_pos, gl_tcp_rotmat, gl_pos, gl_rotmat)

    def cvt_pose_in_flange_to_functional(self, pos_in_flange, rotmat_in_flange):
        """
        convert a pose in the flange frame to the functional joint frame
        :param pos_in_flange:
        :param rotmat_in_flange:
        :return:
        author: weiwei
        date: 20240301
        """
        tmp_loc_pos = self.loc_flange_pos + self.loc_flange_rotmat @ pos_in_flange
        tmp_loc_rotmat = self.loc_flange_rotmat @ rotmat_in_flange
        return (tmp_loc_pos, tmp_loc_rotmat)

    def get_gl_flange(self):
        gl_flange_pos, gl_flange_rotmat = self.cvt_loc_to_gl(loc_pos=self.loc_flange_pos,
                                                             loc_rotmat=self.loc_flange_rotmat)
        return (gl_flange_pos, gl_flange_rotmat)

    def cvt_pose_in_flange_to_gl(self,
                                 pos_in_flange=np.zeros(3),
                                 rotmat_in_flange=np.eye(3)):
        """
        given a loc pos and rotmat in flange, convert it to global frame
        :param pos_in_flange: nparray 1x3 in the frame defined by loc_flange_pos, loc_flange_rotmat
        :param rotmat_in_flange: nparray 3x3 in the frame defined by loc_flange_pos, loc_flange_rotmat
        :param
        :return:
        author: weiwei
        date: 20240301
        """
        gl_flange_pos, gl_flange_rotmat = self.get_gl_flange()
        tmp_gl_pos = gl_flange_pos + gl_flange_rotmat.dot(pos_in_flange)
        tmp_gl_rotmat = gl_flange_rotmat.dot(rotmat_in_flange)
        return (tmp_gl_pos, tmp_gl_rotmat)

    def cvt_gl_pose_to_flange(self, gl_pos, gl_rotmat):
        """
        given a world pos and world rotmat
        get the relative pos and relative rotmat with respective to the flange
        :param gl_pos: 1x3 nparray
        :param gl_rotmat: 3x3 nparray
        :return:
        author: weiwei
        date: 20190312
        """
        gl_flange_pos, gl_flange_rotmat = self.get_flange_tcp()
        return rm.rel_pose(gl_flange_pos, gl_flange_rotmat, gl_pos, gl_rotmat)

    def are_jnts_in_ranges(self, jnt_values):
        """
        check if the given jnt_values
        :param jnt_values:
        :return:
        author: weiwei
        date: 20220326toyonaka
        """
        if len(jnt_values) == self.n_dof:
            raise ValueError('The given joint values do not match n_dof')
        jnt_values = np.asarray(jnt_values)
        if np.any(jnt_values < self.jnt_ranges[:, 0]) or np.any(jnt_values > self.jnt_ranges[:, 1]):
            return False
        else:
            return True

    def go_given_conf(self, jnt_values):
        """
        move the robot_s to the given pose
        :return: null
        author: weiwei
        date: 20230927osaka
        """
        return self.fk(jnt_values=jnt_values, toggle_jacobian=False, update=True)

    def go_home(self):
        """
        move the robot_s to initial pose
        :return: null
        author: weiwei
        date: 20161211osaka
        """
        return self.go_given_conf(jnt_values=self.home)

    def go_zero(self):
        """
        move the robot_s to initial pose
        :return: null
        author: weiwei
        date: 20161211osaka
        """
        return self.go_given_conf(jnt_values=np.zeros(self.n_dof))

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

    def ik(self,
           tgt_pos: np.ndarray,
           tgt_rotmat: np.ndarray,
           seed_jnt_values=None,
           toggle_dbg=False):
        """
        trac ik solver runs num_ik and opt_ik in parallel, and return the faster result
        :param tgt_pos: 1x3 nparray, single value or list
        :param tgt_rotmat: 3x3 nparray, single value or list
        :param seed_jnt_values: the starting configuration used in the numerical iteration
        :return:
        """
        if self._ik_solver is None:
            raise Exception("IK solver undefined. Use JLChain.finalize to define it.")
        jnt_values = self._ik_solver.ik(tgt_pos=tgt_pos,
                                        tgt_rotmat=tgt_rotmat,
                                        seed_jnt_values=seed_jnt_values,
                                        toggle_dbg=toggle_dbg)
        return jnt_values

    def copy(self):
        return copy.deepcopy(self)


if __name__ == "__main__":
    import time
    import pickle
    from tqdm import tqdm
    import visualization.panda.world as wd
    import robot_sim._kinematics.model_generator as rkmg
    import robot_sim._kinematics.constant as rkc
    import modeling.geometric_model as mgm

    base = wd.World(cam_pos=[1.25, .75, .75], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)

    jlc = JLChain(n_dof=6)
    jlc.anchor.loc_flange_pos = np.array([0, .1, .1])
    jlc.anchor.loc_flange_rotmat = rm.rotmat_from_axangle(np.array([1, 0, 0]), np.pi / 4)
    jlc.jnts[0].loc_pos = np.array([0, 0, 0])
    jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[0].motion_range = np.array([-np.pi / 2, np.pi / 2])
    # jlc.jnts[1].change_type(rkc.JntType.PRISMATIC)
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
    jlc.loc_tcp_pos = np.array([0, 0, .01])
    jlc.loc_flange_pos = np.array([0.1, 0.1, 0.1])
    # jlc.finalize(ik_solver=None)
    jlc.finalize(ik_solver='d')
    # rkmg.gen_jlc_stick(jlc, stick_rgba=bc.navy_blue, toggle_jnt_frames=True, toggle_tcp_frame=True).attach_to(base)
    # base.run()
    seed_jnt_values = jlc.get_jnt_values()

    success = 0
    num_win = 0
    opt_win = 0
    time_list = []
    tgt_list = []
    for i in tqdm(range(100), desc="ik"):
        random_jnts = jlc.rand_conf()
        tgt_pos, tgt_rotmat = jlc.fk(jnt_values=random_jnts, update=False, toggle_jacobian=False)
        tic = time.time()
        joint_values_with_dbg_info = jlc.ik(tgt_pos=tgt_pos,
                                            tgt_rotmat=tgt_rotmat,
                                            toggle_dbg=False)
        toc = time.time()
        time_list.append(toc - tic)
        print(time_list[-1])
        if joint_values_with_dbg_info is not None:
            success += 1
            if joint_values_with_dbg_info[0] == 'o':
                opt_win += 1
            elif joint_values_with_dbg_info[0] == 'n':
                num_win += 1
            mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
            jlc.fk(jnt_values=joint_values_with_dbg_info, update=True, toggle_jacobian=False)
            rkmg.gen_jlc_stick(jlc, stick_rgba=bc.navy_blue, toggle_tcp_frame=True,
                               toggle_jnt_frames=True, toggle_flange_frame=True).attach_to(base)
            base.run()
        else:
            tgt_list.append((tgt_pos, tgt_rotmat))
    print(f'success: {success}')
    print(f'num_win: {num_win}, opt_win: {opt_win}')
    print('average', np.mean(time_list))
    print('max', np.max(time_list))
    print('min', np.min(time_list))
    base.run()
