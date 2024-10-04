"""
This trac ik solver is implemented following instructions from the trac ik paper.
P. Beeson and B. Ames, "TRAC-IK: An open-source library for improved solving of generic inverse _kinematics,"
IEEE-RAS International Conference on Humanoid Robots (Humanoids), Seoul, Korea (South), 2015, pp. 928-935,
doi: 10.1109/HUMANOIDS.2015.7363472.

Key differences: KDL_RR is implemented as PINV_CW
Known issues: The PINV_CW solver has a much lower success rate. Random restart does not improve performance.

author: weiwei
date: 20231107
"""

import numpy as np
import multiprocessing as mp
from wrs import basis as rm, robot_sim as rkc
import scipy.optimize as sopt


class NumIKSolverProc(mp.Process):

    def __init__(self,
                 anchor,
                 joints,
                 tcp_joint_id,
                 tcp_loc_homomat,
                 wln_ratio,
                 param_queue,
                 state_queue,
                 result_queue):
        super(NumIKSolverProc, self).__init__()
        self._param_queue = param_queue
        self._state_queue = state_queue
        self._result_queue = result_queue
        # nik related preparation
        self.n_dof = len(joints)
        self.anchor = anchor
        self.joints = joints
        self.tcp_joint_id = tcp_joint_id
        self.tcp_loc_homomat = tcp_loc_homomat
        self.max_link_length = self._get_max_link_length()
        self.clamp_pos_err = 2 * self.max_link_length
        self.clamp_rot_err = np.pi / 3
        self.jnt_wt_ratio = wln_ratio
        # maximum reach
        self.max_rng = 10.0
        # # extract min max for quick access
        jnt_limits = []
        for i in range(len(joints)):
            jnt_limits.append(joints[i].motion_range)
        self.joint_ranges = np.asarray(jnt_limits)
        self.min_jnt_vals = self.joint_ranges[:, 0]
        self.max_jnt_vals = self.joint_ranges[:, 1]
        self.jnt_rngs = self.max_jnt_vals - self.min_jnt_vals
        self.jnt_rngs_mid = (self.max_jnt_vals + self.min_jnt_vals) / 2
        self.min_jnt_threshold = self.min_jnt_vals + self.jnt_rngs * self.jnt_wt_ratio
        self.max_jnt_threshold = self.max_jnt_vals - self.jnt_rngs * self.jnt_wt_ratio

    def _get_max_link_length(self):
        max_len = 0
        for i in range(1, self.n_dof):
            if self.joints[i].type == rkc.JntType.REVOLUTE:
                tmp_vec = self.joints[i].gl_pos_q - self.joints[i - 1].gl_pos_q
                tmp_len = np.linalg.norm(tmp_vec)
                if tmp_len > max_len:
                    max_len = tmp_len
        return max_len

    def _jnt_wt_mat(self, jnt_values):
        """
        get the joint weight mat
        :param jnt_values:
        :return: W, W^(1/2)
        author: weiwei
        date: 20201126
        """
        jnt_wt = np.ones(self.n_dof)
        # min damping interval
        selection = jnt_values < self.min_jnt_threshold
        normalized_diff = ((jnt_values - self.min_jnt_vals) / (self.min_jnt_threshold - self.min_jnt_vals))[selection]
        jnt_wt[selection] = -2 * np.power(normalized_diff, 3) + 3 * np.power(normalized_diff, 2)
        # max damping interval
        selection = jnt_values > self.max_jnt_threshold
        normalized_diff = ((self.max_jnt_vals - jnt_values) / (self.max_jnt_vals - self.max_jnt_threshold))[selection]
        jnt_wt[selection] = -2 * np.power(normalized_diff, 3) + 3 * np.power(normalized_diff, 2)
        jnt_wt[jnt_values >= self.max_jnt_vals] = 0
        jnt_wt[jnt_values <= self.min_jnt_vals] = 0
        return np.diag(jnt_wt), np.diag(np.sqrt(jnt_wt))

    def _clamp_tcp_err(self, tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec):
        clamped_tcp_vec = np.copy(tcp_err_vec)
        if tcp_pos_err_val >= self.clamp_pos_err:
            clamped_tcp_vec[:3] = self.clamp_pos_err * tcp_err_vec[:3] / tcp_pos_err_val
        if tcp_rot_err_val >= self.clamp_rot_err:
            clamped_tcp_vec[3:6] = self.clamp_rot_err * tcp_err_vec[3:6] / tcp_rot_err_val
        return clamped_tcp_vec

    def run(self):
        def _fk(anchor, joints, tcp_joint_id, tcp_loc_homomat, joint_values, toggle_jacobian):
            """
            joints = jlc.joints
            author: weiwei
            date: 20231105
            """
            n_dof = len(joints)
            homomat = anchor.homomat
            j_pos = np.zeros((n_dof, 3))
            j_axis = np.zeros((n_dof, 3))
            for i in range(tcp_joint_id + 1):
                j_pos[i, :] = homomat[:3, 3] + homomat[:3, :3] @ joints[i].loc_pos
                homomat = homomat @ joints[i].get_motion_homomat(motion_value=joint_values[i])
                j_axis[i, :] = homomat[:3, :3] @ joints[i].loc_motion_ax
            tcp_gl_homomat = homomat @ tcp_loc_homomat
            tcp_gl_pos = tcp_gl_homomat[:3, 3]
            tcp_gl_rotmat = tcp_gl_homomat[:3, :3]
            if toggle_jacobian:
                j_mat = np.zeros((6, n_dof))
                for i in range(tcp_joint_id + 1):
                    if joints[i].type == rkc.JntType.REVOLUTE:
                        vec_jnt2tcp = tcp_gl_pos - j_pos[i, :]
                        j_mat[:3, i] = np.cross(j_axis[i, :], vec_jnt2tcp)
                        j_mat[3:6, i] = j_axis[i, :]
                    if joints[i].type == rkc.JntType.PRISMATIC:
                        j_mat[:3, i] = j_axis[i, :]
                return tcp_gl_pos, tcp_gl_rotmat, j_mat
            else:
                return tcp_gl_pos, tcp_gl_rotmat

        while True:
            tgt_pos, tgt_rotmat, seed_jnt_values, max_n_iter = self._param_queue.get()
            iter_jnt_vals = seed_jnt_values.copy()
            counter = 0
            while self._result_queue.empty():  # check if other solver succeeded in the beginning
                tcp_gl_pos, tcp_gl_rotmat, j_mat = _fk(self.anchor,
                                                       self.joints,
                                                       self.tcp_joint_id,
                                                       self.tcp_loc_homomat,
                                                       joint_values=iter_jnt_vals,
                                                       toggle_jacobian=True)
                tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec = rm.diff_between_poses(src_pos=tcp_gl_pos,
                                                                                      src_rotmat=tcp_gl_rotmat,
                                                                                      tgt_pos=tgt_pos,
                                                                                      tgt_rotmat=tgt_rotmat)
                if tcp_pos_err_val < 1e-4 and tcp_rot_err_val < 1e-3:
                    self._result_queue.put(('n', iter_jnt_vals))
                    break
                clamped_err_vec = self._clamp_tcp_err(tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec)
                wln, wln_sqrt = self._jnt_wt_mat(iter_jnt_vals)
                # weighted clamping
                k_phi = 0.1
                phi_q = ((2 * iter_jnt_vals - self.jnt_rngs_mid) / self.jnt_rngs) * k_phi
                clamping = -(np.identity(wln.shape[0]) - wln) @ phi_q
                # pinv with weighted clamping
                delta_jnt_values = clamping + wln_sqrt @ np.linalg.pinv(j_mat @ wln_sqrt, rcond=1e-4) @ (
                        clamped_err_vec - j_mat @ clamping)
                iter_jnt_vals = iter_jnt_vals + delta_jnt_values
                if counter > max_n_iter:
                    # optik failed
                    break
                counter += 1
            self._state_queue.put(1)


class OptIKSolverProc(mp.Process):
    def __init__(self,
                 anchor,
                 joints,
                 tcp_joint_id,
                 tcp_loc_homomat,
                 param_queue,
                 state_queue,
                 result_queue):
        super(OptIKSolverProc, self).__init__()
        self._param_queue = param_queue
        self._result_queue = result_queue
        self._state_queue = state_queue
        self.anchor = anchor
        self.joints = joints
        jnt_limits = []
        for i in range(len(joints)):
            jnt_limits.append(joints[i].motion_range)
        self.joint_ranges = np.asarray(jnt_limits)
        self.tcp_joint_id = tcp_joint_id
        self.tcp_loc_homomat = tcp_loc_homomat

    def _rand_conf(self):
        """
        generate a random configuration
        author: weiwei
        date: 20200326
        """
        return np.multiply(np.random.rand(len(self.joints)),
                           (self.joint_ranges[:, 1] - self.joint_ranges[:, 0])) + self.joint_ranges[:, 0]

    def run(self):  # OptIKSolver.sqpss
        """
        sqpss is faster than sqp
        :return:
        author: weiwei
        date: 20231101
        """

        def _fk(anchor, joints, tcp_joint_id, tcp_loc_homomat, joint_values, toggle_jacobian):
            """
            joints = jlc.joints
            author: weiwei
            date: 20231105
            """
            n_dof = len(joints)
            homomat = anchor.homomat
            j_pos = np.zeros((n_dof, 3))
            j_axis = np.zeros((n_dof, 3))
            for i in range(tcp_joint_id + 1):
                j_pos[i, :] = homomat[:3, 3] + homomat[:3, :3] @ joints[i].loc_pos
                homomat = homomat @ joints[i].get_motion_homomat(motion_value=joint_values[i])
                j_axis[i, :] = homomat[:3, :3] @ joints[i].loc_motion_ax
            tcp_gl_homomat = homomat @ tcp_loc_homomat
            tcp_gl_pos = tcp_gl_homomat[:3, 3]
            tcp_gl_rotmat = tcp_gl_homomat[:3, :3]
            if toggle_jacobian:
                j_mat = np.zeros((6, n_dof))
                for i in range(tcp_joint_id + 1):
                    if joints[i].type == rkc.JntType.REVOLUTE:
                        vec_jnt2tcp = tcp_gl_pos - j_pos[i, :]
                        j_mat[:3, i] = np.cross(j_axis[i, :], vec_jnt2tcp)
                        j_mat[3:6, i] = j_axis[i, :]
                    if joints[i].type == rkc.JntType.PRISMATIC:
                        j_mat[:3, i] = j_axis[i, :]
                return tcp_gl_pos, tcp_gl_rotmat, j_mat
            else:
                return tcp_gl_pos, tcp_gl_rotmat

        def _objective(x, tgt_pos, tgt_rotmat):
            tcp_gl_pos, tcp_gl_rotmat = _fk(self.anchor,
                                            self.joints,
                                            self.tcp_joint_id,
                                            self.tcp_loc_homomat,
                                            joint_values=x,
                                            toggle_jacobian=False)
            tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec = rm.diff_between_poses(src_pos=tcp_gl_pos,
                                                                                  src_rotmat=tcp_gl_rotmat,
                                                                                  tgt_pos=tgt_pos,
                                                                                  tgt_rotmat=tgt_rotmat)
            return tcp_err_vec.dot(tcp_err_vec)

        def _call_back(x):
            """
            check if other solvers succeeded at the end of each iteration
            :param x:
            :return:
            """
            if not self._result_queue.empty():
                self._state_queue.put(1)
                raise StopIteration

        # sqpss with random restart
        while True:
            tgt_pos, tgt_rotmat, seed_jnt_values, max_n_iter = self._param_queue.get()
            options = {'maxiter': max_n_iter}
            counter = 0
            while True:
                counter += 1
                try:
                    result = sopt.minimize(fun=_objective,
                                           args=(tgt_pos, tgt_rotmat),
                                           x0=seed_jnt_values,
                                           method='SLSQP',
                                           bounds=self.joint_ranges,
                                           options=options,
                                           callback=_call_back)
                except StopIteration:
                    break  # other solver succeeded
                if self._result_queue.empty():
                    if result.success and result.fun < 1e-4:
                        self._result_queue.put(('o', result.x))
                        break
                    else:
                        if counter > 10:
                            self._result_queue.put(None)
                            break
                        else:
                            seed_jnt_values = self._rand_conf()
                            continue
                break
            self._state_queue.put(1)


class TracIKSolver(object):
    """
    author: weiwei
    date: 20231102
    """

    def __init__(self, jlc, wln_ratio=.05):
        self.jlc = jlc
        self._default_seed_jnt_values = self.jlc.get_jnt_values()
        self._nik_param_queue = mp.Queue()
        self._oik_param_queue = mp.Queue()
        self._nik_state_queue = mp.Queue()
        self._oik_state_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self.nik_solver_proc = NumIKSolverProc(self.jlc.anchor,
                                               self.jlc.jnts,
                                               self.jlc.flange_jnt_id,
                                               self.jlc.loc_flange_homomat,
                                               wln_ratio,
                                               self._nik_param_queue,
                                               self._nik_state_queue,
                                               self._result_queue)
        self.oik_solver_proc = OptIKSolverProc(self.jlc.anchor,
                                               self.jlc.jnts,
                                               self.jlc.flange_jnt_id,
                                               self.jlc.loc_flange_homomat,
                                               self._oik_param_queue,
                                               self._oik_state_queue,
                                               self._result_queue)
        self.nik_solver_proc.start()
        self.oik_solver_proc.start()
        self._tcp_gl_pos, self._tcp_gl_rotmat = self.jlc.get_gl_tcp()
        # run once to avoid long waiting time in the beginning
        self._oik_param_queue.put((self._tcp_gl_pos, self._tcp_gl_rotmat, self._default_seed_jnt_values, 10))
        self._oik_state_queue.get()
        self._result_queue.get()

    def __call__(self,
                 tgt_pos,
                 tgt_rotmat,
                 seed_jnt_values=None,
                 max_n_iter=100,
                 toggle_dbg=False):
        return self.ik(tgt_pos=tgt_pos,
                       tgt_rotmat=tgt_rotmat,
                       seed_jnt_values=seed_jnt_values,
                       max_n_iter=max_n_iter,
                       toggle_dbg=toggle_dbg)

    def ik(self,
           tgt_pos,
           tgt_rotmat,
           seed_jnt_values=None,
           max_n_iter=100,
           toggle_dbg=False):
        """
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param max_n_iter:
        :param toggle_dbg: the function will return a tuple like (solver, jnt_values); solver is 'o' (opt) or 'n' (num)
        :return:
        author: weiwei
        date: 20231107
        """
        if seed_jnt_values is None:
            seed_jnt_values = self._default_seed_jnt_values
        self._nik_param_queue.put((tgt_pos, tgt_rotmat, seed_jnt_values, max_n_iter))
        self._oik_param_queue.put((tgt_pos, tgt_rotmat, seed_jnt_values, max_n_iter))
        if self._nik_state_queue.get() and self._oik_state_queue.get():
            result = self._result_queue.get()
            if toggle_dbg:
                return result
            else:
                if result is None:
                    return None
                else:
                    return result[1]
