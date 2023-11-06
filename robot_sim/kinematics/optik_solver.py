import numpy as np
import basis.robot_math as rm
import robot_sim.kinematics.constant as rkc
import robot_sim.kinematics.jl as lib_jl
import warnings as wns
import scipy.optimize as sopt


class OptIKSolver(object):

    def __init__(self, jlc):
        self.jlc = jlc
        self.max_link_length = self._get_max_link_length()
        self.clamp_pos_err = 2 * self.max_link_length
        self.clamp_rot_err = np.pi / 3

    def _get_max_link_length(self):
        max_len = 0
        for i in range(1, self.jlc.n_dof):
            if self.jlc.jnts[i].type == rkc.JointType.REVOLUTE:
                tmp_vec = self.jlc.jnts[i].gl_pos_q - self.jlc.jnts[i - 1].gl_pos_q
                tmp_len = np.linalg.norm(tmp_vec)
                if tmp_len > max_len:
                    max_len = tmp_len
        return max_len

    def _clamp_tcp_err(self, tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec):
        clamped_tcp_vec = np.copy(tcp_err_vec)
        if tcp_pos_err_val >= self.clamp_pos_err:
            clamped_tcp_vec[:3] = self.clamp_pos_err * tcp_err_vec[:3] / tcp_pos_err_val
        if tcp_rot_err_val >= self.clamp_rot_err:
            clamped_tcp_vec[3:6] = self.clamp_rot_err * tcp_err_vec[3:6] / tcp_rot_err_val
        return clamped_tcp_vec

    def sqpss(self,
              tgt_pos,
              tgt_rotmat,
              seed_jnt_vals=None,
              max_n_iter=1000,
              toggle_debug=False):  # ss = sum of square
        """
        sqpss is faster than sqp
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_vals:
        :param max_n_iter:
        :param toggle_debug:
        :return:
        author: weiwei
        date: 20231101
        """

        def _objective(x, tgt_pos, tgt_rotmat):
            tcp_gl_pos, tcp_gl_rotmat, j_mat = self.jlc.forward_kinematics(jnt_vals=x,
                                                                           toggle_jac=True,
                                                                           update=False)
            tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec = rm.diff_between_posrot(src_pos=tcp_gl_pos,
                                                                                   src_rotmat=tcp_gl_rotmat,
                                                                                   tgt_pos=tgt_pos,
                                                                                   tgt_rotmat=tgt_rotmat)

            # clamped_tcp_err_vec = self._clamp_tcp_err(tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec)
            # delta_jnt_values = np.linalg.pinv(j_mat, rcond=1e-4) @ tcp_err_vec * 1e-3
            return tcp_err_vec.T @ tcp_err_vec, np.diag(np.linalg.pinv(j_mat, rcond=1e-4))

        options = {'ftol': 1e-8,
                   'eps': 1e-8,
                   'maxiter': max_n_iter,
                   'disp': toggle_debug}
        result = sopt.minimize(fun=_objective,
                               args=(tgt_pos, tgt_rotmat),
                               x0=seed_jnt_vals,
                               method='SLSQP',
                               jac=True,
                               bounds=self.jlc.jnt_rngs,
                               options=options)
        print(result)
        # input("Press Enter to continue...")
        if result.success and result.fun < 1e-4:
            return result.x
        else:
            return None

    def sqp(self,
            tgt_pos,
            tgt_rotmat,
            seed_jnt_vals=None,
            max_n_iter=100,
            toggle_debug=False):
        def _objective(x):
            q_diff = seed_jnt_vals - x
            return q_diff.dot(q_diff)

        def _con_tcp(x):
            tcp_gl_pos, tcp_gl_rotmat = self.jlc.forward_kinematics(jnt_vals=x,
                                                                    toggle_jac=False,
                                                                    update=False)
            tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec = rm.diff_between_posrot(src_pos=tcp_gl_pos,
                                                                                   src_rotmat=tcp_gl_rotmat,
                                                                                   tgt_pos=tgt_pos,
                                                                                   tgt_rotmat=tgt_rotmat)
            return 1e-6 - tcp_err_vec.dot(tcp_err_vec)

        constraints = {'type': 'ineq',
                       'fun': _con_tcp}
        options = {'ftol': 1e-6,
                   'eps': 1e-12,
                   'maxiter': max_n_iter,
                   'disp': toggle_debug}
        result = sopt.minimize(fun=_objective,
                               x0=seed_jnt_vals,
                               method='SLSQP',
                               bounds=self.jlc.jnt_rngs,
                               constraints=constraints,
                               options=options)
        print(result)
        return result.x
