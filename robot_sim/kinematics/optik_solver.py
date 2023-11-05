import numpy as np
import basis.robot_math as rm
import robot_sim.kinematics.constant as rkc
import robot_sim.kinematics.jl as lib_jl
import warnings as wns
import scipy.optimize as sopt


class OptIKSolver(object):

    def __init__(self, jlc):
        self.jlc = jlc

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

        def _objective(x):
            tcp_gl_pos, tcp_gl_rotmat = self.jlc.forward_kinematics(joint_values=x,
                                                                    toggle_jacobian=False,
                                                                    update=False)
            tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec = rm.diff_between_posrot(src_pos=tcp_gl_pos,
                                                                                   src_rotmat=tcp_gl_rotmat,
                                                                                   tgt_pos=tgt_pos,
                                                                                   tgt_rotmat=tgt_rotmat)
            return tcp_err_vec.T @ tcp_err_vec

        options = {'ftol': 1e-6,
                   'eps': 1e-12,
                   'maxiter': max_n_iter,
                   'disp': toggle_debug}
        result = sopt.minimize(fun=_objective,
                               x0=seed_jnt_vals,
                               method='SLSQP',
                               bounds=self.jlc.joint_ranges,
                               options=options)
        print(result)
        return result.x

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
            tcp_gl_pos, tcp_gl_rotmat = self.jlc.forward_kinematics(joint_values=x,
                                                                    toggle_jacobian=False,
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
                               bounds=self.jlc.joint_ranges,
                               constraints=constraints,
                               options=options)
        print(result)
        return result.x
