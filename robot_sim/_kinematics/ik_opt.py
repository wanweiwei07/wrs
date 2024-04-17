import numpy as np
import basis.robot_math as rm
import robot_sim._kinematics.constant as rkc
import robot_sim._kinematics.jl as lib_jl
import warnings as wns
import scipy.optimize as sopt


class OptIKSolver(object):

    def __init__(self, jlc):
        self.jlc = jlc
        self.max_link_length = self._get_max_link_length()
        self.clamp_pos_err = 2 * self.max_link_length
        self.clamp_rot_err = np.pi / 3

    def __call__(self,
                 tgt_pos,
                 tgt_rotmat,
                 seed_jnt_values=None,
                 max_n_iter=100,
                 toggle_dbg=False):
        return self.sqpss(tgt_pos=tgt_pos,
                          tgt_rotmat=tgt_rotmat,
                          seed_jnt_values=seed_jnt_values,
                          max_n_iter=max_n_iter,
                          toggle_dbg=toggle_dbg)

    def _get_max_link_length(self):
        max_len = 0
        for i in range(1, self.jlc.n_dof):
            if self.jlc.jnts[i].type == rkc.JntType.REVOLUTE:
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
              seed_jnt_values=None,
              max_n_iter=1000,
              toggle_dbg=False):  # ss = sum of square
        """
        sqpss is faster than sqp
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param max_n_iter:
        :param toggle_dbg:
        :return:
        author: weiwei
        date: 20231101
        """

        def _objective(x, tgt_pos, tgt_rotmat):
            tcp_gl_pos, tcp_gl_rotmat, j_mat = self.jlc.fk(jnt_values=x,
                                                           toggle_jacobian=True,
                                                           update=False)
            tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec = rm.diff_between_poses(src_pos=tcp_gl_pos,
                                                                                  src_rotmat=tcp_gl_rotmat,
                                                                                  tgt_pos=tgt_pos,
                                                                                  tgt_rotmat=tgt_rotmat)
            # print("obj ",tcp_err_vec.T @ tcp_err_vec)
            return tcp_err_vec.T @ tcp_err_vec

        iteration_count = [0]

        def _callback(x):
            iteration_count[0] += 1
            stick_rgba = rm.bc.cool_map(iteration_count[0] / max_n_iter)
            result = self.jlc.goto_given_conf(jnt_values=x)
            print("call back ", np.degrees(x))
            print("call back ", result[0])
            self.jlc.gen_stickmodel(stick_rgba=stick_rgba, toggle_flange_frame=True,
                                    toggle_jnt_frames=True).attach_to(base)

        print("seed ", np.degrees(seed_jnt_values))
        options = {'ftol': rm._EPS*10e-12,
                   'eps': rm._EPS*10e-12,
                   'maxiter': max_n_iter,
                   'disp': toggle_dbg}
        if toggle_dbg:
            callback_fn = _callback
        else:
            callback_fn = None
        import time
        for i in range(10):
            tic=time.time()
            result = sopt.minimize(fun=_objective,
                                   args=(tgt_pos, tgt_rotmat),
                                   x0=seed_jnt_values,
                                   method='SLSQP',
                                   bounds=self.jlc.jnt_ranges,
                                   callback=callback_fn,
                                   options=options)
            toc=time.time()
            print(toc-tic)
            base.run()
            # print(result)
            # input("Press Enter to continue...")
            if result.success and result.fun < 1e-4:
                return result.x
            else:
                break
                seed_jnt_values = self.jlc.rand_conf()
                continue
        return None

    def sqp(self,
            tgt_pos,
            tgt_rotmat,
            seed_jnt_values=None,
            max_n_iter=100,
            toggle_dbg=False):
        def _objective(x):
            q_diff = seed_jnt_values - x
            return q_diff.dot(q_diff)

        def _con_tcp(x):
            tcp_gl_pos, tcp_gl_rotmat = self.jlc.fk(jnt_values=x,
                                                    toggle_jacobian=False,
                                                    update=False)
            tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec = rm.diff_between_poses(src_pos=tcp_gl_pos,
                                                                                  src_rotmat=tcp_gl_rotmat,
                                                                                  tgt_pos=tgt_pos,
                                                                                  tgt_rotmat=tgt_rotmat)
            return 1e-6 - tcp_err_vec.dot(tcp_err_vec)

        constraints = {'type': 'ineq',
                       'fun': _con_tcp}
        options = {'ftol': rm._EPS,
                   'eps': rm._EPS,
                   'maxiter': max_n_iter,
                   'disp': toggle_dbg}
        result = sopt.minimize(fun=_objective,
                               x0=seed_jnt_values,
                               method='SLSQP',
                               bounds=self.jlc.jnt_ranges,
                               constraints=constraints,
                               options=options)
        print(result)
        return result.x
