import numpy as np
from wrs import basis as rm
import scipy.optimize as sopt


def _objective(x, jlc, tgt_pos, tgt_rotmat):
    tcp_gl_pos, tcp_gl_rotmat, j_mat = jlc.fk(jnt_values=x,
                                              toggle_jacobian=True,
                                              update=False)
    f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_poses(src_pos=tcp_gl_pos,
                                                                  src_rotmat=tcp_gl_rotmat,
                                                                  tgt_pos=tgt_pos,
                                                                  tgt_rotmat=tgt_rotmat)
    return (f2t_pos_err - 1e-4) ** 2 + (f2t_rot_err - 1e-3) ** 2


class OptIKSolver(object):

    def __init__(self, jlc):
        self.jlc = jlc
        self.clamp_pos_err = .1
        self.clamp_rot_err = np.pi / 10.0
        self._max_n_iter = 7  # max_n_iter of the backbone solver

    def __call__(self,
                 tgt_pos,
                 tgt_rotmat,
                 seed_jnt_values=None,
                 max_n_iter=None,
                 toggle_dbg=False):
        return self.genetic(tgt_pos=tgt_pos,
                            tgt_rotmat=tgt_rotmat,
                            seed_jnt_values=seed_jnt_values,
                            max_n_iter=max_n_iter,
                            toggle_dbg=toggle_dbg)

    def _clamp_tgt_err(self, f2t_pos_err, f2t_rot_err, f2t_err_vec):
        clamped_vec = np.copy(f2t_err_vec)
        if f2t_pos_err >= self.clamp_pos_err:
            clamped_vec[:3] = self.clamp_pos_err * f2t_err_vec[:3] / f2t_pos_err
        if f2t_rot_err >= self.clamp_rot_err:
            clamped_vec[3:6] = self.clamp_rot_err * f2t_err_vec[3:6] / f2t_rot_err
        return clamped_vec

    def sqpss(self,
              tgt_pos,
              tgt_rotmat,
              seed_jnt_values=None,
              max_n_iter=None,
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

        def _callback(x):
            if not hasattr(_callback, "iteration_count"):
                # Initialize the static variable
                _callback.static_variable = 0
            # Use the static variable
            _callback.iteration_count[0] += 1
            # stick_rgba = rm.bc.cool_map(_callback.iteration_count[0] / max_n_iter)
            result = self.jlc.goto_given_conf(jnt_values=x)
            print("call back ", np.degrees(x))
            print("call back ", result[0])
            self.jlc.gen_stickmodel(toggle_flange_frame=True, toggle_jnt_frames=True).attach_to(base)

        if seed_jnt_values is None:
            seed_jnt_values = self.jlc.rand_conf()
        options = {'maxiter': self._max_n_iter if max_n_iter is None else max_n_iter,
                   'disp': toggle_dbg}
        if toggle_dbg:
            callback_fn = _callback
        else:
            callback_fn = None
        for i in range(10):
            # tic = time.time()
            result = sopt.minimize(fun=_objective,
                                   args=(self.jlc, tgt_pos, tgt_rotmat),
                                   x0=seed_jnt_values,
                                   method='BFGS',
                                   bounds=self.jlc.jnt_ranges,
                                   jac=True,  # fun will be (obj, jac)
                                   callback=callback_fn,
                                   options=options)
            # toc = time.time()
            # print(toc - tic)
            print(result)
            input("Press Enter to continue...")
            if result.success and result.fun < 1e-4:
                return result.x
            else:
                break
                # seed_jnt_values = self.jlc.rand_conf()
                # continue
        return None

    def genetic(self,
                tgt_pos,
                tgt_rotmat,
                seed_jnt_values=None,
                max_n_iter=100,
                toggle_dbg=False):

        def _callback(x):
            if not hasattr(_callback, "iteration_count"):
                # Initialize the static variable
                _callback.static_variable = 0
            # Use the static variable
            _callback.iteration_count[0] += 1
            # stick_rgba = rm.bc.cool_map(_callback.iteration_count[0] / max_n_iter)
            result = self.jlc.goto_given_conf(jnt_values=x)
            print("call back ", np.degrees(x))
            print("call back ", result[0])
            self.jlc.gen_stickmodel(toggle_flange_frame=True, toggle_jnt_frames=True).attach_to(base)

        if toggle_dbg:
            callback_fn = _callback
        else:
            callback_fn = None
        result = sopt.differential_evolution(func=_objective,
                                             bounds=self.jlc.jnt_ranges,
                                             args=(self.jlc, tgt_pos, tgt_rotmat),
                                             callback=callback_fn,
                                             atol=1e-4,
                                             workers=-1)
        print(result)
        input("Press Enter to continue...")
        if result.success and result.fun < 1e-4:
            return result.x
        else:
            return None
