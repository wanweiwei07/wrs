import numpy as np
import scipy
import basis.robot_math as rm
import robot_sim._kinematics.constant as rkc
import robot_sim._kinematics.jl as lib_jl
import warnings as wns


class NumIKSolver(object):
    """
    **_RR methods were implemented with random restart
    CWLN is from: Clamping Weighted Least-Norm Method for the Manipulator Kinematic Control with Constraints
    PINV_WC is developed by Weiwei and is the most recommended method
    Notes 20231101:
    1. Sovlers with random restart never worked within 10000 iterations.
    2. PINV_WC (pseudo inverse with weighted clamping) is inspired by CWLN (clamping weighted least norm)
    3. CWLN is 1ms slower than PINV_WC as it needs to solve both svd and damped least squares
    """

    def __init__(self, jlc, wln_ratio=.05):
        self.jlc = jlc
        self.max_link_length = self._get_max_link_length()
        self.clamp_pos_err = 2 * self.max_link_length
        self.clamp_rot_err = np.pi / 3
        self.jnt_wt_ratio = wln_ratio
        # maximum reach
        self.max_rng = 10.0
        # # extract min max for quick access
        self.min_jnt_values = self.jlc.jnt_ranges[:, 0]
        self.max_jnt_values = self.jlc.jnt_ranges[:, 1]
        self.jnt_range_values = self.max_jnt_values - self.min_jnt_values
        self.min_jnt_value_thresholds = self.min_jnt_values + self.jnt_range_values * self.jnt_wt_ratio
        self.max_jnt_value_thresholds = self.max_jnt_values - self.jnt_range_values * self.jnt_wt_ratio

    def _get_max_link_length(self):
        max_len = 0
        for i in range(1, self.jlc.n_dof):
            if self.jlc.jnts[i].type == rkc.JntType.REVOLUTE:
                tmp_vec = self.jlc.jnts[i].gl_pos_q - self.jlc.jnts[i - 1].gl_pos_q
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
        jnt_wt = np.ones(self.jlc.n_dof)
        # min damping interval
        selection = jnt_values < self.min_jnt_value_thresholds
        normalized_diff = (jnt_values - self.min_jnt_values) / (self.min_jnt_value_thresholds - self.min_jnt_values)
        jnt_wt[selection] = -2 * np.power(normalized_diff[selection], 3) + 3 * np.power(normalized_diff[selection], 2)
        # max damping interval
        selection = jnt_values > self.max_jnt_value_thresholds
        normalized_diff = (self.max_jnt_values - jnt_values) / (self.max_jnt_values - self.max_jnt_value_thresholds)
        jnt_wt[selection] = -2 * np.power(normalized_diff[selection], 3) + 3 * np.power(normalized_diff[selection], 2)
        jnt_wt[jnt_values >= self.max_jnt_values] = 0
        jnt_wt[jnt_values <= self.min_jnt_values] = 0
        return np.diag(jnt_wt), np.diag(np.sqrt(jnt_wt))

    def _clamp_tgt_err(self, f2t_pos_err, f2t_rot_err, f2t_err_vec):
        clamped_vec = np.copy(f2t_err_vec)
        if f2t_pos_err >= self.clamp_pos_err:
            clamped_vec[:3] = self.clamp_pos_err * f2t_err_vec[:3] / f2t_pos_err
        if f2t_rot_err >= self.clamp_rot_err:
            clamped_vec[3:6] = self.clamp_rot_err * f2t_err_vec[3:6] / f2t_rot_err
        return clamped_vec

    def are_jnts_in_range(self, jnt_values):
        if np.any(jnt_values < self.min_jnt_values):
            return False
        if np.any(jnt_values > self.max_jnt_values):
            return False
        return True

    def pinv_rr(self,
                tgt_pos,
                tgt_rotmat,
                seed_jnt_vals=None,
                max_n_iter=100,
                toggle_dbg=False):
        iter_jnt_values = seed_jnt_vals
        if seed_jnt_vals is None:
            iter_jnt_values = self.jlc.get_jnt_values()
        counter = 0
        while True:
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=iter_jnt_values,
                                                           toggle_jacobian=True,
                                                           update=False)
            f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_posrot(src_pos=flange_pos,
                                                                           src_rotmat=flange_rotmat,
                                                                           tgt_pos=tgt_pos,
                                                                           tgt_rotmat=tgt_rotmat)
            if f2t_pos_err < 1e-4 and f2t_rot_err < 1e-3 and self.jlc.are_jnts_in_ranges(iter_jnt_values):
                return iter_jnt_values
            clamped_tgt_err = self._clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec)
            delta_jnt_vals = np.linalg.pinv(j_mat, rcond=1e-4) @ clamped_tgt_err
            if abs(np.sum(delta_jnt_vals)) < 1e-8:
                # local minimia
                pass
            iter_jnt_values = iter_jnt_values + delta_jnt_vals
            if not self.are_jnts_in_range(iter_jnt_values):
                # random restart
                iter_jnt_values = self.jlc.rand_conf()
            if toggle_dbg:
                import robot_sim._kinematics.model_generator as rkmg
                jnt_values = self.jlc.get_jnt_values()
                self.jlc.goto_given_conf(jnt_values=iter_jnt_values)
                rkmg.gen_jlc_stick(self.jlc, toggle_flange_frame=True).attach_to(base)
                self.jlc.goto_given_conf(jnt_values=jnt_values)
                import modeling.geometric_model as gm
                gm.gen_arrow(spos=flange_pos, epos=flange_pos + f2t_err_vec[:3] * .1).attach_to(base)
                print("f2t_pos_err ", f2t_pos_err, " f2t_rot_err ", f2t_rot_err)
            if counter > max_n_iter:
                return None
            counter += 1

    def dls_rr(self,
               tgt_pos,
               tgt_rotmat,
               seed_jnt_vals=None,
               max_n_iter=100,
               toggle_dbg=False):
        iter_jnt_values = seed_jnt_vals
        if seed_jnt_vals is None:
            iter_jnt_values = self.jlc.get_jnt_values()
        counter = 0
        while True:
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=iter_jnt_values,
                                                           toggle_jacobian=True,
                                                           update=False)
            f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_posrot(src_pos=flange_pos,
                                                                           src_rotmat=flange_rotmat,
                                                                           tgt_pos=tgt_pos,
                                                                           tgt_rotmat=tgt_rotmat)
            if f2t_pos_err < 1e-4 and f2t_rot_err < 1e-3 and self.jlc.are_jnts_in_ranges(iter_jnt_values):
                return iter_jnt_values
            clamped_err_vec = self._clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec)
            delta_jnt_vals = (np.linalg.inv(j_mat.T @ j_mat + 1e-4 * np.eye(j_mat.shape[1])) @
                              j_mat.T @ clamped_err_vec)
            iter_jnt_values = iter_jnt_values + delta_jnt_vals
            if not self.are_jnts_in_range(iter_jnt_values):
                # random restart
                iter_jnt_values = self.jlc.rand_conf()
            if toggle_dbg:
                import robot_sim._kinematics.model_generator as rkmg
                jnt_values = self.jlc.get_jnt_values()
                self.jlc.goto_given_conf(jnt_values=iter_jnt_values)
                rkmg.gen_jlc_stick(self.jlc, toggle_flange_frame=True).attach_to(base)
                self.jlc.goto_given_conf(jnt_values=jnt_values)
                import modeling.geometric_model as gm
                gm.gen_arrow(spos=flange_pos, epos=tgt_pos).attach_to(base)
                print("f2t_pos_err ", f2t_pos_err, " f2t_rot_err ", f2t_rot_err)
            if counter > max_n_iter:
                return None
            counter += 1

    def jt_rr(self,
              tgt_pos,
              tgt_rotmat,
              seed_jnt_vals=None,
              max_n_iter=100,
              toggle_dbg=False):
        """
        the jacobian transpose method
        paper: Buss, Introduction to Inverse Kinematics with Jacobian Transpose,
        Pseudoinverse and Damped Least Squares methods
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_vals:
        :param max_n_iter:
        :param toggle_dbg:
        :return:
        """
        iter_jnt_vals = seed_jnt_vals
        if seed_jnt_vals is None:
            iter_jnt_vals = self.jlc.get_jnt_values()
        counter = 0
        while True:
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=iter_jnt_vals,
                                                           toggle_jacobian=True,
                                                           update=False)
            f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_posrot(src_pos=flange_pos,
                                                                           src_rotmat=flange_rotmat,
                                                                           tgt_pos=tgt_pos,
                                                                           tgt_rotmat=tgt_rotmat)
            if f2t_pos_err < 1e-4 and f2t_rot_err < 1e-3:
                return iter_jnt_vals
            jjt = j_mat @ j_mat.T
            jjt_dot_e = jjt @ f2t_err_vec
            weight = np.dot(f2t_err_vec, jjt_dot_e) / np.dot(jjt_dot_e, jjt_dot_e)
            delta_jnt_vals = weight * (j_mat.T @ f2t_err_vec)
            iter_jnt_vals = iter_jnt_vals + delta_jnt_vals
            if not self.are_jnts_in_range(iter_jnt_vals):
                # random restart
                iter_jnt_vals = self.jlc.rand_conf()
            if toggle_dbg:
                import robot_sim._kinematics.model_generator as rkmg
                jnt_values = self.jlc.get_jnt_values()
                self.jlc.goto_given_conf(jnt_values=iter_jnt_vals)
                rkmg.gen_jlc_stick(self.jlc, toggle_flange_frame=True).attach_to(base)
                self.jlc.goto_given_conf(jnt_values=jnt_values)
                import modeling.geometric_model as gm
                gm.gen_arrow(spos=flange_pos, epos=tgt_pos).attach_to(base)
                print("f2t_pos_err ", f2t_pos_err, "f2t_rot_err ", f2t_rot_err)
            if counter > max_n_iter:
                return None
            counter += 1

    def pinv_cw(self,
                tgt_pos,
                tgt_rotmat,
                seed_jnt_vals=None,
                max_n_iter=100,
                toggle_dbg=False):
        """
        improved cwln method (replaced the least damping in cwln with moore-penrose inverse)
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_vals:
        :param max_n_iter:
        :param toggle_dbg:
        :return:
        author: weiwei
        date: 20231101
        """
        iter_jnt_values = seed_jnt_vals
        if seed_jnt_vals is None:
            iter_jnt_values = self.jlc.get_jnt_values()
        counter = 0
        while True:
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=iter_jnt_values,
                                                           toggle_jacobian=True,
                                                           update=False)
            f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_posrot(src_pos=flange_pos,
                                                                           src_rotmat=flange_rotmat,
                                                                           tgt_pos=tgt_pos,
                                                                           tgt_rotmat=tgt_rotmat)
            if f2t_pos_err < 1e-4 and f2t_rot_err < 1e-3 and self.jlc.are_jnts_in_ranges(iter_jnt_values):
                return iter_jnt_values
            clamped_err_vec = self._clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec)
            wln, wln_sqrt = self._jnt_wt_mat(iter_jnt_values)
            # weighted clamping
            k_phi = 0.1
            tmp_mm_jnt_values = self.max_jnt_values + self.min_jnt_values
            phi_q = ((2 * iter_jnt_values - tmp_mm_jnt_values) / self.jnt_range_values) * k_phi
            clamping = -(np.identity(wln.shape[0]) - wln) @ phi_q
            # pinv with weighted clamping
            delta_jnt_vals = clamping + wln_sqrt @ np.linalg.pinv(j_mat @ wln_sqrt, rcond=1e-4) @ (
                    clamped_err_vec - j_mat @ clamping)
            iter_jnt_values = iter_jnt_values + delta_jnt_vals
            if toggle_dbg:
                import robot_sim._kinematics.model_generator as rkmg
                jnt_values = self.jlc.get_jnt_values()
                self.jlc.goto_given_conf(jnt_values=iter_jnt_values)
                stick_rgba = rm.bc.lnk_stick_rgba
                stick_rgba[3] = .5
                rkmg.gen_jlc_stick(self.jlc, stick_rgba=stick_rgba, toggle_jnt_frames=True,
                                   toggle_flange_frame=True).attach_to(base)
                self.jlc.goto_given_conf(jnt_values=jnt_values)
                import modeling.geometric_model as gm
                gm.gen_arrow(spos=flange_pos, epos=tgt_pos).attach_to(base)
                print("f2t_pos_err ", f2t_pos_err, " f2t_rot_err ", f2t_rot_err)
            if counter > max_n_iter:
                return None
                # raise Exception("No IK solution")
            counter += 1

    def pinv_gpm(self,
                 tgt_pos,
                 tgt_rotmat,
                 seed_jnt_vals=None,
                 max_n_iter=100,
                 toggle_dbg=False):
        """
        gradient projection method, only applicabled to redundant manipulators, slower than cwln
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_vals:
        :param max_n_iter:
        :param toggle_dbg:
        :return:
        author: weiwei
        date: 20231101
        """
        iter_jnt_values = seed_jnt_vals
        if seed_jnt_vals is None:
            iter_jnt_values = self.jlc.get_jnt_values()
        counter = 0
        while True:
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=iter_jnt_values,
                                                           toggle_jacobian=True,
                                                           update=False)
            f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_posrot(src_pos=flange_pos,
                                                                           src_rotmat=flange_rotmat,
                                                                           tgt_pos=tgt_pos,
                                                                           tgt_rotmat=tgt_rotmat)
            if f2t_pos_err < 1e-4 and f2t_rot_err < 1e-3 and self.jlc.are_jnts_in_ranges(iter_jnt_values):
                return iter_jnt_values
            clamped_err_vec = self._clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec)
            mp_inv = np.linalg.pinv(j_mat)
            null_projector = np.identity(mp_inv.shape[0]) - mp_inv @ j_mat
            jnt_mid_values = (self.max_jnt_values + self.min_jnt_values) / 2
            iter_secondary = (jnt_mid_values - iter_jnt_values) * 0.1
            iter_jnt_values += mp_inv @ clamped_err_vec + null_projector @ iter_secondary
            if toggle_dbg:
                import modeling.geometric_model as mgm
                import robot_sim._kinematics.model_generator as rkmg
                mgm.gen_arrow(spos=flange_pos, epos=flange_pos + clamped_err_vec[:3]).attach_to(base)
                self.jlc.goto_given_conf(jnt_values=iter_jnt_values)
                stick_rgba = rm.bc.lnk_stick_rgba
                stick_rgba[3] = .5
                rkmg.gen_jlc_stick(self.jlc, stick_rgba=stick_rgba, toggle_jnt_frames=True,
                                   toggle_flange_frame=True).attach_to(base)
                print("f2t_pos_err ", f2t_pos_err, ", f2t_rot_err ", f2t_rot_err)
            counter += 1
            if counter >= max_n_iter:
                if toggle_dbg:
                    base.run()
                return None
                # raise Exception("No IK solution")

    def cwln(self,
             tgt_pos,
             tgt_rotmat,
             seed_jnt_vals=None,
             max_n_iter=100,
             toggle_dbg=False):
        # original method from the following paper:
        # paper: Huang, clamping weighted least-norm method for themanipulator kinematic control with constraints
        # does not work on redundant jlcs
        iter_jnt_vals = seed_jnt_vals
        if seed_jnt_vals is None:
            iter_jnt_vals = self.jlc.get_jnt_values()
        counter = 0
        while True:
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=iter_jnt_vals,
                                                           toggle_jacobian=True,
                                                           update=False)
            f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_posrot(src_pos=flange_pos,
                                                                           src_rotmat=flange_rotmat,
                                                                           tgt_pos=tgt_pos,
                                                                           tgt_rotmat=tgt_rotmat)
            if f2t_pos_err < 1e-4 and f2t_rot_err < 1e-3:
                return iter_jnt_vals
            clamped_err_vec = self._clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec)
            wln, wln_sqrt = self._jnt_wt_mat(iter_jnt_vals)
            # weighted clamping
            k_phi = 0.1
            tmp_mm_jnt_values = self.max_jnt_values + self.min_jnt_values
            phi_q = ((2 * iter_jnt_vals - tmp_mm_jnt_values) / self.jnt_range_values) * k_phi
            clamping = -(np.identity(wln.shape[0]) - wln) @ phi_q
            # lambda coefficient
            # min_svd_val = scipy.linalg.svdvals(wln_sqrt)[-1]
            # lam = 1e-9 if min_svd_val < 1e-4 else 0
            lam = 1e-9
            # cwln
            delta_jnt_values = clamping + wln @ j_mat.T @ np.linalg.inv(
                j_mat @ wln @ j_mat.T + lam * np.eye(j_mat.shape[1])) @ (clamped_err_vec - j_mat @ clamping)
            iter_jnt_vals = iter_jnt_vals + delta_jnt_values
            if toggle_dbg:
                import robot_sim._kinematics.model_generator as rkmg
                jnt_values = self.jlc.get_jnt_values()
                self.jlc.goto_given_conf(jnt_values=iter_jnt_vals)
                rkmg.gen_jlc_stick(self.jlc, toggle_flange_frame=True).attach_to(base)
                self.jlc.goto_given_conf(jnt_values=jnt_values)
                import modeling.geometric_model as gm
                gm.gen_arrow(spos=flange_pos, epos=tgt_pos).attach_to(base)
                print("f2t_pos_err ", f2t_pos_err, "f2t_rot_err ", f2t_rot_err)
            if counter > max_n_iter:
                return None
            counter += 1
