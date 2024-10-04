import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim._kinematics.constant as rkc
import wrs.robot_sim._kinematics.model_generator as rkmg
import wrs.modeling.geometric_model as mgm


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

    def __init__(self, jlc, wln_ratio=.1):
        self.jlc = jlc
        self.max_link_length = self._get_max_link_length()
        # self.clamp_pos_err = self.max_link_length
        # self.clamp_rot_err = np.pi / 6.0
        self.clamp_pos_err = .1
        self.clamp_rot_err = np.pi / 10.0
        self.jnt_wt_ratio = wln_ratio
        # maximum reach
        self.max_rng = 10.0
        # # extract min max for quick access
        self.min_jnt_values = self.jlc.jnt_ranges[:, 0]
        self.max_jnt_values = self.jlc.jnt_ranges[:, 1]
        self.jnt_range_values = self.max_jnt_values - self.min_jnt_values
        self.min_jnt_value_thresholds = self.min_jnt_values + self.jnt_range_values * self.jnt_wt_ratio
        self.max_jnt_value_thresholds = self.max_jnt_values - self.jnt_range_values * self.jnt_wt_ratio

    def __call__(self,
                 tgt_pos,
                 tgt_rotmat,
                 seed_jnt_values=None,
                 max_n_iter=100,
                 toggle_dbg=False):
        return self.pinv(tgt_pos=tgt_pos,
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
        damping_selection = jnt_values < self.min_jnt_value_thresholds
        normalized_diff = (jnt_values - self.min_jnt_values) / (self.min_jnt_value_thresholds - self.min_jnt_values)
        damping_diff = normalized_diff[damping_selection]
        jnt_wt[damping_selection] = -2 * np.power(damping_diff, 3) + 3 * np.power(damping_diff, 2)
        cutting_selection = jnt_values <= self.min_jnt_values
        jnt_wt[cutting_selection] = 0.0
        # max damping interval
        damping_selection = jnt_values > self.max_jnt_value_thresholds
        normalized_diff = (self.max_jnt_values - jnt_values) / (self.max_jnt_values - self.max_jnt_value_thresholds)
        damping_diff = normalized_diff[damping_selection]
        jnt_wt[damping_selection] = -2 * np.power(damping_diff, 3) + 3 * np.power(damping_diff, 2)
        cutting_selection = jnt_values >= self.max_jnt_values
        jnt_wt[cutting_selection] = 0.0
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

    def pinv(self,
             tgt_pos,
             tgt_rotmat,
             seed_jnt_values=None,
             max_n_iter=100,
             toggle_dbg=False):
        iter_jnt_values = seed_jnt_values
        if seed_jnt_values is None:
            iter_jnt_values = self.jlc.get_jnt_values()
        counter = 0
        while True:
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=iter_jnt_values,
                                                           toggle_jacobian=True,
                                                           update=False)
            f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_poses(src_pos=flange_pos,
                                                                          src_rotmat=flange_rotmat,
                                                                          tgt_pos=tgt_pos,
                                                                          tgt_rotmat=tgt_rotmat)
            if f2t_pos_err < 1e-4 and f2t_rot_err < 1e-3:
                return iter_jnt_values
            clamped_err_vec = self._clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec)
            delta_jnt_values = np.linalg.pinv(j_mat, rcond=1e-4) @ clamped_err_vec
            if toggle_dbg:
                print("f2t_pos_err ", f2t_pos_err, " f2t_rot_err ", f2t_rot_err)
                print("clamped_tgt_err ", clamped_err_vec)
                print("coutner/max_n_iter ", counter, max_n_iter)
            if abs(np.sum(delta_jnt_values)) < 1e-6:
                return None
            iter_jnt_values = iter_jnt_values + delta_jnt_values
            if not self.are_jnts_in_range(iter_jnt_values) or counter > max_n_iter:
                return None
            counter += 1

    def pinv_rr(self,
                tgt_pos,
                tgt_rotmat,
                seed_jnt_values=None,
                max_n_iter=100,
                toggle_dbg=False):
        iter_jnt_values = seed_jnt_values
        if seed_jnt_values is None:
            iter_jnt_values = self.jlc.get_jnt_values()
        counter = 0
        while True:
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=iter_jnt_values,
                                                           toggle_jacobian=True,
                                                           update=False)
            f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_poses(src_pos=flange_pos,
                                                                          src_rotmat=flange_rotmat,
                                                                          tgt_pos=tgt_pos,
                                                                          tgt_rotmat=tgt_rotmat)
            if f2t_pos_err < 1e-4 and f2t_rot_err < 1e-3 and self.jlc.are_jnts_in_ranges(iter_jnt_values):
                return iter_jnt_values
            clamped_err_vec = self._clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec)
            delta_jnt_values = np.linalg.pinv(j_mat, rcond=1e-4) @ clamped_err_vec
            if abs(np.sum(delta_jnt_values)) < 1e-8:
                # print("local minima")
                # local minimia
                pass
            iter_jnt_values = iter_jnt_values + delta_jnt_values
            # iter_jnt_values = np.mod(iter_jnt_values, 2 * np.pi)
            # iter_jnt_values = np.where(iter_jnt_values > np.pi, iter_jnt_values - 2 * np.pi, iter_jnt_values)
            if not self.are_jnts_in_range(iter_jnt_values):
                # random restart
                # print("random restart")
                iter_jnt_values = self.jlc.rand_conf()
            if toggle_dbg:
                jnt_values = self.jlc.get_jnt_values()
                self.jlc.goto_given_conf(jnt_values=iter_jnt_values)
                rkmg.gen_jlc_stick(self.jlc, toggle_flange_frame=True).attach_to(base)
                self.jlc.goto_given_conf(jnt_values=jnt_values)
                # import modeling.geometric_model as gm
                # mgm.gen_arrow(spos=flange_pos, epos=flange_pos + f2t_err_vec[:3] * 1).attach_to(base)
                print("f2t_pos_err ", f2t_pos_err, " f2t_rot_err ", f2t_rot_err)
                print("clamped_tgt_err ", clamped_err_vec)
                print(counter, max_n_iter)
            if counter > max_n_iter:
                return None
            counter += 1

    def dls(self,
            tgt_pos,
            tgt_rotmat,
            seed_jnt_values=None,
            max_n_iter=100,
            toggle_dbg=False):
        """
        this is slower than pinv, weiwei20240627
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param max_n_iter:
        :param toggle_dbg:
        :return:
        """
        iter_jnt_values = seed_jnt_values
        if seed_jnt_values is None:
            iter_jnt_values = self.jlc.get_jnt_values()
        counter = 0
        while True:
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=iter_jnt_values,
                                                           toggle_jacobian=True,
                                                           update=False)
            f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_poses(src_pos=flange_pos,
                                                                          src_rotmat=flange_rotmat,
                                                                          tgt_pos=tgt_pos,
                                                                          tgt_rotmat=tgt_rotmat)
            if f2t_pos_err < 1e-4 and f2t_rot_err < 1e-3 and self.jlc.are_jnts_in_ranges(iter_jnt_values):
                return iter_jnt_values
            clamped_err_vec = self._clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec)
            delta_jnt_values = np.linalg.lstsq(j_mat, clamped_err_vec, rcond=1e-4)[0]
            # delta_jnt_values = (np.linalg.inv(j_mat.T @ j_mat + 1e-4 * np.eye(j_mat.shape[1])) @
            #                     j_mat.T @ clamped_err_vec)
            if toggle_dbg:
                print("f2t_pos_err ", f2t_pos_err, " f2t_rot_err ", f2t_rot_err)
                print("clamped_tgt_err ", clamped_err_vec)
                print("coutner/max_n_iter ", counter, max_n_iter)
            if abs(np.sum(delta_jnt_values)) < 1e-6:
                return None
            iter_jnt_values = iter_jnt_values + delta_jnt_values
            if not self.are_jnts_in_range(iter_jnt_values) or counter > max_n_iter:
                return None
            counter += 1

    def dls_rr(self,
               tgt_pos,
               tgt_rotmat,
               seed_jnt_values=None,
               max_n_iter=100,
               toggle_dbg=False):
        iter_jnt_values = seed_jnt_values
        if seed_jnt_values is None:
            iter_jnt_values = self.jlc.get_jnt_values()
        counter = 0
        while True:
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=iter_jnt_values,
                                                           toggle_jacobian=True,
                                                           update=False)
            f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_poses(src_pos=flange_pos,
                                                                          src_rotmat=flange_rotmat,
                                                                          tgt_pos=tgt_pos,
                                                                          tgt_rotmat=tgt_rotmat)
            if f2t_pos_err < 1e-4 and f2t_rot_err < 1e-3 and self.jlc.are_jnts_in_ranges(iter_jnt_values):
                return iter_jnt_values
            clamped_err_vec = self._clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec)
            delta_jnt_values = (np.linalg.inv(j_mat.T @ j_mat + 1e-4 * np.eye(j_mat.shape[1])) @
                                j_mat.T @ clamped_err_vec)
            iter_jnt_values = iter_jnt_values + delta_jnt_values
            if not self.are_jnts_in_range(iter_jnt_values):
                # random restart
                iter_jnt_values = self.jlc.rand_conf()
            if toggle_dbg:
                import wrs.robot_sim._kinematics.model_generator as rkmg
                jnt_values = self.jlc.get_jnt_values()
                self.jlc.goto_given_conf(jnt_values=iter_jnt_values)
                rkmg.gen_jlc_stick(self.jlc, toggle_flange_frame=True).attach_to(base)
                self.jlc.goto_given_conf(jnt_values=jnt_values)
                mgm.gen_arrow(spos=flange_pos, epos=tgt_pos).attach_to(base)
                print("f2t_pos_err ", f2t_pos_err, " f2t_rot_err ", f2t_rot_err)
            if counter > max_n_iter:
                return None
            counter += 1

    def jt_rr(self,
              tgt_pos,
              tgt_rotmat,
              seed_jnt_values=None,
              max_n_iter=100,
              toggle_dbg=False):
        """
        the jacobian transpose method
        paper: Buss, Introduction to Inverse Kinematics with Jacobian Transpose,
        Pseudoinverse and Damped Least Squares methods
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param max_n_iter:
        :param toggle_dbg:
        :return:
        """
        iter_jnt_values = seed_jnt_values
        if seed_jnt_values is None:
            iter_jnt_values = self.jlc.get_jnt_values()
        counter = 0
        while True:
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=iter_jnt_values,
                                                           toggle_jacobian=True,
                                                           update=False)
            f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_poses(src_pos=flange_pos,
                                                                          src_rotmat=flange_rotmat,
                                                                          tgt_pos=tgt_pos,
                                                                          tgt_rotmat=tgt_rotmat)
            if f2t_pos_err < 1e-4 and f2t_rot_err < 1e-3:
                return iter_jnt_values
            jjt = j_mat @ j_mat.T
            jjt_dot_e = jjt @ f2t_err_vec
            weight = np.dot(f2t_err_vec, jjt_dot_e) / np.dot(jjt_dot_e, jjt_dot_e)
            delta_jnt_values = weight * (j_mat.T @ f2t_err_vec)
            iter_jnt_values = iter_jnt_values + delta_jnt_values
            if not self.are_jnts_in_range(iter_jnt_values):
                # random restart
                iter_jnt_values = self.jlc.rand_conf()
            if toggle_dbg:
                jnt_values = self.jlc.get_jnt_values()
                self.jlc.goto_given_conf(jnt_values=iter_jnt_values)
                rkmg.gen_jlc_stick(self.jlc, toggle_flange_frame=True).attach_to(base)
                self.jlc.goto_given_conf(jnt_values=jnt_values)
                mgm.gen_arrow(spos=flange_pos, epos=tgt_pos).attach_to(base)
                print("f2t_pos_err ", f2t_pos_err, "f2t_rot_err ", f2t_rot_err)
            if counter > max_n_iter:
                return None
            counter += 1

    def pinv_cw(self,
                tgt_pos,
                tgt_rotmat,
                seed_jnt_values=None,
                max_n_iter=100,
                toggle_dbg=False):
        """
        improved cwln method (replaced the least damping in cwln with moore-penrose inverse)
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param max_n_iter:
        :param toggle_dbg:
        :return:
        author: weiwei
        date: 20231101
        """
        iter_jnt_values = seed_jnt_values
        if iter_jnt_values is None:
            iter_jnt_values = self.jlc.get_jnt_values()
        counter = 0
        while True:
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=iter_jnt_values,
                                                           toggle_jacobian=True,
                                                           update=False)
            f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_poses(src_pos=flange_pos,
                                                                          src_rotmat=flange_rotmat,
                                                                          tgt_pos=tgt_pos,
                                                                          tgt_rotmat=tgt_rotmat)
            if f2t_pos_err < 1e-4 and f2t_rot_err < 1e-3 and self.jlc.are_jnts_in_ranges(iter_jnt_values):
                return iter_jnt_values
            clamped_err_vec = self._clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec)
            # clamped_err_vec = f2t_err_vec * .01
            wln, wln_sqrt = self._jnt_wt_mat(iter_jnt_values)
            # weighted clamping
            k_phi = .1
            tmp_mm_jnt_values = self.max_jnt_values + self.min_jnt_values
            phi_q = ((2 * iter_jnt_values - tmp_mm_jnt_values) / self.jnt_range_values) * k_phi
            clamping = -(np.identity(wln.shape[0]) - wln) @ phi_q
            # pinv with weighted clamping
            delta_jnt_values = clamping + wln_sqrt @ np.linalg.pinv(j_mat @ wln_sqrt, rcond=1e-4) @ (
                    clamped_err_vec - j_mat @ clamping)
            if toggle_dbg:
                print("previous iter joint values ", np.degrees(iter_jnt_values))
            # print(max(abs(clamped_err_vec)), max(abs(np.degrees(delta_jnt_values))))
            iter_jnt_values = iter_jnt_values + delta_jnt_values
            # iter_jnt_values = np.mod(iter_jnt_values, 4 * np.pi) - 2 * np.pi
            # iter_jnt_values = np.where(iter_jnt_values > 2 * np.pi, iter_jnt_values - 2 * np.pi, iter_jnt_values)
            if toggle_dbg:
                import wrs.robot_sim._kinematics.model_generator as rkmg
                jnt_values = self.jlc.get_jnt_values()
                self.jlc.goto_given_conf(jnt_values=iter_jnt_values)
                stick_rgba = rm.const.cool_map(counter / max_n_iter)
                rkmg.gen_jlc_stick(self.jlc, stick_rgba=stick_rgba, toggle_jnt_frames=True,
                                   toggle_flange_frame=True).attach_to(base)
                self.jlc.goto_given_conf(jnt_values=jnt_values)
                mgm.gen_arrow(spos=flange_pos, epos=tgt_pos).attach_to(base)
                print("f2t_pos_err ", f2t_pos_err, " f2t_rot_err ", f2t_rot_err)
                print("phi_q ", phi_q)
                print("clamping ", clamping)
                print("jnt weight ", np.diag(wln))
                print("delta_jnt_values ", np.degrees(delta_jnt_values))
                print("current iter joint values ", np.degrees(iter_jnt_values))
                # print("clamped_tgt_err ", clamped_err_vec)
                print(counter, max_n_iter)
            if counter > max_n_iter:
                # base.run()
                return None
                # raise Exception("No IK solution")
            counter += 1

    def pinv_gpm(self,
                 tgt_pos,
                 tgt_rotmat,
                 seed_jnt_values=None,
                 max_n_iter=100,
                 toggle_dbg=False):
        """
        gradient projection method, only applicabled to redundant manipulators, slower than cwln
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param max_n_iter:
        :param toggle_dbg:
        :return:
        author: weiwei
        date: 20231101
        """
        iter_jnt_values = seed_jnt_values
        if seed_jnt_values is None:
            iter_jnt_values = self.jlc.get_jnt_values()
        counter = 0
        while True:
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=iter_jnt_values,
                                                           toggle_jacobian=True,
                                                           update=False)
            f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_poses(src_pos=flange_pos,
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
                import wrs.robot_sim._kinematics.model_generator as rkmg
                mgm.gen_arrow(spos=flange_pos, epos=flange_pos + clamped_err_vec[:3]).attach_to(base)
                self.jlc.goto_given_conf(jnt_values=iter_jnt_values)
                stick_rgba = rm.const.lnk_stick_rgba
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
             seed_jnt_values=None,
             max_n_iter=100,
             toggle_dbg=False):
        # original method from the following paper:
        # paper: Huang, clamping weighted least-norm method for themanipulator kinematic control with constraints
        # does not work on redundant jlcs
        iter_jnt_values = seed_jnt_values
        if iter_jnt_values is None:
            iter_jnt_values = self.jlc.get_jnt_values()
        counter = 0
        while True:
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=iter_jnt_values,
                                                           toggle_jacobian=True,
                                                           update=False)
            f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_poses(src_pos=flange_pos,
                                                                          src_rotmat=flange_rotmat,
                                                                          tgt_pos=tgt_pos,
                                                                          tgt_rotmat=tgt_rotmat)
            if f2t_pos_err < 1e-4 and f2t_rot_err < 1e-3:
                return iter_jnt_values
            clamped_err_vec = self._clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec)
            wln, wln_sqrt = self._jnt_wt_mat(iter_jnt_values)
            # weighted clamping
            k_phi = 0.1
            tmp_mm_jnt_values = self.max_jnt_values + self.min_jnt_values
            phi_q = ((2 * iter_jnt_values - tmp_mm_jnt_values) / self.jnt_range_values) * k_phi
            clamping = -(np.identity(wln.shape[0]) - wln) @ phi_q
            # lambda coefficient
            # min_svd_val = scipy.linalg.svdvals(wln_sqrt)[-1]
            # lam = 1e-9 if min_svd_val < 1e-4 else 0
            lam = 1e-9
            # cwln
            delta_jnt_values = clamping + wln @ j_mat.T @ np.linalg.inv(
                j_mat @ wln @ j_mat.T + lam * np.eye(j_mat.shape[1])) @ (clamped_err_vec - j_mat @ clamping)
            iter_jnt_values = iter_jnt_values + delta_jnt_values
            iter_jnt_values = np.mod(iter_jnt_values, 2 * np.pi)
            iter_jnt_values = np.where(iter_jnt_values > np.pi, iter_jnt_values - 2 * np.pi, iter_jnt_values)
            if toggle_dbg:
                jnt_values = self.jlc.get_jnt_values()
                self.jlc.goto_given_conf(jnt_values=iter_jnt_values)
                rkmg.gen_jlc_stick(self.jlc, toggle_flange_frame=True).attach_to(base)
                self.jlc.goto_given_conf(jnt_values=jnt_values)
                mgm.gen_arrow(spos=flange_pos, epos=tgt_pos).attach_to(base)
                print("f2t_pos_err ", f2t_pos_err, "f2t_rot_err ", f2t_rot_err)
            if counter > max_n_iter:
                return None
            counter += 1
