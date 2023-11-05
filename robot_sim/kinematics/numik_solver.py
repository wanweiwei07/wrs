import numpy as np
import scipy
import basis.robot_math as rm
import robot_sim.kinematics.constant as rkc
import robot_sim.kinematics.jl as lib_jl
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
        self.min_jnt_vals = self.jlc.joint_ranges[:, 0]
        self.max_jnt_vals = self.jlc.joint_ranges[:, 1]
        self.jnt_rngs = self.max_jnt_vals - self.min_jnt_vals
        self.jnt_rngs_mid = (self.max_jnt_vals + self.min_jnt_vals) / 2
        self.min_jnt_threshold = self.min_jnt_vals + self.jnt_rngs * self.jnt_wt_ratio
        self.max_jnt_threshold = self.max_jnt_vals - self.jnt_rngs * self.jnt_wt_ratio

    def _get_max_link_length(self):
        max_len = 0
        for i in range(1, self.jlc.n_dof):
            if self.jlc.joints[i].type == rkc.JointType.REVOLUTE:
                tmp_vec = self.jlc.joints[i].gl_pos_q - self.jlc.joints[i - 1].gl_pos_q
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

    def are_jnts_in_range(self, jnt_values):
        if np.any(jnt_values < self.min_jnt_vals):
            return False
        if np.any(jnt_values > self.max_jnt_vals):
            return False

    def pinv_rr(self,
                tgt_pos,
                tgt_rotmat,
                seed_jnt_vals=None,
                max_n_iter=100,
                toggle_debug=False):
        iter_jnt_vals = seed_jnt_vals
        if seed_jnt_vals is None:
            iter_jnt_vals = self.jlc.get_joint_values()
        counter = 0
        while True:
            tcp_gl_pos, tcp_gl_rotmat, j_mat = self.jlc.forward_kinematics(joint_values=iter_jnt_vals,
                                                                           toggle_jacobian=True,
                                                                           update=False)
            tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec = rm.diff_between_posrot(src_pos=tcp_gl_pos,
                                                                                   src_rotmat=tcp_gl_rotmat,
                                                                                   tgt_pos=tgt_pos,
                                                                                   tgt_rotmat=tgt_rotmat)
            if tcp_pos_err_val < 1e-4 and tcp_rot_err_val < 1e-3:
                return iter_jnt_vals
            clamped_tcp_err_vec = self._clamp_tcp_err(tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec)
            delta_jnt_values = np.linalg.pinv(j_mat, rcond=1e-4) @ clamped_tcp_err_vec
            print(abs(np.sum(delta_jnt_values)))
            if abs(np.sum(delta_jnt_values)) < 1e-8:
                # local minimia
                pass
            iter_jnt_vals = iter_jnt_vals + delta_jnt_values
            if not self.are_jnts_in_range(iter_jnt_vals):
                # random restart
                iter_jnt_vals = self.jlc.rand_conf()
            if toggle_debug:
                import robot_sim.kinematics.model_generator as rkmg
                joint_values = self.jlc.get_joint_values()
                self.jlc.go_given_conf(joint_values=iter_jnt_vals)
                rkmg.gen_jlc_stick(self.jlc, toggle_tcp_frame=True, toggle_joint_frame=True).attach_to(base)
                self.jlc.go_given_conf(joint_values=joint_values)
                import modeling.geometric_model as gm
                gm.gen_arrow(spos=tcp_gl_pos, epos=tcp_gl_pos + tcp_err_vec[:3] * .1).attach_to(base)
                print("tcp_pos_err ", tcp_pos_err_val, " tcp_rot_err ", tcp_rot_err_val)
            if counter > max_n_iter:
                raise Exception("No IK solution")
            counter += 1

    def dls_rr(self,
               tgt_pos,
               tgt_rotmat,
               seed_jnt_vals=None,
               max_n_iter=100,
               toggle_debug=False):
        iter_jnt_vals = seed_jnt_vals
        if seed_jnt_vals is None:
            iter_jnt_vals = self.jlc.get_joint_values()
        counter = 0
        while True:
            tcp_gl_pos, tcp_gl_rotmat, j_mat = self.jlc.forward_kinematics(joint_values=iter_jnt_vals,
                                                                           toggle_jacobian=True,
                                                                           update=False)
            tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec = rm.diff_between_posrot(src_pos=tcp_gl_pos,
                                                                                   src_rotmat=tcp_gl_rotmat,
                                                                                   tgt_pos=tgt_pos,
                                                                                   tgt_rotmat=tgt_rotmat)
            if tcp_pos_err_val < 1e-4 and tcp_rot_err_val < 1e-3:
                return iter_jnt_vals
            clamped_err_vec = self._clamp_tcp_err(tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec)
            delta_jnt_values = (np.linalg.inv(j_mat.T @ j_mat + 1e-4 * np.eye(j_mat.shape[1])) @
                                j_mat.T @ clamped_err_vec)
            iter_jnt_vals = iter_jnt_vals + delta_jnt_values
            if not self.are_jnts_in_range(iter_jnt_vals):
                # random restart
                iter_jnt_vals = self.jlc.rand_conf()
            if toggle_debug:
                import robot_sim.kinematics.model_generator as rkmg
                joint_values = self.jlc.get_joint_values()
                self.jlc.go_given_conf(joint_values=iter_jnt_vals)
                rkmg.gen_jlc_stick(self.jlc, toggle_tcp_frame=True, toggle_joint_frame=True).attach_to(base)
                self.jlc.go_given_conf(joint_values=joint_values)
                import modeling.geometric_model as gm
                gm.gen_arrow(spos=tcp_gl_pos, epos=tgt_pos).attach_to(base)
                print("tcp_pos_err ", tcp_pos_err_val, " tcp_rot_err ", tcp_rot_err_val)
            if counter > max_n_iter:
                raise Exception("No IK solution")
            counter += 1

    def jt_rr(self,
              tgt_pos,
              tgt_rotmat,
              seed_jnt_vals=None,
              max_n_iter=100,
              toggle_debug=False):
        iter_jnt_vals = seed_jnt_vals
        if seed_jnt_vals is None:
            iter_jnt_vals = self.jlc.get_joint_values()
        counter = 0
        while True:
            tcp_gl_pos, tcp_gl_rotmat, j_mat = self.jlc.forward_kinematics(joint_values=iter_jnt_vals,
                                                                           toggle_jacobian=True,
                                                                           update=False)
            tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec = rm.diff_between_posrot(src_pos=tcp_gl_pos,
                                                                                   src_rotmat=tcp_gl_rotmat,
                                                                                   tgt_pos=tgt_pos,
                                                                                   tgt_rotmat=tgt_rotmat)
            if tcp_pos_err_val < 1e-4 and tcp_rot_err_val < 1e-3:
                return iter_jnt_vals
            jjt = j_mat @ j_mat.T
            jjt_dot_e = jjt @ tcp_err_vec
            weight = np.dot(tcp_err_vec, jjt_dot_e) / np.dot(jjt_dot_e, jjt_dot_e)
            delta_jnt_values = weight * (j_mat.T @ tcp_err_vec)
            iter_jnt_vals = iter_jnt_vals + delta_jnt_values
            if not self.are_jnts_in_range(iter_jnt_vals):
                # random restart
                iter_jnt_vals = self.jlc.rand_conf()
            if toggle_debug:
                import robot_sim.kinematics.model_generator as rkmg
                joint_values = self.jlc.get_joint_values()
                self.jlc.go_given_conf(joint_values=iter_jnt_vals)
                rkmg.gen_jlc_stick(self.jlc, toggle_tcp_frame=True, toggle_joint_frame=True).attach_to(base)
                self.jlc.go_given_conf(joint_values=joint_values)
                import modeling.geometric_model as gm
                gm.gen_arrow(spos=tcp_gl_pos, epos=tgt_pos).attach_to(base)
                print("tcp_pos_err ", tcp_pos_err_val, " tcp_rot_err ", tcp_rot_err_val)
            if counter > max_n_iter:
                raise Exception("No IK solution")
            counter += 1

    def pinv_wc(self,
                tgt_pos,
                tgt_rotmat,
                seed_jnt_vals=None,
                max_n_iter=100,
                toggle_debug=False):
        """
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_vals:
        :param max_n_iter:
        :param toggle_debug:
        :return:
        author: weiwei
        date: 20231101
        """
        iter_jnt_vals = seed_jnt_vals
        if seed_jnt_vals is None:
            iter_jnt_vals = self.jlc.get_joint_values()
        counter = 0
        while True:
            tcp_gl_pos, tcp_gl_rotmat, j_mat = self.jlc.forward_kinematics(joint_values=iter_jnt_vals,
                                                                           toggle_jacobian=True,
                                                                           update=False)
            tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec = rm.diff_between_posrot(src_pos=tcp_gl_pos,
                                                                                   src_rotmat=tcp_gl_rotmat,
                                                                                   tgt_pos=tgt_pos,
                                                                                   tgt_rotmat=tgt_rotmat)
            if tcp_pos_err_val < 1e-4 and tcp_rot_err_val < 1e-3:
                return iter_jnt_vals
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
            if toggle_debug:
                import robot_sim.kinematics.model_generator as rkmg
                joint_values = self.jlc.get_joint_values()
                self.jlc.go_given_conf(joint_values=iter_jnt_vals)
                rkmg.gen_jlc_stick(self.jlc, toggle_tcp_frame=True, toggle_joint_frame=True).attach_to(base)
                self.jlc.go_given_conf(joint_values=joint_values)
                import modeling.geometric_model as gm
                gm.gen_arrow(spos=tcp_gl_pos, epos=tgt_pos).attach_to(base)
                print("tcp_pos_err ", tcp_pos_err_val, " tcp_rot_err ", tcp_rot_err_val)
            if counter > max_n_iter:
                raise Exception("No IK solution")
            counter += 1

    def cwln(self,
             tgt_pos,
             tgt_rotmat,
             seed_jnt_vals=None,
             max_n_iter=100,
             toggle_debug=False):
        # Paper: Clamping weighted least-norm method for themanipulator kinematic control with constraints
        iter_jnt_vals = seed_jnt_vals
        if seed_jnt_vals is None:
            iter_jnt_vals = self.jlc.get_joint_values()
        counter = 0
        while True:
            tcp_gl_pos, tcp_gl_rotmat, j_mat = self.jlc.forward_kinematics(joint_values=iter_jnt_vals,
                                                                           toggle_jacobian=True,
                                                                           update=False)
            tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec = rm.diff_between_posrot(src_pos=tcp_gl_pos,
                                                                                   src_rotmat=tcp_gl_rotmat,
                                                                                   tgt_pos=tgt_pos,
                                                                                   tgt_rotmat=tgt_rotmat)
            if tcp_pos_err_val < 1e-4 and tcp_rot_err_val < 1e-3:
                return iter_jnt_vals
            clamped_err_vec = self._clamp_tcp_err(tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec)
            wln, wln_sqrt = self._jnt_wt_mat(iter_jnt_vals)
            # weighted clamping
            k_phi = 0.1
            phi_q = ((2 * iter_jnt_vals - self.jnt_rngs_mid) / self.jnt_rngs) * k_phi
            clamping = -(np.identity(wln.shape[0]) - wln) @ phi_q
            # lambda coefficient
            min_svd_val = scipy.linalg.svdvals(wln_sqrt)[-1]
            lam = 1e-9 if min_svd_val < 1e-4 else 0
            # cwln
            delta_jnt_values = clamping + wln @ j_mat.T @ np.linalg.inv(
                j_mat @ wln @ j_mat.T + lam * np.eye(j_mat.shape[1])) @ (clamped_err_vec - j_mat @ clamping)
            iter_jnt_vals = iter_jnt_vals + delta_jnt_values
            if toggle_debug:
                import robot_sim.kinematics.model_generator as rkmg
                joint_values = self.jlc.get_joint_values()
                self.jlc.go_given_conf(joint_values=iter_jnt_vals)
                rkmg.gen_jlc_stick(self.jlc, toggle_tcp_frame=True, toggle_joint_frame=True).attach_to(base)
                self.jlc.go_given_conf(joint_values=joint_values)
                import modeling.geometric_model as gm
                gm.gen_arrow(spos=tcp_gl_pos, epos=tgt_pos).attach_to(base)
                print("tcp_pos_err ", tcp_pos_err_val, " tcp_rot_err ", tcp_rot_err_val)
            if counter > max_n_iter:
                raise Exception("No IK solution")
            counter += 1

    def num_ik(self,
               tgt_pos,
               tgt_rotmat,
               tcp_jnt_id=None,
               tcp_loc_pos=None,
               tcp_loc_rotmat=None,
               seed_jnt_values=None,
               max_n_iter=100,
               policy_for_local_minima="randomrestart",
               toggle_debug=False):
        """
        solveik numerically using the Levenberg-Marquardt Method
        the details of this method can be found in: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
        NOTE: if list, len(tgt_pos)=len(tgt_rotmat) <= len(tcp_joint_id)=len(tcp_loc_pos)=len(tcp_loc_rotmat)
        :param tgt_pos: the position of the goal, 1-by-3 numpy ndarray
        :param tgt_rotmat: the orientation of the goal, 3-by-3 numpyndarray
        :param tcp_loc_pos: 1x3 nparray, decribed in the local frame of self.joints[tcp_joint_id], single value or list
        :param tcp_loc_rotmat: 3x3 nparray, decribed in the local frame of self.joints[tcp_joint_id], single value or list
        :param tcp_jnt_id: a joint ID in the self.tgtjnts
        :param seed_jnt_values: the starting configuration used in the numerical iteration
        :param max_n_iter: max number of numercial iternations
        :param policy_for_local_minima: what to do at local minima: "accept", "randomrestart", "end_type"
        :return: a 1xn numpy ndarray
        author: weiwei
        date: 20180203, 20200328
        """
        delta_pos = tgt_pos - self.jlc.joints[0].gl_pos0
        if np.linalg.norm(delta_pos) > self.max_rng:
            print("The goal is outside maximum range!")
            return None
        iter_jnt_values = self.jlc.home if seed_jnt_values is None else seed_jnt_values
        ws_wtdiagmat = np.diag(self.ws_wtlist)
        if toggle_debug:
            if "lib_jlm" not in dir():
                import robot_sim.kinematics.model_generator as jlm
            if "plt" not in dir():
                import matplotlib.pyplot as plt
            dqbefore = []
            dqcorrected = []
            dqnull = []
            ajpath = []
        random_restart = False
        prev_erronorm = 0.0
        max_errornorm = 0.0
        for i in range(max_n_iter):
            j = self.jlc.jacobian(joint_values=iter_jnt_values,
                                  tcp_joint_id=tcp_jnt_id,
                                  tcp_loc_pos=tcp_loc_pos,
                                  tcp_loc_rotmat=tcp_loc_rotmat,
                                  update=False)
            tcp_err_vec = self.tcp_error(tgt_pos, tgt_rotmat, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
            tcp_err_scalar = tcp_err_vec.T.dot(ws_wtdiagmat).dot(tcp_err_vec)
            # err = .05 / errnorm * err if errnorm > .05 else err
            if tcp_err_scalar > max_errornorm:
                max_errornorm = tcp_err_scalar
            if toggle_debug:
                print(tcp_err_scalar)
                ajpath.append(iter_jnt_values)
            if tcp_err_scalar < 1e-9:
                if toggle_debug:
                    print(f"Number of IK iterations before finding a result: {i}")
                    fig = plt.figure()
                    axbefore = fig.add_subplot(411)
                    axbefore.set_title('Original dq')
                    axnull = fig.add_subplot(412)
                    axnull.set_title('dqref on Null space')
                    axcorrec = fig.add_subplot(413)
                    axcorrec.set_title('Minimized dq')
                    axaj = fig.add_subplot(414)
                    axbefore.plot(dqbefore)
                    axnull.plot(dqnull)
                    axcorrec.plot(dqcorrected)
                    axaj.plot(ajpath)
                    plt.show()
                return iter_jnt_values
            else:
                # judge local minima
                if abs(tcp_err_scalar - prev_erronorm) < 1e-12:
                    if toggle_debug:
                        fig = plt.figure()
                        axbefore = fig.add_subplot(411)
                        axbefore.set_title('Original dq')
                        axnull = fig.add_subplot(412)
                        axnull.set_title('dqref on Null space')
                        axcorrec = fig.add_subplot(413)
                        axcorrec.set_title('Minimized dq')
                        axaj = fig.add_subplot(414)
                        axbefore.plot(dqbefore)
                        axnull.plot(dqnull)
                        axcorrec.plot(dqcorrected)
                        axaj.plot(ajpath)
                        plt.show()
                    if policy_for_local_minima == 'accept':
                        print('Bypassing local minima! The return value is a local minima, not an exact IK result.')
                        return iter_jnt_values
                    elif policy_for_local_minima == 'randomrestart':
                        print('Local Minima! Random restart at local minima!')
                        jnt_values_iter = self.jlc.rand_conf()
                        self.jlc.fk(jnt_values_iter)
                        random_restart = True
                        continue
                    else:
                        print('No feasible IK solution!')
                        break
                else:
                    # -- notes --
                    ## note1: do not use np.linalg.inv since it is not precise
                    ## note2: use np.linalg.solve if the system is exactly determined, it is faster
                    ## note3: use np.linalg.lstsq if there might be singularity (no regularization)
                    ## see https://stackoverflow.com/questions/34170618/normal-equation-and-numpy-least-squares-solve-methods-difference-in-regress
                    ## note4: null space https://www.slideserve.com/marietta/kinematic-redundancy
                    ## note5: avoid joint limits; Paper Name: Clamping weighted least-norm method for the manipulator kinematic control: Avoiding joint limits
                    ## note6: constant damper; Sugihara Paper: https://www.mi.ams.eng.osaka-u.ac.jp/member/sugihara/pub/jrsj_ik.pdf
                    # strecthingcoeff = 1 / (1 + math.exp(1 / ((errnorm / self.max_rng) * 1000 + 1)))
                    # strecthingcoeff = -2*math.pow(errnorm / errnormmax, 3)+3*math.pow(errnorm / errnormmax, 2)
                    # print("stretching ", strecthingcoeff)
                    # dampercoeff = (strecthingcoeff + .1) * 1e-6  # a non-zero regulation coefficient
                    damper_coeff = 1e-3 * tcp_err_scalar + 1e-6  # a non-zero regulation coefficient
                    # -- lft moore-penrose inverse --
                    ## jtj = armjac.T.dot(armjac)
                    ## regulator = regcoeff*np.identity(jtj.shape[0])
                    ## jstar = np.linalg.inv(jtj+regulator).dot(armjac.T)
                    ## dq = jstar.dot(err)
                    # -- rgt moore-penrose inverse --
                    # # jjt
                    # jjt = j.dot(j.T)
                    # damper = dampercoeff * np.identity(jjt.shape[0])
                    # jsharp = j.T.dot(np.linalg.inv(jjt + damper))
                    # weighted jjt
                    qs_wtdiagmat = self._wln_weightmat(jnt_values_iter)
                    # WLN
                    w_jt = qs_wtdiagmat.dot(j.T)
                    j_w_jt = j.dot(w_jt)
                    damper = damper_coeff * np.identity(j_w_jt.shape[0])
                    jsharp = w_jt.dot(np.linalg.inv(j_w_jt + damper))
                    # Clamping (Paper Name: Clamping weighted least-norm method for the manipulator kinematic control)
                    phi_q = ((2 * jnt_values_iter - self.jnt_rngs_mid) / self.jnt_rngs)
                    clamping = -(np.identity(qs_wtdiagmat.shape[0]) - qs_wtdiagmat).dot(phi_q)
                    # # if do not use WLN
                    # j_jt = j.dot(j.T)
                    # damper = dampercoeff * np.identity(j_jt.shape[0])
                    # jsharp = j.T.dot(np.linalg.inv(j_jt + damper))
                    # update dq
                    dq = .1 * jsharp.dot(tcp_err_vec)
                    if not random_restart:
                        w_init = 0.1
                    else:
                        w_init = 0
                    w_middle = 1
                    ns_projmat = np.identity(jnt_values_iter.size) - jsharp.dot(j)
                    dqref_init = (jnt_values_ref - jnt_values_iter)
                    dqref_on_ns = ns_projmat.dot(w_init * dqref_init + w_middle * clamping)
                    dq_minimized = dq + dqref_on_ns
                    if toggle_debug:
                        dqbefore.append(dq)
                        dqcorrected.append(dq_minimized)
                        dqnull.append(dqref_on_ns)
                jnt_values_iter += dq_minimized  # translation problem
                # isdragged, jntvalues_iter = self.check_jntsrange_drag(jntvalues_iter)
                # print(jnt_values_iter)
                self.jlc.fk(joint_values=jnt_values_iter)
                # if toggle_debug:
                #     self.jlc.gen_stickmodel(tcp_joint_id=tcp_joint_id, tcp_loc_pos=tcp_loc_pos,
                #                                    tcp_loc_rotmat=tcp_loc_rotmat, toggle_joint_frame=True).attach_to(base)
            prev_erronorm = tcp_err_scalar
        if toggle_debug:
            fig = plt.figure()
            axbefore = fig.add_subplot(411)
            axbefore.set_title('Original dq')
            axnull = fig.add_subplot(412)
            axnull.set_title('dqref on Null space')
            axcorrec = fig.add_subplot(413)
            axcorrec.set_title('Minimized dq')
            axaj = fig.add_subplot(414)
            axbefore.plot(dqbefore)
            axnull.plot(dqnull)
            axcorrec.plot(dqcorrected)
            axaj.plot(ajpath)
            plt.show()
            self.jlc.gen_stickmodel(tcp_jnt_id=tcp_jnt_id, tcp_loc_pos=tcp_loc_pos,
                                    tcp_loc_rotmat=tcp_loc_rotmat, toggle_jntscs=True).attach_to(base)
            # base.run()
        self.jlc.fk(jnt_values_bk)
        print('Failed to solve the IK, returning None.')
        return None

    def regulate_jnts(self):
        """
        check if the given joint_values is inside the oeprating range
        The joint values out of range will be pulled back to their maxima
        :return: Two parameters, one is true or false indicating if the joint values are inside the range or not
                The other is the joint values after dragging.
                If the joints were not dragged, the same joint values will be returned
        author: weiwei
        date: 20161205
        """
        counter = 0
        for id in self.jlc.tgtjnts:
            if self.jlc.joints[id]["end_type"] == 'revolute':
                if self.jlc.joints[id]['motion_rng'][1] - self.jlc.joints[id]['motion_rng'][0] >= math.pi * 2:
                    rm.regulate_angle(self.jlc.joints[id]['motion_rng'][0],
                                      self.jlc.joints[id]['motion_rng'][1],
                                      self.jlc.joints[id]["movement"])
            counter += 1

    def check_jntranges_drag(self, jnt_values):
        """
        check if the given joint_values is inside the oeprating range
        The joint values out of range will be pulled back to their maxima
        :param jnt_values: a 1xn numpy ndarray
        :return: Two parameters, one is true or false indicating if the joint values are inside the range or not
                The other is the joint values after dragging.
                If the joints were not dragged, the same joint values will be returned
        author: weiwei
        date: 20161205
        """
        counter = 0
        isdragged = np.zeros_like(jnt_values)
        jntvaluesdragged = jnt_values.copy()
        for id in self.jlc.tgtjnts:
            if self.jlc.joints[id]["end_type"] == 'revolute':
                if self.jlc.joints[id]['motion_rng'][1] - self.jlc.joints[id]['motion_rng'][0] < math.pi * 2:
                    # if joint_values[counter] < jlinstance.joints[id]['motion_rng'][0]:
                    #     isdragged[counter] = 1
                    #     jntvaluesdragged[counter] = jlinstance.joints[id]['motion_rng'][0]
                    # elif joint_values[counter] > jlinstance.joints[id]['motion_rng'][1]:
                    #     isdragged[counter] = 1
                    #     jntvaluesdragged[counter] = jlinstance.joints[id]['motion_rng'][1]
                    print("Drag revolute")
                    if jnt_values[counter] < self.jlc.joints[id]['motion_rng'][0] or jnt_values[counter] > \
                            self.jlc.joints[id]['motion_rng'][1]:
                        isdragged[counter] = 1
                        jntvaluesdragged[counter] = (self.jlc.joints[id]['motion_rng'][1] +
                                                     self.jlc.joints[id][
                                                         'motion_rng'][0]) / 2
            elif self.jlc.joints[id]["end_type"] == 'prismatic':  # prismatic
                # if joint_values[counter] < jlinstance.joints[id]['motion_rng'][0]:
                #     isdragged[counter] = 1
                #     jntvaluesdragged[counter] = jlinstance.joints[id]['motion_rng'][0]
                # elif joint_values[counter] > jlinstance.joints[id]['motion_rng'][1]:
                #     isdragged[counter] = 1
                #     jntvaluesdragged[counter] = jlinstance.joints[id]['motion_rng'][1]
                print("Drag prismatic")
                if jnt_values[counter] < self.jlc.joints[id]['motion_rng'][0] or jnt_values[counter] > \
                        self.jlc.joints[id]['motion_rng'][1]:
                    isdragged[counter] = 1
                    jntvaluesdragged[counter] = (self.jlc.joints[id]['motion_rng'][1] + self.jlc.joints[id][
                        "rngmin"]) / 2
        return isdragged, jntvaluesdragged

    def numik_rel(self, deltapos, deltarotmat, tcp_jnt_id=None, tcp_loc_pos=None, tcp_loc_rotmat=None):
        """
        add deltapos, deltarotmat to the current end_type
        :param deltapos:
        :param deltarotmat:
        :param tcp_jnt_id: a joint ID in the self.tgtjnts
        :param tcp_loc_pos: 1x3 nparray, decribed in the local frame of self.joints[tcp_joint_id], single value or list
        :param tcp_loc_rotmat: 3x3 nparray, decribed in the local frame of self.joints[tcp_joint_id], single value or list
        :return:
        author: weiwei
        date: 20170412, 20200331
        """
        tcp_gl_pos, tcp_gl_rotmat = self.get_gl_tcp(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
        if isinstance(tcp_jnt_id, list):
            tgt_pos = []
            tgt_rotmat = []
            for i, jid in enumerate(tcp_jnt_id):
                tgt_pos.append(tcp_gl_pos[i] + deltapos[i])
                tgt_rotmat.append(np.dot(deltarotmat, tcp_gl_rotmat[i]))
            start_conf = self.jlc.getjntvalues()
            # return numik(rjlinstance, tgt_pos, tgt_rotmat, seed_joint_values=seed_joint_values, tcp_joint_id=tcp_joint_id, tcp_loc_pos=tcp_loc_pos, tcp_loc_rotmat=tcp_loc_rotmat)
        else:
            tgt_pos = tcp_gl_pos + deltapos
            tgt_rotmat = np.dot(deltarotmat, tcp_gl_rotmat)
            start_conf = self.jlc.getjntvalues()
        return self.numik(tgt_pos, tgt_rotmat, start_conf=start_conf, tcp_jnt_id=tcp_jnt_id, tcp_loc_pos=tcp_loc_pos,
                          tcp_loc_rotmat=tcp_loc_rotmat)
