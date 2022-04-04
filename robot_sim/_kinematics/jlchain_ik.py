import math
import numpy as np
import basis.robot_math as rm
import warnings as wns


class JLChainIK(object):

    def __init__(self, jlc_object, wln_ratio=.05):
        self.jlc_object = jlc_object
        self.wln_ratio = wln_ratio
        # IK macros
        wt_pos = 0.628  # 0.628m->1 == 0.01->0.00628m
        wt_agl = 1 / (math.pi * math.pi)  # pi->1 == 0.01->0.18degree
        self.ws_wtlist = [wt_pos, wt_pos, wt_pos, wt_agl, wt_agl, wt_agl]
        # maximum reach
        self.max_rng = 20.0
        # # extract min max for quick access
        self.jmvmin = self.jlc_object.jnt_ranges[:, 0]
        self.jmvmax = self.jlc_object.jnt_ranges[:, 1]
        self.jmvrng = self.jmvmax - self.jmvmin
        self.jmvmiddle = (self.jmvmax + self.jmvmin) / 2
        self.jmvmin_threshhold = self.jmvmin + self.jmvrng * self.wln_ratio
        self.jmvmax_threshhold = self.jmvmax - self.jmvrng * self.wln_ratio

    def _jacobian_sgl(self, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat):
        """
        compute the jacobian matrix of a rjlinstance
        only a single tcp_jnt_id is acceptable
        :param tcp_jnt_id: the joint id where the tool center pose is specified, single vlaue
        :param tcp_loc_pos:
        :param tcp_loc_rotmat:
        :return: j, a 6xn nparray
        author: weiwei
        date: 20161202, 20200331, 20200706
        """
        tcp_gl_pos, tcp_gl_rotmat = self.get_gl_tcp(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
        j = np.zeros((6, len(self.jlc_object.tgtjnts)))
        counter = 0
        for jid in self.jlc_object.tgtjnts:
            grax = self.jlc_object.jnts[jid]["gl_motionax"]
            if self.jlc_object.jnts[jid]["type"] == 'revolute':
                diffq = tcp_gl_pos - self.jlc_object.jnts[jid]["gl_posq"]
                j[:3, counter] = np.cross(grax, diffq)
                j[3:6, counter] = grax
            if self.jlc_object.jnts[jid]["type"] == 'prismatic':
                j[:3, counter] = grax
            counter += 1
            if jid == tcp_jnt_id:
                break
        return j

    def _wln_weightmat(self, jntvalues):
        """
        get the wln weightmat
        :param jntvalues:
        :return:
        author: weiwei
        date: 20201126
        """
        wtmat = np.ones(len(self.jlc_object.tgtjnts))
        # min damping interval
        selection = jntvalues < self.jmvmin_threshhold
        normalized_diff_at_selected = ((jntvalues - self.jmvmin) / (self.jmvmin_threshhold - self.jmvmin))[selection]
        wtmat[selection] = -2 * np.power(normalized_diff_at_selected, 3) + 3 * np.power(normalized_diff_at_selected, 2)
        # max damping interval
        selection = jntvalues > self.jmvmax_threshhold
        normalized_diff_at_selected = ((self.jmvmax - jntvalues) / (self.jmvmax - self.jmvmax_threshhold))[selection]
        wtmat[selection] = -2 * np.power(normalized_diff_at_selected, 3) + 3 * np.power(normalized_diff_at_selected, 2)
        wtmat[jntvalues >= self.jmvmax] = 0
        wtmat[jntvalues <= self.jmvmin] = 0
        return np.diag(wtmat)

    def jacobian(self, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat):
        """
        compute the jacobian matrix of a rjlinstance
        multiple tcp_jnt_id acceptable
        :param tcp_jnt_id: the joint id where the tool center pose is specified, single vlaue or list
        :param tcp_loc_pos:
        :param tcp_loc_rotmat:
        :return: j, a sum(len(option))xn nparray
        author: weiwei
        date: 20161202, 20200331, 20200706, 20201114
        """
        if isinstance(tcp_jnt_id, list):
            j = np.zeros((6 * (len(tcp_jnt_id)), len(self.jlc_object.tgtjnts)))
            for i, this_tcp_jnt_id in enumerate(tcp_jnt_id):
                j[6 * i:6 * i + 6, :] = self._jacobian_sgl(this_tcp_jnt_id, tcp_loc_pos[i], tcp_loc_rotmat[i])
            return j
        else:
            return self._jacobian_sgl(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)

    def manipulability(self,
                       tcp_jnt_id,
                       tcp_loc_pos,
                       tcp_loc_rotmat):
        """
        compute the yoshikawa manipulability of the rjlinstance
        :param tcp_jnt_id: the joint id where the tool center pose is specified, single vlaue or list
        :return:
        author: weiwei
        date: 20200331
        """
        j = self.jacobian(tcp_jnt_id,
                          tcp_loc_pos,
                          tcp_loc_rotmat)
        return math.sqrt(np.linalg.det(np.dot(j, j.transpose())))

    def manipulability_axmat(self,
                             tcp_jnt_id,
                             tcp_loc_pos,
                             tcp_loc_rotmat,
                             type="translational"):
        """
        compute the yasukawa manipulability of the rjlinstance
        :param tcp_jnt_id: the joint id where the tool center pose is specified, single vlaue or list
        :param type: translational, rotational
        :return: axmat with each column being the manipulability
        """
        j = self.jacobian(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
        if type == "translational":
            jjt = np.dot(j[:3, :], j.transpose()[:, :3])
        elif type == "rotational":
            jjt = np.dot(j[3:, :], j.transpose()[:, 3:])
        else:
            raise Exception("The parameter 'type' must be 'translational' or 'rotational'!")
        pcv, pcaxmat = np.linalg.eig(jjt)
        axmat = np.eye(3)
        axmat[:, 0] = np.sqrt(pcv[0]) * pcaxmat[:, 0]
        axmat[:, 1] = np.sqrt(pcv[1]) * pcaxmat[:, 1]
        axmat[:, 2] = np.sqrt(pcv[2]) * pcaxmat[:, 2]
        return axmat

    def get_gl_tcp(self, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat):
        """
        Get the global tool center pose given tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat
        tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat are the tool center pose parameters. They are
        used for temporary computation, the self.tcp_xxx parameters will not be changed
        in case None is provided, the self.tcp_jnt_id, self.tcp_loc_pos, self.tcp_loc_rotmat will be used
        :param tcp_jnt_id: a joint ID in the self.tgtjnts
        :param tcp_loc_pos: 1x3 nparray, decribed in the local frame of self.jnts[tcp_jnt_id], single value or list
        :param tcp_loc_rotmat: 3x3 nparray, decribed in the local frame of self.jnts[tcp_jnt_id], single value or list
        :return: a single value or a list depending on the input
        author: weiwei
        date: 20200706
        """
        if tcp_jnt_id is None:
            tcp_jnt_id = self.jlc_object.tcp_jnt_id
        if tcp_loc_pos is None:
            tcp_loc_pos = self.jlc_object.tcp_loc_pos
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.jlc_object.tcp_loc_rotmat
        if isinstance(tcp_jnt_id, list):
            returnposlist = []
            returnrotmatlist = []
            for i, jid in enumerate(tcp_jnt_id):
                tcp_gl_pos = np.dot(self.jlc_object.jnts[jid]["gl_rotmatq"], tcp_loc_pos[i]) + \
                             self.jlc_object.jnts[jid]["gl_posq"]
                tcp_gl_rotmat = np.dot(self.jlc_object.jnts[jid]["gl_rotmatq"], tcp_loc_rotmat[i])
                returnposlist.append(tcp_gl_pos)
                returnrotmatlist.append(tcp_gl_rotmat)
            return [returnposlist, returnrotmatlist]
        else:
            tcp_gl_pos = np.dot(self.jlc_object.jnts[tcp_jnt_id]["gl_rotmatq"], tcp_loc_pos) + \
                         self.jlc_object.jnts[tcp_jnt_id]["gl_posq"]
            tcp_gl_rotmat = np.dot(self.jlc_object.jnts[tcp_jnt_id]["gl_rotmatq"], tcp_loc_rotmat)
            return tcp_gl_pos, tcp_gl_rotmat

    def tcp_error(self, tgt_pos, tgt_rot, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat):
        """
        compute the error between the rjlinstance's end and tgt_pos, tgt_rotmat
        NOTE: if list, len(tgt_pos)=len(tgt_rotmat) <= len(tcp_jnt_id)=len(tcp_loc_pos)=len(tcp_loc_rotmat)
        :param tgt_pos: the position vector of the goal (could be a single value or a list of jntid)
        :param tgt_rot: the rotation matrix of the goal (could be a single value or a list of jntid)
        :param tcp_jnt_id: a joint ID in the self.tgtjnts
        :param tcp_loc_pos: 1x3 nparray, decribed in the local frame of self.jnts[tcp_jnt_id], single value or list
        :param tcp_loc_rotmat: 3x3 nparray, decribed in the local frame of self.jnts[tcp_jnt_id], single value or list
        :return: a 1x6 nparray where the first three indicates the displacement in pos,
                    the second three indictes the displacement in rot
        author: weiwei
        date: 20180827, 20200331, 20200705
        """
        tcp_gl_pos, tcp_gl_rotmat = self.get_gl_tcp(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
        if isinstance(tgt_pos, list):
            deltapw = np.zeros(6 * len(tgt_pos))
            for i, this_tgt_pos in enumerate(tgt_pos):
                deltapw[6 * i:6 * i + 3] = (this_tgt_pos - tcp_gl_pos[i])
                deltapw[6 * i + 3:6 * i + 6] = rm.deltaw_between_rotmat(tcp_gl_rotmat[i], tgt_rot[i])
            return deltapw
        else:
            deltapw = np.zeros(6)
            deltapw[0:3] = (tgt_pos - tcp_gl_pos)
            deltapw[3:6] = rm.deltaw_between_rotmat(tcp_gl_rotmat, tgt_rot)
            return deltapw

    def regulate_jnts(self):
        """
        check if the given jntvalues is inside the oeprating range
        The joint values out of range will be pulled back to their maxima
        :return: Two parameters, one is true or false indicating if the joint values are inside the range or not
                The other is the joint values after dragging.
                If the joints were not dragged, the same joint values will be returned
        author: weiwei
        date: 20161205
        """
        counter = 0
        for id in self.jlc_object.tgtjnts:
            if self.jlc_object.jnts[id]["type"] == 'revolute':
                if self.jlc_object.jnts[id]['motion_rng'][1] - self.jlc_object.jnts[id]['motion_rng'][0] >= math.pi * 2:
                    rm.regulate_angle(self.jlc_object.jnts[id]['motion_rng'][0],
                                      self.jlc_object.jnts[id]['motion_rng'][1],
                                      self.jlc_object.jnts[id]["movement"])
            counter += 1

    def check_jntranges_drag(self, jnt_values):
        """
        check if the given jntvalues is inside the oeprating range
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
        for id in self.jlc_object.tgtjnts:
            if self.jlc_object.jnts[id]["type"] == 'revolute':
                if self.jlc_object.jnts[id]['motion_rng'][1] - self.jlc_object.jnts[id]['motion_rng'][0] < math.pi * 2:
                    # if jntvalues[counter] < jlinstance.jnts[id]['motion_rng'][0]:
                    #     isdragged[counter] = 1
                    #     jntvaluesdragged[counter] = jlinstance.jnts[id]['motion_rng'][0]
                    # elif jntvalues[counter] > jlinstance.jnts[id]['motion_rng'][1]:
                    #     isdragged[counter] = 1
                    #     jntvaluesdragged[counter] = jlinstance.jnts[id]['motion_rng'][1]
                    print("Drag revolute")
                    if jnt_values[counter] < self.jlc_object.jnts[id]['motion_rng'][0] or jnt_values[counter] > \
                            self.jlc_object.jnts[id]['motion_rng'][1]:
                        isdragged[counter] = 1
                        jntvaluesdragged[counter] = (self.jlc_object.jnts[id]['motion_rng'][1] +
                                                     self.jlc_object.jnts[id][
                                                         'motion_rng'][0]) / 2
            elif self.jlc_object.jnts[id]["type"] == 'prismatic':  # prismatic
                # if jntvalues[counter] < jlinstance.jnts[id]['motion_rng'][0]:
                #     isdragged[counter] = 1
                #     jntvaluesdragged[counter] = jlinstance.jnts[id]['motion_rng'][0]
                # elif jntvalues[counter] > jlinstance.jnts[id]['motion_rng'][1]:
                #     isdragged[counter] = 1
                #     jntvaluesdragged[counter] = jlinstance.jnts[id]['motion_rng'][1]
                print("Drag prismatic")
                if jnt_values[counter] < self.jlc_object.jnts[id]['motion_rng'][0] or jnt_values[counter] > \
                        self.jlc_object.jnts[id]['motion_rng'][1]:
                    isdragged[counter] = 1
                    jntvaluesdragged[counter] = (self.jlc_object.jnts[id]['motion_rng'][1] + self.jlc_object.jnts[id][
                        "rngmin"]) / 2
        return isdragged, jntvaluesdragged

    def num_ik(self,
               tgt_pos,
               tgt_rot,
               seed_jnt_values=None,
               max_niter=100,
               tcp_jnt_id=None,
               tcp_loc_pos=None,
               tcp_loc_rotmat=None,
               local_minima="randomrestart",
               toggle_debug=False):
        """
        solveik numerically using the Levenberg-Marquardt Method
        the details of this method can be found in: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
        NOTE: if list, len(tgt_pos)=len(tgt_rotmat) <= len(tcp_jnt_id)=len(tcp_loc_pos)=len(tcp_loc_rotmat)
        :param tgt_pos: the position of the goal, 1-by-3 numpy ndarray
        :param tgt_rot: the orientation of the goal, 3-by-3 numpyndarray
        :param seed_jnt_values: the starting configuration used in the numerical iteration
        :param max_niter: max number of numercial iternations
        :param tcp_jnt_id: a joint ID in the self.tgtjnts
        :param tcp_loc_pos: 1x3 nparray, decribed in the local frame of self.jnts[tcp_jnt_id], single value or list
        :param tcp_loc_rotmat: 3x3 nparray, decribed in the local frame of self.jnts[tcp_jnt_id], single value or list
        :param local_minima: what to do at local minima: "accept", "randomrestart", "end"
        :return: a 1xn numpy ndarray
        author: weiwei
        date: 20180203, 20200328
        """
        deltapos = tgt_pos - self.jlc_object.jnts[0]['gl_pos0']
        if np.linalg.norm(deltapos) > self.max_rng:
            print("The goal is outside maximum range!")
            return None
        if tcp_jnt_id is None:
            tcp_jnt_id = self.jlc_object.tcp_jnt_id
        if tcp_loc_pos is None:
            tcp_loc_pos = self.jlc_object.tcp_loc_pos
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.jlc_object.tcp_loc_rotmat
        # trim list
        if isinstance(tgt_pos, list):
            tcp_jnt_id = tcp_jnt_id[0:len(tgt_pos)]
            tcp_loc_pos = tcp_loc_pos[0:len(tgt_pos)]
            tcp_loc_rotmat = tcp_loc_rotmat[0:len(tgt_pos)]
        elif isinstance(tcp_jnt_id, list):
            tcp_jnt_id = tcp_jnt_id[0]
            tcp_loc_pos = tcp_loc_pos[0]
            tcp_loc_rotmat = tcp_loc_rotmat[0]
        jnt_values_bk = self.jlc_object.get_jnt_values()
        jnt_values_iter = self.jlc_object.homeconf if seed_jnt_values is None else seed_jnt_values.copy()
        self.jlc_object.fk(jnt_values=jnt_values_iter)
        jnt_values_ref = jnt_values_iter.copy()
        if isinstance(tcp_jnt_id, list):
            diaglist = []
            for i in tcp_jnt_id:
                diaglist += self.ws_wtlist
            ws_wtdiagmat = np.diag(diaglist)
        else:
            ws_wtdiagmat = np.diag(self.ws_wtlist)
        if toggle_debug:
            if "jlm" not in dir():
                import robot_sim._kinematics.jlchain_mesh as jlm
            if "plt" not in dir():
                import matplotlib.pyplot as plt
            dqbefore = []
            dqcorrected = []
            dqnull = []
            ajpath = []
        random_restart = False
        errnormlast = 0.0
        errnormmax = 0.0
        for i in range(max_niter):
            j = self.jacobian(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
            err = self.tcp_error(tgt_pos, tgt_rot, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
            errnorm = err.T.dot(ws_wtdiagmat).dot(err)
            # err = .05 / errnorm * err if errnorm > .05 else err
            if errnorm > errnormmax:
                errnormmax = errnorm
            if toggle_debug:
                print(errnorm)
                ajpath.append(self.jlc_object.get_jnt_values())
            if errnorm < 1e-9:
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
                jntvalues_return = self.jlc_object.get_jnt_values()
                self.jlc_object.fk(jnt_values=jnt_values_bk)
                return jntvalues_return
            else:
                # judge local minima
                if abs(errnorm - errnormlast) < 1e-12:
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
                    if local_minima == 'accept':
                        print('Bypassing local minima! The return value is a local minima, not an exact IK result.')
                        jntvalues_return = self.jlc_object.get_jnt_values()
                        self.jlc_object.fk(jnt_values_bk)
                        return jntvalues_return
                    elif local_minima == 'randomrestart':
                        print('Local Minima! Random restart at local minima!')
                        jnt_values_iter = self.jlc_object.rand_conf()
                        self.jlc_object.fk(jnt_values_iter)
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
                    dampercoeff = 1e-3 * errnorm + 1e-6  # a non-zero regulation coefficient
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
                    damper = dampercoeff * np.identity(j_w_jt.shape[0])
                    jsharp = w_jt.dot(np.linalg.inv(j_w_jt + damper))
                    # Clamping (Paper Name: Clamping weighted least-norm method for the manipulator kinematic control)
                    phi_q = ((2 * jnt_values_iter - self.jmvmiddle) / self.jmvrng)
                    clamping = -(np.identity(qs_wtdiagmat.shape[0]) - qs_wtdiagmat).dot(phi_q)
                    # # if do not use WLN
                    # j_jt = j.dot(j.T)
                    # damper = dampercoeff * np.identity(j_jt.shape[0])
                    # jsharp = j.T.dot(np.linalg.inv(j_jt + damper))
                    # update dq
                    dq = .1 * jsharp.dot(err)
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
                self.jlc_object.fk(jnt_values=jnt_values_iter)
                # if toggle_debug:
                #     self.jlc_object.gen_stickmodel(tcp_jnt_id=tcp_jnt_id, tcp_loc_pos=tcp_loc_pos,
                #                                    tcp_loc_rotmat=tcp_loc_rotmat, toggle_jntscs=True).attach_to(base)
            errnormlast = errnorm
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
            self.jlc_object.gen_stickmodel(tcp_jnt_id=tcp_jnt_id, tcp_loc_pos=tcp_loc_pos,
                                           tcp_loc_rotmat=tcp_loc_rotmat, toggle_jntscs=True).attach_to(base)
            # base.run()
        self.jlc_object.fk(jnt_values_bk)
        print('Failed to solve the IK, returning None.')
        return None

    def numik_rel(self, deltapos, deltarotmat, tcp_jnt_id=None, tcp_loc_pos=None, tcp_loc_rotmat=None):
        """
        add deltapos, deltarotmat to the current end
        :param deltapos:
        :param deltarotmat:
        :param tcp_jnt_id: a joint ID in the self.tgtjnts
        :param tcp_loc_pos: 1x3 nparray, decribed in the local frame of self.jnts[tcp_jnt_id], single value or list
        :param tcp_loc_rotmat: 3x3 nparray, decribed in the local frame of self.jnts[tcp_jnt_id], single value or list
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
            start_conf = self.jlc_object.getjntvalues()
            # return numik(rjlinstance, tgt_pos, tgt_rotmat, seed_jnt_values=seed_jnt_values, tcp_jnt_id=tcp_jnt_id, tcp_loc_pos=tcp_loc_pos, tcp_loc_rotmat=tcp_loc_rotmat)
        else:
            tgt_pos = tcp_gl_pos + deltapos
            tgt_rotmat = np.dot(deltarotmat, tcp_gl_rotmat)
            start_conf = self.jlc_object.getjntvalues()
        return self.numik(tgt_pos, tgt_rotmat, start_conf=start_conf, tcp_jnt_id=tcp_jnt_id, tcp_loc_pos=tcp_loc_pos,
                          tcp_loc_rotmat=tcp_loc_rotmat)
