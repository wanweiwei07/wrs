import math
import numpy as np
import basis.robotmath as rm
import warnings as wns


class JntLnksIK(object):

    def __init__(self, jlobject, wlnratio=.15):
        self.jlobject = jlobject
        # IK macros
        wt_pos = 0.628  # 0.628m->1 == 0.01->0.00628m
        wt_agl = 1 / (math.pi * math.pi)  # pi->1 == 0.01->0.18degree
        self.ws_wtlist = [wt_pos, wt_pos, wt_pos, wt_agl, wt_agl, wt_agl]
        # maximum reach
        self.max_rng = 2

        # extract min max for quick access
        self.jmvmin = np.zeros(jlobject.ndof)
        self.jmvmax = np.zeros(jlobject.ndof)
        counter = 0
        for id in jlobject.tgtjnts:
            self.jmvmin[counter] = jlobject.jnts[id]['rngmin']
            self.jmvmax[counter] = jlobject.jnts[id]['rngmax']
            counter += 1
        self.jmvrng = self.jmvmax - self.jmvmin
        self.jmvmin_threshhold = self.jmvmin + self.jmvrng * wlnratio
        self.jmvmax_threshhold = self.jmvmax - self.jmvrng * wlnratio

    def _jacobian_sgl(self, tcp_jntid):
        """
        compute the jacobian matrix of a rjlinstance
        only a single tcp_jntid is acceptable
        :param tcp_jntid: the joint id where the tool center pose is specified, single vlaue
        :return: jmat, a 6xn nparray
        author: weiwei
        date: 20161202, 20200331, 20200706
        """
        jmat = np.zeros((6, len(self.jlobject.tgtjnts)))
        counter = 0
        for jid in self.jlobject.tgtjnts:
            grax = self.jlobject.jnts[jid]["gl_motionax"]
            if self.jlobject.jnts[jid]["type"] == "revolute":
                diffq = self.jlobject.jnts[tcp_jntid]["gl_posq"] - self.jlobject.jnts[jid]["gl_posq"]
                jmat[:3, counter] = np.cross(grax, diffq)
                jmat[3:6, counter] = grax
            if self.jlobject.jnts[jid]["type"] == "prismatic":
                jmat[:3, counter] = grax
            counter += 1
            if jid == tcp_jntid:
                break
        return jmat

    def _wln_weightmat(self, jntvalues):
        """
        get the wln weightmat
        :param jntvalues:
        :return:
        author: weiwei
        date: 20201126
        """
        wtmat = np.ones(self.jlobject.ndof)
        # min damping interval
        selection = (jntvalues - self.jmvmin_threshhold < 0)
        diff_selected = self.jmvmin_threshhold[selection] - jntvalues[selection]
        wtmat[selection] = -2 * np.power(diff_selected, 3) + 3 * np.power(diff_selected, 2)
        # max damping interval
        selection = (jntvalues - self.jmvmax_threshhold > 0)
        diff_selected = jntvalues[selection] - self.jmvmax_threshhold[selection]
        wtmat[selection] = -2 * np.power(diff_selected, 3) + 3 * np.power(diff_selected, 2)
        wtmat[jntvalues >= self.jmvmax] = 1e-6
        wtmat[jntvalues <= self.jmvmin] = 1e-6
        return np.diag(wtmat)

    def jacobian(self, tcp_jntid):
        """
        compute the jacobian matrix of a rjlinstance
        multiple tcp_jntid acceptable
        :param tcp_jntid: the joint id where the tool center pose is specified, single vlaue or list
        :return: jmat, a sum(len(option))xn nparray
        author: weiwei
        date: 20161202, 20200331, 20200706, 20201114
        """
        if isinstance(tcp_jntid, list):
            jmat = np.zeros((6 * (len(tcp_jntid)), len(self.jlobject.tgtjnts)))
            for i, this_tcp_jntid in enumerate(tcp_jntid):
                jmat[6 * i:6 * i + 6, :] = self._jacobian_sgl(this_tcp_jntid)
            return jmat
        else:
            return self._jacobian_sgl(tcp_jntid)

    def manipulability(self, tcp_jntid):
        """
        compute the yoshikawa manipulability of the rjlinstance
        :param tcp_jntid: the joint id where the tool center pose is specified, single vlaue or list
        :return:
        author: weiwei
        date: 20200331
        """
        jmat = self.jacobian(tcp_jntid)
        return math.sqrt(np.linalg.det(np.dot(jmat, jmat.transpose())))

    def manipulability_axmat(self, tcp_jntid):
        """
        compute the yasukawa manipulability of the rjlinstance
        :param tcp_jntid: the joint id where the tool center pose is specified, single vlaue or list
        :return: axmat with each column being the manipulability
        """
        armjac = self.jacobian(tcp_jntid)
        jjt = np.dot(armjac, armjac.T)
        pcv, pcaxmat = np.linalg.eig(jjt)
        # only keep translation
        axmat = np.eye(3)
        axmat[:, 0] = np.sqrt(pcv[0]) * pcaxmat[:3, 0]
        axmat[:, 1] = np.sqrt(pcv[1]) * pcaxmat[:3, 1]
        axmat[:, 2] = np.sqrt(pcv[2]) * pcaxmat[:3, 2]
        return axmat

    def get_globaltcp(self, tcp_jntid, tcp_loc_pos, tcp_loc_rotmat):
        """
        Get the global tool center pose given tcp_jntid, tcp_loc_pos, tcp_loc_rotmat
        tcp_jntid, tcp_loc_pos, tcp_loc_rotmat are the tool center pose parameters. They are
        used for temporary computation, the self.tcp_xxx parameters will not be changed
        in case None is provided, the self.tcp_jntid, self.tcp_loc_pos, self.tcp_loc_rotmat will be used
        :param tcp_jntid: a joint ID in the self.tgtjnts
        :param tcp_loc_pos: 1x3 nparray, decribed in the local frame of self.jnts[tcp_jntid], single value or list
        :param tcp_loc_rotmat: 3x3 nparray, decribed in the local frame of self.jnts[tcp_jntid], single value or list
        :return: a single value or a list depending on the input
        author: weiwei
        date: 20200706
        """
        if tcp_jntid is None:
            tcp_jntid = self.jlobject.tcp_jntid
        if tcp_loc_pos is None:
            tcp_loc_pos = self.jlobject.tcp_loc_pos
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.jlobject.tcp_loc_rotmat
        if isinstance(tcp_jntid, list):
            returnposlist = []
            returnrotmatlist = []
            for i, jid in enumerate(tcp_jntid):
                tcp_globalpos = np.dot(self.jlobject.jnts[jid]["gl_rotmatq"], tcp_loc_pos[i]) + \
                                self.jlobject.jnts[jid]["gl_posq"]
                tcp_globalrotmat = np.dot(self.jlobject.jnts[jid]["gl_rotmatq"], tcp_loc_rotmat[i])
                returnposlist.append(tcp_globalpos)
                returnrotmatlist.append(tcp_globalrotmat)
            return [returnposlist, returnrotmatlist]
        else:
            tcp_globalpos = np.dot(self.jlobject.jnts[tcp_jntid]["gl_rotmatq"], tcp_loc_pos) + \
                            self.jlobject.jnts[tcp_jntid]["gl_posq"]
            tcp_globalrotmat = np.dot(self.jlobject.jnts[tcp_jntid]["gl_rotmatq"], tcp_loc_rotmat)
            return [tcp_globalpos, tcp_globalrotmat]

    def tcperror(self, tgtpos, tgtrot, tcp_jntid, tcp_loc_pos, tcp_loc_rotmat):
        """
        compute the error between the rjlinstance's end and tgtpos, tgtrot
        NOTE: if list, len(tgtpos)=len(tgtrot) <= len(tcp_jntid)=len(tcp_loc_pos)=len(tcp_loc_rotmat)
        :param tgtpos: the position vector of the goal (could be a single value or a list of jntid)
        :param tgtrot: the rotation matrix of the goal (could be a single value or a list of jntid)
        :param tcp_jntid: a joint ID in the self.tgtjnts
        :param tcp_loc_pos: 1x3 nparray, decribed in the local frame of self.jnts[tcp_jntid], single value or list
        :param tcp_loc_rotmat: 3x3 nparray, decribed in the local frame of self.jnts[tcp_jntid], single value or list
        :return: a 1x6 nparray where the first three indicates the displacement in pos,
                    the second three indictes the displacement in rot
        author: weiwei
        date: 20180827, 20200331, 20200705
        """
        tcp_globalpos, tcp_globalrotmat = self.get_globaltcp(tcp_jntid, tcp_loc_pos, tcp_loc_rotmat)
        if isinstance(tgtpos, list):
            deltapw = np.zeros(6 * len(tgtpos))
            for i, thistgtpos in enumerate(tgtpos):
                deltapw[6 * i:6 * i + 3] = (thistgtpos - tcp_globalpos[i])
                deltapw[6 * i + 3:6 * i + 6] = rm.deltaw_between_rotmat(tgtrot[i], tcp_globalrotmat[i].T)
            return deltapw
        else:
            deltapw = np.zeros(6)
            deltapw[0:3] = (tgtpos - tcp_globalpos)
            deltapw[3:6] = rm.deltaw_between_rotmat(tgtrot, tcp_globalrotmat.T)
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
        for id in self.jlobject.tgtjnts:
            if self.jlobject.jnts[id]["type"] is "revolute":
                if self.jlobject.jnts[id]["rngmax"] - self.jlobject.jnts[id]["rngmin"] >= math.pi * 2:
                    rm.regulate_angle(self.jlobject.jnts[id]["rngmin"], self.jlobject.jnts[id]["rngmax"],
                                      self.jlobject.jnts[id]["movement"])
            counter += 1

    def check_jntsrange_drag(self, jntvalues):
        """
        check if the given jntvalues is inside the oeprating range
        The joint values out of range will be pulled back to their maxima
        :param jntvalues: a 1xn numpy ndarray
        :return: Two parameters, one is true or false indicating if the joint values are inside the range or not
                The other is the joint values after dragging.
                If the joints were not dragged, the same joint values will be returned
        author: weiwei
        date: 20161205
        """
        counter = 0
        isdragged = np.zeros_like(jntvalues)
        jntvaluesdragged = jntvalues.copy()
        for id in self.jlobject.tgtjnts:
            if self.jlobject.jnts[id]["type"] == "revolute":
                if self.jlobject.jnts[id]["rngmax"] - self.jlobject.jnts[id]["rngmin"] < math.pi * 2:
                    # if jntvalues[counter] < jlinstance.jnts[id]["rngmin"]:
                    #     isdragged[counter] = 1
                    #     jntvaluesdragged[counter] = jlinstance.jnts[id]["rngmin"]
                    # elif jntvalues[counter] > jlinstance.jnts[id]["rngmax"]:
                    #     isdragged[counter] = 1
                    #     jntvaluesdragged[counter] = jlinstance.jnts[id]["rngmax"]
                    print("Drag revolute")
                    if jntvalues[counter] < self.jlobject.jnts[id]["rngmin"] or jntvalues[counter] > \
                            self.jlobject.jnts[id]["rngmax"]:
                        isdragged[counter] = 1
                        jntvaluesdragged[counter] = (self.jlobject.jnts[id]["rngmax"] + self.jlobject.jnts[id][
                            "rngmin"]) / 2
            elif self.jlobject.jnts[id]["type"] == "prismatic":  # prismatic
                # if jntvalues[counter] < jlinstance.jnts[id]["rngmin"]:
                #     isdragged[counter] = 1
                #     jntvaluesdragged[counter] = jlinstance.jnts[id]["rngmin"]
                # elif jntvalues[counter] > jlinstance.jnts[id]["rngmax"]:
                #     isdragged[counter] = 1
                #     jntvaluesdragged[counter] = jlinstance.jnts[id]["rngmax"]
                print("Drag prismatic")
                if jntvalues[counter] < self.jlobject.jnts[id]["rngmin"] or jntvalues[counter] > \
                        self.jlobject.jnts[id]["rngmax"]:
                    isdragged[counter] = 1
                    jntvaluesdragged[counter] = (self.jlobject.jnts[id]["rngmax"] + self.jlobject.jnts[id][
                        "rngmin"]) / 2
        return isdragged, jntvaluesdragged

    def numik(self, tgtpos, tgtrot, startconf=None, tcp_jntid=None, tcp_loc_pos=None, tcp_loc_rotmat=None,
              local_minima="accept", toggle_debug=False):
        """
        solveik numerically using the Levenberg-Marquardt Method
        the details of this method can be found in: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
        NOTE: if list, len(tgtpos)=len(tgtrot) <= len(tcp_jntid)=len(tcp_loc_pos)=len(tcp_loc_rotmat)
        :param tgtpos: the position of the goal, 1-by-3 numpy ndarray
        :param tgtrot: the orientation of the goal, 3-by-3 numpyndarray
        :param startconf: the starting configuration used in the numerical iteration
        :param tcp_jntid: a joint ID in the self.tgtjnts
        :param tcp_loc_pos: 1x3 nparray, decribed in the local frame of self.jnts[tcp_jntid], single value or list
        :param tcp_loc_rotmat: 3x3 nparray, decribed in the local frame of self.jnts[tcp_jntid], single value or list
        :param local_minima: what to do at local minima: "accept", "randomrestart", "end"
        :return: a 1xn numpy ndarray
        author: weiwei
        date: 20180203, 20200328
        """
        deltapos = tgtpos - self.jlobject.jnts[0]["gl_pos0"]
        if np.linalg.norm(deltapos) > self.max_rng:
            wns.WarningMessage("The goal is outside maximum range!")
            return None
        if tcp_jntid is None:
            tcp_jntid = self.jlobject.tcp_jntid
        if tcp_loc_pos is None:
            tcp_loc_pos = self.jlobject.tcp_loc_pos
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.jlobject.tcp_loc_rotmat
        # trim list
        if isinstance(tgtpos, list):
            tcp_jntid = tcp_jntid[0:len(tgtpos)]
            tcp_loc_pos = tcp_loc_pos[0:len(tgtpos)]
            tcp_loc_rotmat = tcp_loc_rotmat[0:len(tgtpos)]
        elif isinstance(tcp_jntid, list):
            tcp_jntid = tcp_jntid[0]
            tcp_loc_pos = tcp_loc_pos[0]
            tcp_loc_rotmat = tcp_loc_rotmat[0]
        jntvalues_bk = self.jlobject.get_jntvalues()
        jntvalues_iter = self.jlobject.homeconf if startconf is None else startconf.copy()
        self.jlobject.fk(jntvalues_iter)
        jntvalues_ref = jntvalues_iter.copy()

        if isinstance(tcp_jntid, list):
            diaglist = []
            for i in tcp_jntid:
                diaglist += self.ws_wtlist
            ws_wtdiagmat = np.diag(diaglist)
        else:
            ws_wtdiagmat = np.diag(self.ws_wtlist)
        # sqrtinv_ws_wtdiagmat = np.linalg.inv(np.diag(np.sqrt(np.diag(ws_wtdiagmat))))

        if toggle_debug:
            if "jlm" not in dir():
                import robotsim._kinematics.jlchainmesh as jlm
            if "plt" not in dir():
                import matplotlib.pyplot as plt
            # jlmgen = jlm.JntLnksMesh()
            dqbefore = []
            dqcorrected = []
            dqnull = []
            ajpath = []
        errnormlast = 0.0
        errnormmax = 0.0
        for i in range(100):
            jmat = self.jacobian(tcp_jntid)
            err = self.tcperror(tgtpos, tgtrot, tcp_jntid, tcp_loc_pos, tcp_loc_rotmat)
            errnorm = err.T.dot(ws_wtdiagmat).dot(err)
            # err = .05 / errnorm * err if errnorm > .05 else err
            if errnorm > errnormmax:
                errnormmax = errnorm
            if toggle_debug:
                print(errnorm)
                ajpath.append(self.jlobject.get_jntvalues())
            if errnorm < 1e-6:
                if toggle_debug:
                    fig = plt.figure()
                    axbefore = fig.add_subplot(411)
                    axbefore.set_title("Original dq")
                    axnull = fig.add_subplot(412)
                    axnull.set_title("dqref on Null space")
                    axcorrec = fig.add_subplot(413)
                    axcorrec.set_title("Minimized dq")
                    axaj = fig.add_subplot(414)
                    axbefore.plot(dqbefore)
                    axnull.plot(dqnull)
                    axcorrec.plot(dqcorrected)
                    axaj.plot(ajpath)
                    plt.show()
                # self.regulate_jnts()
                jntvalues_return = self.jlobject.getjntvalues()
                self.jlobject.fk(jntvalues_bk)
                return jntvalues_return
            else:
                # judge local minima
                if abs(errnorm - errnormlast) < 1e-12:
                    if toggle_debug:
                        fig = plt.figure()
                        axbefore = fig.add_subplot(411)
                        axbefore.set_title("Original dq")
                        axnull = fig.add_subplot(412)
                        axnull.set_title("dqref on Null space")
                        axcorrec = fig.add_subplot(413)
                        axcorrec.set_title("Minimized dq")
                        axaj = fig.add_subplot(414)
                        axbefore.plot(dqbefore)
                        axnull.plot(dqnull)
                        axcorrec.plot(dqcorrected)
                        axaj.plot(ajpath)
                        plt.show()
                    if local_minima == "accept":
                        wns.warn(
                            "Bypassing local minima! The return value is a local minima, rather than the exact IK result.")
                        jntvalues_return = self.jlobject.get_jntvalues()
                        self.jlobject.fk(jntvalues_bk)
                        return jntvalues_return
                    elif local_minima == "randomrestart":
                        wns.warn("Local Minima! Random restart at local minima!")
                        jntvalues_iter = self.jlobject.rand_conf()
                        self.jlobject.fk(jntvalues_iter)
                        continue
                    else:
                        print("No feasible IK solution!")
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
                    dampercoeff = 1e-3*errnorm + 1e-6  # a non-zero regulation coefficient
                    # -- lft moore-penrose inverse --
                    ## jtj = armjac.T.dot(armjac)
                    ## regulator = regcoeff*np.identity(jtj.shape[0])
                    ## jstar = np.linalg.inv(jtj+regulator).dot(armjac.T)
                    ## dq = jstar.dot(err)
                    # -- rgt moore-penrose inverse --
                    # # jjt
                    # jjt = jmat.dot(jmat.T)
                    # damper = dampercoeff * np.identity(jjt.shape[0])
                    # jsharp = jmat.T.dot(np.linalg.inv(jjt + damper))
                    # weighted jjt
                    qs_wtdiagmat = self._wln_weightmat(jntvalues_iter)
                    winv_jt = np.linalg.inv(qs_wtdiagmat).dot(jmat.T)
                    j_winv_jt = jmat.dot(winv_jt)
                    damper = dampercoeff * np.identity(j_winv_jt.shape[0])
                    jsharp = winv_jt.dot(np.linalg.inv(j_winv_jt + damper))
                    dq = .1 * jsharp.dot(err)
                    # dq = rm.regulate_angle(-math.pi, math.pi, dq)
                    # dq = Jsharp dx+(I-Jsharp J)dq0
                    dqref = (jntvalues_ref - jntvalues_iter)
                    dqref_on_ns = (np.identity(dqref.shape[0]) - jsharp.dot(jmat)).dot(dqref)
                    # dqref_on_ns = rm.regulate_angle(-math.pi, math.pi, dqref_on_ns)
                    dq_minimized = dq + dqref_on_ns
                    if toggle_debug:
                        dqbefore.append(dq)
                        dqcorrected.append(dq_minimized)
                        dqnull.append(dqref_on_ns)
                jntvalues_iter += dq_minimized  # translation problem
                # isdragged, jntvalues_iter = self.check_jntsrange_drag(jntvalues_iter)
                # print(jntvalues_iter)
                self.jlobject.fk(jntvalues_iter)
                # if toggle_debug:
                #     jlmgen.gensnp(jlinstance, tcp_jntid=tcp_jntid, tcp_loc_pos=tcp_loc_pos,
                #                   tcp_loc_rotmat=tcp_loc_rotmat, togglejntscs=True).reparentTo(base.render)
            errnormlast = errnorm
        if toggle_debug:
            fig = plt.figure()
            axbefore = fig.add_subplot(411)
            axbefore.set_title("Original dq")
            axnull = fig.add_subplot(412)
            axnull.set_title("dqref on Null space")
            axcorrec = fig.add_subplot(413)
            axcorrec.set_title("Minimized dq")
            axaj = fig.add_subplot(414)
            axbefore.plot(dqbefore)
            axnull.plot(dqnull)
            axcorrec.plot(dqcorrected)
            axaj.plot(ajpath)
            plt.show()
            self.jlobject.gen_stickmodel(tcp_jntid=tcp_jntid, tcp_loc_pos=tcp_loc_pos,
                                         tcp_loc_rotmat=tcp_loc_rotmat, togglejntscs=True).attach_to(base)
            base.run()
        self.jlobject.fk(jntvalues_bk)
        return None

    def numik_rel(self, deltapos, deltarotmat, tcp_jntid=None, tcp_loc_pos=None, tcp_loc_rotmat=None):
        """
        add deltapos, deltarotmat to the current end
        :param deltapos:
        :param deltarotmat:
        :param tcp_jntid: a joint ID in the self.tgtjnts
        :param tcp_loc_pos: 1x3 nparray, decribed in the local frame of self.jnts[tcp_jntid], single value or list
        :param tcp_loc_rotmat: 3x3 nparray, decribed in the local frame of self.jnts[tcp_jntid], single value or list
        :return:
        author: weiwei
        date: 20170412, 20200331
        """
        tcp_globalpos, tcp_globalrotmat = self.get_globaltcp(tcp_jntid, tcp_loc_pos, tcp_loc_rotmat)
        if isinstance(tcp_jntid, list):
            tgtpos = []
            tgtrotmat = []
            for i, jid in enumerate(tcp_jntid):
                tgtpos.append(tcp_globalpos[i] + deltapos[i])
                tgtrotmat.append(np.dot(deltarotmat, tcp_globalrotmat[i]))
            startconf = self.jlobject.getjntvalues()
            # return numik(rjlinstance, tgtpos, tgtrotmat, startconf=startconf, tcp_jntid=tcp_jntid, tcp_loc_pos=tcp_loc_pos, tcp_loc_rotmat=tcp_loc_rotmat)
        else:
            tgtpos = tcp_globalpos + deltapos
            tgtrotmat = np.dot(deltarotmat, tcp_globalrotmat)
            startconf = self.jlobject.getjntvalues()
        return self.numik(tgtpos, tgtrotmat, startconf=startconf, tcp_jntid=tcp_jntid, tcp_loc_pos=tcp_loc_pos,
                          tcp_loc_rotmat=tcp_loc_rotmat)
