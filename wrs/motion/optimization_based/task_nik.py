import math
import numpy as np
import warnings as wns
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm


class NIK(object):

    def __init__(self, robot, component_name, wln_ratio=.05):
        self.rbt = robot
        self.component_name = component_name
        self.jlc_object = self.rbt.manipulator_dict[component_name].jlc
        self.wln_ratio = wln_ratio
        # workspace bound
        self.max_rng = 2.0 # meter
        # IK macros
        wt_pos = 0.628  # 0.628m->1 == 0.01->0.00628m
        wt_agl = 1 / (math.pi * math.pi)  # pi->1 == 0.01->0.18degree
        self.ws_wtlist = [wt_pos, wt_pos, wt_pos, wt_agl, wt_agl, wt_agl]
        # maximum reach
        self.jnt_bounds = np.array(self.rbt.get_jnt_rngs(component_name))
        # extract min max for quick access
        self.jmvmin = self.jnt_bounds[:,0]
        self.jmvmax = self.jnt_bounds[:,1]
        self.jmvrng = self.jmvmax-self.jmvmin
        self.jmvmin_threshhold = self.jmvmin + self.jmvrng * self.wln_ratio
        self.jmvmax_threshhold = self.jmvmax - self.jmvrng * self.wln_ratio

    def set_jlc(self, jlc_name):
        self.component_name=jlc_name
        self.jnt_bounds = np.array(self.rbt.get_jnt_rngs(jlc_name))
        # extract min max for quick access
        self.jmvmin = self.jnt_bounds[:,0]
        self.jmvmax = self.jnt_bounds[:,1]
        self.jmvrng = self.jmvmax-self.jmvmin
        self.jmvmin_threshhold = self.jmvmin + self.jmvrng * self.wln_ratio
        self.jmvmax_threshhold = self.jmvmax - self.jmvrng * self.wln_ratio

    def jacobian(self, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat):
        """
        compute the jacobian matrix of a rjlinstance
        only a single tcp_joint_id is acceptable
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
            if self.jlc_object.jnts[jid]["end_type"] == 'revolute':
                diffq = tcp_gl_pos - self.jlc_object.jnts[jid]["gl_posq"]
                j[:3, counter] = np.cross(grax, diffq)
                j[3:6, counter] = grax
            if self.jlc_object.jnts[jid]["end_type"] == 'prismatic':
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
        wtmat = np.ones(self.jlc_object.n_dof)
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

    def manipulability(self, tcp_jnt_id):
        """
        compute the yoshikawa manipulability of the rjlinstance
        :param tcp_jnt_id: the joint id where the tool center pose is specified, single vlaue or list
        :return:
        author: weiwei
        date: 20200331
        """
        j = self.jacobian(tcp_jnt_id)
        return math.sqrt(np.linalg.det(np.dot(j, j.transpose())))

    def manipulability_axmat(self, tcp_jnt_id):
        """
        compute the yasukawa manipulability of the rjlinstance
        :param tcp_jnt_id: the joint id where the tool center pose is specified, single vlaue or list
        :return: axes_mat with each column being the manipulability
        """
        armjac = self.jacobian(tcp_jnt_id)
        jjt = np.dot(armjac, armjac.T)
        pcv, pcaxmat = np.linalg.eig(jjt)
        # only keep translation
        axmat = np.eye(3)
        axmat[:, 0] = np.sqrt(pcv[0]) * pcaxmat[:3, 0]
        axmat[:, 1] = np.sqrt(pcv[1]) * pcaxmat[:3, 1]
        axmat[:, 2] = np.sqrt(pcv[2]) * pcaxmat[:3, 2]
        return axmat

    def get_gl_tcp(self, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat):
        """
        Get the global tool center pose given tcp_joint_id, _loc_flange_pos, _loc_flange_rotmat
        tcp_joint_id, _loc_flange_pos, _loc_flange_rotmat are the tool center pose parameters. They are
        used for temporary computation, the self.tcp_xxx parameters will not be changed
        in case None is provided, the self.tcp_joint_id, self._loc_flange_pos, self._loc_flange_rotmat will be used
        :param tcp_jnt_id: a joint ID in the self.tgtjnts
        :param tcp_loc_pos: 1x3 nparray, decribed in the local frame of self.joints[tcp_joint_id], single value or list
        :param tcp_loc_rotmat: 3x3 nparray, decribed in the local frame of self.joints[tcp_joint_id], single value or list
        :return: a single value or a list depending on the input
        author: weiwei
        date: 20200706
        """
        if tcp_jnt_id is None:
            tcp_jnt_id = self.jlc_object.flange_jnt_id
        if tcp_loc_pos is None:
            tcp_loc_pos = self.jlc_object._loc_flange_pos
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.jlc_object._loc_flange_rotmat
        tcp_gl_pos = np.dot(self.jlc_object.jnts[tcp_jnt_id]["gl_rotmatq"], tcp_loc_pos) + \
                     self.jlc_object.jnts[tcp_jnt_id]["gl_posq"]
        tcp_gl_rotmat = np.dot(self.jlc_object.jnts[tcp_jnt_id]["gl_rotmatq"], tcp_loc_rotmat)
        return tcp_gl_pos, tcp_gl_rotmat

    def tcp_error(self, tgt_pos, tgt_rot, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat):
        """
        compute the error between the rjlinstance's end_type and tgt_pos, tgt_rotmat
        NOTE: if list, len(tgt_pos)=len(tgt_rotmat) <= len(tcp_joint_id)=len(_loc_flange_pos)=len(_loc_flange_rotmat)
        :param tgt_pos: the position vector of the goal (could be a single value or a list of jntid)
        :param tgt_rot: the rotation matrix of the goal (could be a single value or a list of jntid)
        :param tcp_jnt_id: a joint ID in the self.tgtjnts
        :param tcp_loc_pos: 1x3 nparray, decribed in the local frame of self.joints[tcp_joint_id], single value or list
        :param tcp_loc_rotmat: 3x3 nparray, decribed in the local frame of self.joints[tcp_joint_id], single value or list
        :return: a 1x6 nparray where the first three indicates the displacement in pos,
                    the second three indictes the displacement in rotmat
        author: weiwei
        date: 20180827, 20200331, 20200705
        """
        tcp_globalpos, tcp_globalrotmat = self.get_gl_tcp(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
        deltapw = np.zeros(6)
        deltapw[0:3] = (tgt_pos - tcp_globalpos)
        deltapw[3:6] = rm.delta_w_between_rotmat(tgt_rot, tcp_globalrotmat.T)
        return deltapw

    def regulate_jnts(self):
        """
        check if the given jnt_values is inside the oeprating range
        The joint values out of range will be pulled back to their maxima
        :return: Two parameters, one is true or false indicating if the joint values are inside the range or not
                The other is the joint values after dragging.
                If the joints were not dragged, the same joint values will be returned
        author: weiwei
        date: 20161205
        """
        counter = 0
        for id in self.jlc_object.tgtjnts:
            if self.jlc_object.jnts[id]["end_type"] == 'revolute':
                if self.jlc_object.jnts[id]['motion_range'][1] - self.jlc_object.jnts[id]['motion_range'][0] >= math.pi * 2:
                    rm.regulate_angle(self.jlc_object.jnts[id]['motion_range'][0], self.jlc_object.jnts[id]['motion_range'][1],
                                      self.jlc_object.jnts[id]["movement"])
            counter += 1

    def check_jntranges_drag(self, jntvalues):
        """
        check if the given jnt_values is inside the oeprating range
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
        for id in self.jlc_object.tgtjnts:
            if self.jlc_object.jnts[id]["end_type"] == 'revolute':
                if self.jlc_object.jnts[id]['motion_range'][1] - self.jlc_object.jnts[id]['motion_range'][0] < math.pi * 2:
                    print("Drag revolute")
                    if jntvalues[counter] < self.jlc_object.jnts[id]['motion_range'][0] or jntvalues[counter] > \
                            self.jlc_object.jnts[id]['motion_range'][1]:
                        isdragged[counter] = 1
                        jntvaluesdragged[counter] = (self.jlc_object.jnts[id]['motion_range'][1] + self.jlc_object.jnts[id][
                            'motion_range'][0]) / 2
            elif self.jlc_object.jnts[id]["end_type"] == 'prismatic':  # prismatic
                print("Drag prismatic")
                if jntvalues[counter] < self.jlc_object.jnts[id]['motion_range'][0] or jntvalues[counter] > \
                        self.jlc_object.jnts[id]['motion_range'][1]:
                    isdragged[counter] = 1
                    jntvaluesdragged[counter] = (self.jlc_object.jnts[id]['motion_range'][1] + self.jlc_object.jnts[id][
                        "rngmin"]) / 2
        return isdragged, jntvaluesdragged

    def num_ik(self,
               tgt_pos,
               tgt_rot,
               seed_jnt_values=None,
               tcp_jnt_id=None,
               tcp_loc_pos=None,
               tcp_loc_rotmat=None,
               local_minima="accept",
               toggle_debug=False):
        """
        solveik numerically using the Levenberg-Marquardt Method
        the details of this method can be found in: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
        NOTE: if list, len(tgt_pos)=len(tgt_rotmat) <= len(tcp_joint_id)=len(_loc_flange_pos)=len(_loc_flange_rotmat)
        :param tgt_pos: the position of the goal, 1-by-3 numpy ndarray
        :param tgt_rot: the orientation of the goal, 3-by-3 numpyndarray
        :param seed_jnt_values: the starting configuration used in the numerical iteration
        :param tcp_jnt_id: a joint ID in the self.tgtjnts
        :param tcp_loc_pos: 1x3 nparray, decribed in the local frame of self.joints[tcp_joint_id], single value or list
        :param tcp_loc_rotmat: 3x3 nparray, decribed in the local frame of self.joints[tcp_joint_id], single value or list
        :param local_minima: what to do at local minima: "accept", "randomrestart", "end_type"
        :return: a 1xn numpy ndarray
        author: weiwei
        date: 20180203, 20200328
        """
        deltapos = tgt_pos - self.jlc_object.jnts[0]['gl_pos0']
        if np.linalg.norm(deltapos) > self.max_rng:
            wns.WarningMessage("The goal is outside maximum range!")
            return None
        if tcp_jnt_id is None:
            tcp_jnt_id = self.jlc_object.flange_jnt_id
        if tcp_loc_pos is None:
            tcp_loc_pos = self.jlc_object._loc_flange_pos
            print(self.jlc_object._loc_flange_pos)
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.jlc_object._loc_flange_rotmat
        jntvalues_bk = self.jlc_object.get_jnt_values()
        jntvalues_iter = self.jlc_object.home_conf if seed_jnt_values is None else seed_jnt_values.copy()
        self.jlc_object.fk(joint_values=jntvalues_iter)
        jntvalues_ref = jntvalues_iter.copy()
        ws_wtdiagmat = np.diag(self.ws_wtlist)
        if toggle_debug:
            if "lib_jlm" not in dir():
                pass
            if "plt" not in dir():
                import matplotlib.pyplot as plt
            # jlmgen = lib_jlm.JntLnksMesh()
            dqbefore = []
            dqcorrected = []
            dqnull = []
            ajpath = []
        errnormlast = 0.0
        errnormmax = 0.0
        for i in range(1000):
            j = self.jacobian(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
            j1 = j[:3, :]
            j2 = j[3:6, :]
            err = self.tcp_error(tgt_pos, tgt_rot, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
            err_pos = err[:3]
            err_rot = err[3:6]
            errnorm_pos = err_pos.T.dot(err_pos)
            errnorm_rot = np.linalg.norm(err_rot)
            # if errnorm_rot < math.pi/6:
            #     err_rot = np.zeros(3)
                # errnorm_rot = 0
            errnorm = err.T.dot(ws_wtdiagmat).dot(err)
            # err = .05 / errnorm * err if errnorm > .05 else err
            if errnorm > errnormmax:
                errnormmax = errnorm
            if toggle_debug:
                print(errnorm_pos, errnorm_rot, errnorm)
                ajpath.append(self.jlc_object.get_jnt_values())
            if errnorm_pos < 1e-6 and errnorm_rot < math.pi/6:
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
                # self.regulate_jnts()
                jntvalues_return = self.jlc_object.get_jnt_values()
                self.jlc_object.fk(joint_values=jntvalues_bk)
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
                        wns.warn(
                            'Bypassing local minima! The return value is a local minima, rather than the exact IK result.')
                        jntvalues_return = self.jlc_object.get_jnt_values()
                        self.jlc_object.fk(jntvalues_bk)
                        return jntvalues_return
                    elif local_minima == 'randomrestart':
                        wns.warn('Local Minima! Random restart at local minima!')
                        jntvalues_iter = self.jlc_object.rand_conf()
                        self.jlc_object.fk(jntvalues_iter)
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
                    dampercoeff = 1e-3*errnorm + 1e-6  # a non-zero regulation coefficient
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
                    qs_wtdiagmat = self._wln_weightmat(jntvalues_iter)
                    # WLN
                    winv_j1t = np.linalg.inv(qs_wtdiagmat).dot(j1.T)
                    j1_winv_j1t = j1.dot(winv_j1t)
                    damper = dampercoeff * np.identity(j1_winv_j1t.shape[0])
                    j1sharp = winv_j1t.dot(np.linalg.inv(j1_winv_j1t + damper))
                    n1 = np.identity(jntvalues_ref.shape[0]) - j1sharp.dot(j1)
                    j2n1 = j2.dot(n1)
                    winv_j2n1t = np.linalg.inv(qs_wtdiagmat).dot(j2n1.T)
                    j2n1_winv_j2n1t = j2n1.dot(winv_j2n1t)
                    damper = dampercoeff * np.identity(j2n1_winv_j2n1t.shape[0])
                    j2n1sharp = winv_j2n1t.dot(np.linalg.inv(j2n1_winv_j2n1t + damper))
                    err_pos = .1 * err_pos
                    # if errnorm_rot == 0:
                    dq = j1sharp.dot(err_pos)
                    dqref = (jntvalues_ref - jntvalues_iter)
                    dqref_on_ns = (np.identity(jntvalues_ref.shape[0]) - j1sharp.dot(j1)).dot(dqref)
                    dq_minimized = dq + dqref_on_ns
                    # else:
                    # err_rot = .1 * err_rot
                    # dq = j1sharp.dot(err_pos)+j2n1sharp.dot(err_rot-j2.dot(j1sharp.dot(err_pos)))
                    # dqref_on_ns = np.zeros(jntvalues_ref.shape[0])
                    # dq_minimized = dq
                    if toggle_debug:
                        dqbefore.append(dq)
                        dqcorrected.append(dq_minimized)
                        dqnull.append(dqref_on_ns)
                jntvalues_iter += dq_minimized  # translation problem
                # isdragged, jntvalues_iter = self.check_jntsrange_drag(jntvalues_iter)
                # print(jntvalues_iter)
                self.jlc_object.fk(joint_values=jntvalues_iter)
                # if toggle_dbg:
                #     jlmgen.gensnp(jlinstance, tcp_joint_id=tcp_joint_id, _loc_flange_pos=_loc_flange_pos,
                #                   _loc_flange_rotmat=_loc_flange_rotmat, togglejntscs=True).reparentTo(base.render)
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
            self.jlc_object.gen_stickmodel(toggle_jnt_frames=True).attach_to(base)
            base.run()
        self.jlc_object.fk(jntvalues_bk)
        wns.warn('Failed to solve the IK, returning None.')
        return None

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
        tcp_globalpos, tcp_globalrotmat = self.get_gl_tcp(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)
        tgt_pos = tcp_globalpos + deltapos
        tgt_rotmat = np.dot(deltarotmat, tcp_globalrotmat)
        start_conf = self.jlc_object.getjntvalues()
        return self.numik(tgt_pos, tgt_rotmat, start_conf=start_conf, tcp_jnt_id=tcp_jnt_id, tcp_loc_pos=tcp_loc_pos,
                          tcp_loc_rotmat=tcp_loc_rotmat)


if __name__ == '__main__':
    import time
    import wrs.visualization.panda.world as wd
    import wrs.robot_sim.robots.yumi.yumi as ym

    base = wd.World(cam_pos=[1.5, 0, 3], lookat_pos=[0, 0, .5])
    mgm.gen_frame().attach_to(base)
    yumi_instance = ym.Yumi(enable_cc=True)
    component_name= 'rgt_arm'
    tgt_pos = np.array([.5, -.3, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0,1,0], math.pi/2)
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    niksolver = NIK(yumi_instance, component_name='rgt_arm')
    tic = time.time()
    jnt_values = niksolver.num_ik(tgt_pos, tgt_rotmat, toggle_debug=True)
    toc = time.time()
    print(toc - tic)
    yumi_instance.fk(component_name, jnt_values)
    yumi_meshmodel = yumi_instance.gen_meshmodel()
    yumi_meshmodel.attach_to(base)
    yumi_instance.show_cdprimit()
    yumi_instance.gen_stickmodel().attach_to(base)
    tic = time.time()
    result = yumi_instance.is_collided()
    toc = time.time()
    print(result, toc - tic)
    base.run()

