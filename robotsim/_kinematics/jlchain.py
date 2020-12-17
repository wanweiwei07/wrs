import math
import numpy as np
import basis.robotmath as rm
import robotsim._kinematics.jlchainmesh as jlm
import robotsim._kinematics.jlchainik as jlik


class JLChain(object):
    """
    Joint Link Chain, no branches allowed
    Usage:
    1. Inherit this class and overwrite self._initjntlnks()/self.tgtjnts to define new joint links
    2. Define multiple instances of this class to compose a complicated structure
    Notes:
    The joint types include "revolute", "prismatic", "end"; One JlChain object alwyas has two "end" joints
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(6), name='jlchain'):
        """
        initialize a manipulator
        naming rules
        allvalues -- all values: values at all joints including the fixed ones at the base and the end (both are 0)
        conf -- configuration: target joint values
        :param pos:
        :param rotmat:
        :param homeconf:
        :param name:
        """
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        self.ndof = homeconf.shape[0]
        self._zeroconf = np.zeros(self.ndof)
        self._homeconf = homeconf.astype('float64')
        # initialize joints and links
        self.lnks, self.jnts = self._init_jlchain()
        self.tgtjnts = range(1, self.ndof + 1)
        self.goto_homeconf()
        # default tcp
        self.tcp_jntid = -1
        self.tcp_loc_pos = np.zeros(3)
        self.tcp_loc_rotmat = np.eye(3)
        # mesh generator
        self._mt = jlm.JLChainMesh(self)
        self._ikt = jlik.JLChainIK(self)

    def _init_jlchain(self):
        """
        init joints and links chains
        there are two lists of dictionaries where the first one is joints, the second one is links
        links: a list of dictionaries with each dictionary holding the properties of a link
        joints: a list of dictionaries with each dictionary holding the properties of a joint
        njoints is assumed to be equal to nlinks+1
        joint i connects link i-1 and link i
        :return:
        author: weiwei
        date: 20161202tsukuba, 20190328toyonaka, 20200330toyonaka
        """
        lnks = [dict() for i in range(self.ndof + 1)]
        jnts = [dict() for i in range(self.ndof + 2)]
        for id in range(self.ndof + 1):
            lnks[id]['name'] = 'link0'
            lnks[id]['loc_pos'] = np.array([0, 0, 0])
            lnks[id]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, 0)
            lnks[id]['com'] = np.zeros(3)
            lnks[id]['inertia'] = np.eye(3)
            lnks[id]['mass'] = 0  # the visual adjustment is ignored for simplisity
            lnks[id]['meshfile'] = None
            lnks[id]['collisionmodel'] = None
            lnks[id]['cdprimit_cache'] = [False, None] # p1: need update? p2: tmp nodepath that holds the primit
            lnks[id]['scale'] = None  # 3 list
            lnks[id]['rgba'] = [.7, .7, .7, 1]  # 4 list
        for id in range(self.ndof + 2):
            jnts[id]['type'] = 'revolute'
            jnts[id]['parent'] = id - 1
            jnts[id]['child'] = id + 1
            jnts[id]['loc_pos'] = np.array([0, .1, 0]) if id > 0 else np.array([0, 0, 0])
            jnts[id]['loc_rotmat'] = np.eye(3)
            jnts[id]['loc_motionax'] = np.array([0, 0, 1])  # rot ax for rev joint, linear ax for pris joint
            jnts[id]['gl_pos0'] = jnts[id]['loc_pos']  # to be updated by self._update_fk
            jnts[id]['gl_rotmat0'] = jnts[id]['loc_rotmat']  # to be updated by self._update_fk
            jnts[id]['gl_motionax'] = jnts[id]['loc_motionax']  # to be updated by self._update_fk
            jnts[id]['gl_posq'] = jnts[id]['gl_pos0']  # to be updated by self._update_fk
            jnts[id]['gl_rotmatq'] = jnts[id]['gl_rotmat0']  # to be updated by self._update_fk
            jnts[id]['rngmin'] = -math.pi
            jnts[id]['rngmax'] = +math.pi
            jnts[id]['motion_val'] = 0
        jnts[0]['gl_pos0'] = self.pos  # This is not necessary, for easy read
        jnts[0]['gl_rotmat0'] = self.rotmat
        jnts[0]['type'] = 'end'
        jnts[self.ndof + 1]['loc_pos'] = np.array([0, 0, 0])
        jnts[self.ndof + 1]['child'] = -1
        jnts[self.ndof + 1]['type'] = 'end'
        return lnks, jnts

    def _update_fk(self):
        """
        Update the kinematics
        Note that this function should not be called explicitly
        It is called automatically by functions like movexxx
        :return: updated links and joints
        author: weiwei
        date: 20161202, 20201009osaka
        """
        id = 0
        while id != -1:
            # update joint values
            pjid = self.jnts[id]['parent']
            if pjid == -1:
                self.jnts[id]['gl_pos0'] = self.pos
                self.jnts[id]['gl_rotmat0'] = self.rotmat
            else:
                self.jnts[id]['gl_pos0'] = self.jnts[pjid]['gl_posq'] + np.dot(self.jnts[pjid]['gl_rotmatq'],
                                                                               self.jnts[id]['loc_pos'])
                self.jnts[id]['gl_rotmat0'] = np.dot(self.jnts[pjid]['gl_rotmatq'], self.jnts[id]['loc_rotmat'])
            self.jnts[id]['gl_motionax'] = np.dot(self.jnts[id]['gl_rotmat0'], self.jnts[id]['loc_motionax'])
            if self.jnts[id]['type'] == "end":
                self.jnts[id]['gl_rotmatq'] = self.jnts[id]['gl_rotmat0']
                self.jnts[id]['gl_posq'] = self.jnts[id]['gl_pos0']
            elif self.jnts[id]['type'] == "revolute":
                self.jnts[id]['gl_rotmatq'] = np.dot(self.jnts[id]['gl_rotmat0'],
                                                     rm.rotmat_from_axangle(self.jnts[id]['loc_motionax'],
                                                                            self.jnts[id]['motion_val']))
                self.jnts[id]['gl_posq'] = self.jnts[id]['gl_pos0']
            elif self.jnts[id]['type'] == "prismatic":
                self.jnts[id]['gl_rotmatq'] = self.jnts[id]['gl_rotmat0']
                self.jnts[id]['gl_posq'] = self.jnts[id]['gl_pos0'] + self.jnts[id]['loc_motionax'] * self.jnts[id][
                    'motion_val']
            # update link values, child link id = id
            if id < self.ndof + 1:
                self.lnks[id]['gl_pos'] = np.dot(self.jnts[id]['gl_rotmatq'], self.lnks[id]['loc_pos']) + \
                                          self.jnts[id]['gl_posq']
                self.lnks[id]['gl_rotmat'] = np.dot(self.jnts[id]['gl_rotmatq'], self.lnks[id]['loc_rotmat'])
                self.lnks[id]['cdprimit_cache'][0] = True
            id = self.jnts[id]['child']
        return self.lnks, self.jnts

    @property
    def homeconf(self):
        return np.array([self._homeconf[i - 1] for i in self.tgtjnts])

    @property
    def zeroconf(self):
        return np.array([self._zeroconf[i - 1] for i in self.tgtjnts])

    def fix_to(self, pos, rotmat, jnt_values=None):
        # fix the connecting end of the jlchain to the given pos and rotmat
        self.pos = pos
        self.rotmat = rotmat
        self.fk(jnt_values=jnt_values)

    def set_homeconf(self, jnt_values=None):
        """
        :param jnt_values:
        :return:
        """
        if jnt_values is None:
            jnt_values = np.zeros(self.ndof)
        if len(jnt_values) == self.ndof:
            self._homeconf = jnt_values
        else:
            print("The given values must have enough dof!")
            raise Exception

    def reinitialize(self):
        """
        reinitialize jntlinks by updating fk and reconstructing jntlnkmesh
        :return:
        author: weiwei
        date: 20201126
        """
        self.goto_homeconf()
        self._mg = jlm.JLChainMesh(self)

    def settcp(self, tcp_jntid=None, tcp_loc_pos=None, tcp_loc_rotmat=None):
        if tcp_jntid is not None:
            self.tcp_jntid = tcp_jntid
        if tcp_loc_pos is not None:
            self.tcp_loc_pos = tcp_loc_pos
        if tcp_loc_rotmat is not None:
            self.tcp_loc_rotmat = tcp_loc_rotmat

    def get_globaltcp(self, tcp_jntid=None, tcp_loc_pos=None, tcp_loc_rotmat=None):
        """
        tcp_jntid, tcp_loc_pos, tcp_loc_rotmat are the tool center pose parameters. They are
        used for temporary computation, the self.tcp_xxx parameters will not be changed
        in case None is provided, the self.tcp_jntid, self.tcp_loc_pos, self.tcp_loc_rotmat will be used
        :param jntid:
        :param locpos:
        :param locrotmat:
        :return:
        """
        return self._ikt.get_globaltcp(tcp_jntid, tcp_loc_pos, tcp_loc_rotmat)

    def get_jntranges(self):
        """
        get jntsrnage
        :return: [[jnt0min, jnt0max], [jnt1min, jnt1max], ...]
        date: 20180602, 20200704osaka
        author: weiwei
        """
        jnt_limits = []
        for id in self.tgtjnts:
            jnt_limits.append([self.jnts[id]['rngmin'], self.jnts[id]['rngmax']])
        return jnt_limits

    def fk(self, jnt_values=None):
        """
        move the joints using forward kinematics
        :param jnt_values: a 1xn ndarray where each element indicates the value of a joint (in radian or meter)
        :return
        author: weiwei
        date: 20161205, 20201009osaka
        """
        if jnt_values is not None:
            counter = 0
            for id in self.tgtjnts:
                self.jnts[id]['motion_val'] = jnt_values[counter]
                counter += 1
        self._update_fk()

    def goto_homeconf(self):
        """
        move the robot to initial pose
        :return: null
        author: weiwei
        date: 20161211osaka
        """
        self.fk(jnt_values=self.homeconf)

    def goto_zeroconf(self):
        """
        move the robot to initial pose
        :return: null
        author: weiwei
        date: 20161211osaka
        """
        self.fk(jnt_values=self.zeroconf)

    def get_jntvalues(self):
        """
        get the current joint values
        :return: jntvalues: a 1xn ndarray
        author: weiwei
        date: 20161205tsukuba
        """
        jnt_values = np.zeros(len(self.tgtjnts))
        counter = 0
        for id in self.tgtjnts:
            jnt_values[counter] = self.jnts[id]['motion_val']
            counter += 1
        return jnt_values

    # def jacobian(self):  # TODO: merge with jlik
    #     """
    #     compute the jacobian matrix
    #     :return: jmat, a 6xn nparray
    #     author: weiwei
    #     date: 20161202, 20200331, 20200705
    #     """
    #     jmat = np.zeros((6, len(self.tgtjnts)))
    #     counter = 0
    #     for id in self.tgtjnts:
    #         if self.jnts[id]["type"] == "revolute":
    #             grax = self.jnts[id]["gl_motionax"]
    #             jmat[:, counter] = np.append(
    #                 np.cross(grax, self.jnts[self.tgtjnts[-1]]["gl_posq"] - self.jnts[id]["gl_posq"]), grax)
    #         elif self.jnts[id]["type"] == "prismatic":
    #             jmat[:, counter] = np.append(self.jnts[id]["gl_motionax"], np.zeros(3))
    #         counter += 1
    #     return jmat

    # def chkrng(self, jnt_values):  # TODO: merge with jlik
    #     """
    #     check if the given jntvalues is inside the oeprating range
    #     this function doesn't check the waist
    #     :param jntvalues: a 1xn ndarray
    #     :return: True or False indicating inside the range or not
    #     author: weiwei
    #     date: 20161205
    #     """
    #     counter = 0
    #     for id in self.tgtjnts:
    #         if jntvalues[counter] < self.jnts[id]["rngmin"] or jntvalues[counter] > self.jnts[id]["rngmax"]:
    #             print("Joint " + str(id) + " of the arm is out of range!")
    #             print("Value is " + str(jntvalues[counter]) + " .")
    #             print("Range is (" + str(self.jnts[id]["rngmin"]) + ", " + str(self.jnts[id]["rngmax"]) + ").")
    #             return False
    #         counter += 1
    #
    #     return True

    # def chkrngdrag(self, jntvalues):  # TODO: merge with jlik
    #     """
    #     check if the given jntvalues is inside the oeprating range
    #     The joint values out of range will be pulled back to their maxima
    #     :param jntvalues: a 1xn numpy ndarray
    #     :return: Two parameters, one is true or false indicating if the joint values are inside the range or not
    #             The other is the joint values after dragging.
    #             If the joints were not dragged, the same joint values will be returned
    #     author: weiwei
    #     date: 20161205
    #     """
    #     counter = 0
    #     isdragged = np.zeros_like(jntvalues)
    #     jntvaluesdragged = jntvalues.copy()
    #     for id in self.tgtjnts:
    #         if self.jnts[id]["type"] == "revolute":
    #             if self.jnts[id]["rngmax"] - self.jnts[id]["rngmin"] < math.pi * 2:
    #                 if jntvalues[counter] < self.jnts[id]["rngmin"]:
    #                     isdragged[counter] = 1
    #                     jntvaluesdragged[counter] = self.jnts[id]["rngmin"]
    #                 elif jntvalues[counter] > self.jnts[id]["rngmax"]:
    #                     isdragged[counter] = 1
    #                     jntvaluesdragged[counter] = self.jnts[id]["rngmax"]
    #         elif self.jnts[id]["type"] == "prismatic":  # prismatic
    #             if jntvalues[counter] < self.jnts[id]["rngmin"]:
    #                 isdragged[counter] = 1
    #                 jntvaluesdragged[counter] = self.jnts[id]["rngmin"]
    #             elif jntvalues[counter] > self.jnts[id]["rngmax"]:
    #                 isdragged[counter] = 1
    #                 jntvaluesdragged[counter] = self.jnts[id]["rngmax"]
    #         counter += 1
    #
    #     return isdragged, jntvaluesdragged

    def rand_conf(self):
        """
        generate a random configuration
        author: weiwei
        date: 20200326
        """
        jnt_values = np.zeros(len(self.tgtjnts))
        counter = 0
        for i in self.tgtjnts:
            jnt_values[counter] = np.random.uniform(self.jnts[i]["rngmin"], self.jnts[i]["rngmax"])
            counter += 1
        return jnt_values

    def num_ik(self, tgtpos, tgtrot, startconf=None, tcp_jntid=None, tcp_loc_pos=None, tcp_loc_rotmat=None,
               local_minima="accept", toggle_debug=False):
        """
        Numerical IK
        NOTE1: in the numik function of rotjntlinksik,
        tcp_jntid, tcp_loc_pos, tcp_loc_rotmat are the tool center pose parameters. They are
        used for temporary computation, the self.tcp_xxx parameters will not be changed
        in case None is provided, the self.tcp_jntid, self.tcp_loc_pos, self.tcp_loc_rotmat will be used
        NOTE2: if list, len(tgtpos)=len(tgtrot) < len(tcp_jntid)=len(tcp_loc_pos)=len(tcp_loc_rotmat)
        :param tgtpos: 1x3 nparray, single value or list
        :param tgtrot: 3x3 nparray, single value or list
        :param startconf: the starting configuration used in the numerical iteration
        :param tcp_jntid: a joint ID in the self.tgtjnts
        :param tcp_loc_pos: 1x3 nparray, decribed in the local frame of self.jnts[tcp_jntid], single value or list
        :param tcp_loc_rotmat: 3x3 nparray, decribed in the local frame of self.jnts[tcp_jntid], single value or list
        :param local_minima:
        :return:
        """
        return self._ikt.numik(tgtpos, tgtrot, startconf, tcp_jntid, tcp_loc_pos, tcp_loc_rotmat, local_minima,
                                  toggle_debug=toggle_debug)

    def get_worldpose(self, relpos=np.zeros(3), relrot=np.eye(3), tcp_jntid=None, tcp_loc_pos=None, tcp_loc_rotmat=None):
        """
        TODO change name to get_locpose and get_glpose
        given a relative pos and relative rot with respective to the ith jntlnk,
        get the world pos and world rot
        :param relpos: nparray 1x3
        :param relrot: nparray 3x3
        :return:
        author: weiwei
        date: 20190312
        """
        if tcp_jntid is None:
            tcp_jntid = self.tcp_jntid
        if tcp_loc_pos is None:
            tcp_loc_pos = np.zeros(3)
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = np.eye(3)
        tcp_gloc_pos = self.jnts[tcp_jntid]['gl_posq'] + self.jnts[tcp_jntid]['gl_rotmatq'].dot(tcp_loc_pos)
        tcp_gloc_rotmat = self.jnts[tcp_jntid]['gl_rotmatq'].dot(tcp_loc_rotmat)
        objpos = tcp_gloc_pos + tcp_gloc_rotmat.dot(relpos)
        objrot = tcp_gloc_rotmat.dot(relrot)
        return [objpos, objrot]

    def get_relpose(self, worldpos, worldrot, tcp_jntid=None, tcp_loc_pos=None, tcp_loc_rotmat=None):
        """
        given a world pos and world rot
        get the relative pos and relative rot with respective to the ith jntlnk
        :param worldpos: 1x3 nparray
        :param worldrot: 3x3 nparray
        :param tcp_jntid: id of the joint in which the tool center point is defined
        :param tcp_loc_pos: 1x3 nparray, local pose of the tool center point in the frame of the given tcp_jntid
        :param tcp_loc_rotmat: 3x3 nparray, local rotmat of the tool center point
        :return:
        author: weiwei
        date: 20190312
        """
        if tcp_jntid is None:
            tcp_jntid = self.tcp_jntid
        if tcp_loc_pos is None:
            tcp_loc_pos = np.zeros(3)
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = np.eye(3)
        tcp_gloc_pos = self.jnts[tcp_jntid]['gl_posq'] + self.jnts[tcp_jntid]['gl_rotmatq'].dot(tcp_loc_pos)
        tcp_gloc_rotmat = self.jnts[tcp_jntid]['gl_rotmatq'].dot(tcp_loc_rotmat)
        relpos, relrot = rm.relpose(tcp_gloc_pos, tcp_gloc_rotmat, worldpos, worldrot)
        return [relpos, relrot]

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        return self._mt.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list)

    def disable_localcc(self):
        """
        disable local collision checker
        :return:
        """
        self._mt.disable_localcc()

    def gen_meshmodel(self, tcp_jntid=None, tcp_loc_pos=None, tcp_loc_rotmat=None, toggletcpcs=True,
                      togglejntscs=False, name='robotmesh', drawhand=True, rgbargt=None, rgbalft=None):
        return self._mt.gen_meshmodel(tcp_jntid=tcp_jntid, tcp_loc_pos=tcp_loc_pos, tcp_loc_rotmat=tcp_loc_rotmat,
                                      toggletcpcs=toggletcpcs, togglejntscs=togglejntscs, name=name,
                                      drawhand=drawhand, rgbargt=rgbargt, rgbalft=rgbalft)

    def gen_stickmodel(self, rgba=np.array([.5, 0, 0, 1]), thickness=.01, jointratio=1.62, linkratio=.62,
                       tcp_jntid=None, tcp_loc_pos=None, tcp_loc_rotmat=None,
                       toggletcpcs=True, togglejntscs=False, toggleconnjnt=False, name='robotstick'):
        return self._mt.gen_stickmodel(rgba=rgba, thickness=thickness, jointratio=jointratio, linkratio=linkratio,
                                       tcp_jntid=tcp_jntid, tcp_loc_pos=tcp_loc_pos,
                                       tcp_loc_rotmat=tcp_loc_rotmat, toggletcpcs=toggletcpcs,
                                       togglejntscs=togglejntscs, toggleconnjnt=toggleconnjnt, name=name)

    def gen_endsphere(self):
        return self._mt.gen_endsphere()

    def show_cdprimit(self):
        self._mt.show_cdprimit()

    def unshow_cdprimit(self):
        for id in range(self.ndof + 1):
            if self.lnks[id]['collisionmodel'] is not None:
                self.lnks[id]['collisionmodel'].unshow_cdprimit()

    def copy(self):
        """
        TODO 20201211
        :return:
        """
        pass


if __name__ == "__main__":
    import time
    import visualization.panda.world as wd
    import robotsim._kinematics.jlchainmesh as jlm
    import modeling.geometricmodel as gm

    base = wd.World(campos=[3, 0, 3], lookatpos=[0, 0, 0])
    gm.gen_frame().attach_to(base)

    jlinstance = JLChain(homeconf=np.array([0, 0, 0, 0, 0, 0, 0, 0]))
    # rjlinstance.settcp(tcp_jntid=rjlinstance.tgtjnts[-3], tcp_loc_pos=np.array([0,0,30]))
    # jlinstance.jnts[4]['type'] = 'prismatic'
    # jlinstance.jnts[4]['loc_motionax'] = np.array([1, 0, 0])
    # jlinstance.jnts[4]['motion_val'] = 0
    # jlinstance.jnts[4]['rngmax'] = 1
    # jlinstance.jnts[4]['rngmin'] = -1
    jlinstance.fk()

    tgtpos0 = np.array([.45, 0, 0])
    tgtrot0 = np.eye(3)
    tgtpos1 = np.array([.1, 0, 0])
    tgtrot1 = np.eye(3)
    tgtposlist = [tgtpos0, tgtpos1]
    tgtrotlist = [tgtrot0, tgtrot1]
    gm.gen_mycframe(pos=tgtpos0, rotmat=tgtrot0, length=.15, thickness=.01).attach_to(base)
    gm.gen_mycframe(pos=tgtpos1, rotmat=tgtrot1, length=.15, thickness=.01).attach_to(base)

    tcp_jntidlist = [jlinstance.tgtjnts[-1], jlinstance.tgtjnts[-6]]
    tcp_loc_poslist = [np.array([.03, 0, .0]), np.array([.03, 0, .0])]
    tcp_loc_rotmatlist = [np.eye(3), np.eye(3)]
    #
    # tgtposlist = tgtposlist[0]
    # tgtrotlist = tgtrotlist[0]
    # tcp_jntidlist = tcp_jntidlist[0]
    # tcp_loc_poslist = tcp_loc_poslist[0]
    # tcp_loc_rotmatlist = tcp_loc_rotmatlist[0]

    tic = time.time()
    jntvalues = jlinstance.num_ik(tgtposlist, tgtrotlist, startconf=None, tcp_jntid=tcp_jntidlist,
                                  tcp_loc_pos=tcp_loc_poslist, tcp_loc_rotmat=tcp_loc_rotmatlist,
                                  local_minima="accept", toggle_debug=True)
    toc = time.time()
    print("ik cost: ", toc - tic, jntvalues)
    jlinstance.fk(jnt_values=jntvalues)
    jlinstance.gen_stickmodel(tcp_jntid=tcp_jntidlist, tcp_loc_pos=tcp_loc_poslist,
                              tcp_loc_rotmat=tcp_loc_rotmatlist, togglejntscs=True).attach_to(base)
    base.run()