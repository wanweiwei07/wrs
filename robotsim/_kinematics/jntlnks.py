import math
import numpy as np
import basis.robotmath as rm
import robotsim._kinematics.jntlnksik as jlik
import robotsim._kinematics.jntlnksmesh as jlm


class JntLnks(object):
    """
    Joint Links interface
    inherit this class and overwrite self._initjntlnks()/self.tgtjnts to define new joint links
    The joint types include "revolute", "prismatic", "end"
    """

    def __init__(self, position=np.zeros(3), rotmat=np.eye(3), initconf=np.zeros(6), name='manipulator'):
        """
        initialize a manipulator
        naming rules
        allvalues -- all values: values at all joints including the fixed ones at the base and the end (both are 0)
        conf -- configuration: target joint values
        :param position:
        :param rotmat:
        :param initconf:
        :param name:
        """
        self.name = name
        self.position = np.array(position)
        self.rotmat = np.array(rotmat)
        self.ndof = initconf.shape[0]
        self._zeroallvalues = np.zeros(self.ndof + 2)
        self._initallvalues = np.zeros(self.ndof + 2)
        self._initallvalues[1:self.ndof + 1] = initconf
        self.jntrngsafemargin = 0
        # initialize joints and links
        self.links, self.joints = self._initjntlnks()
        self.tgtjnts = list(range(1, self.ndof + 1))
        self.gotoinitconf()
        # default tcp
        self.tcp_jntid = -1
        self.tcp_localpos = np.zeros(3)
        self.tcp_localrotmat = np.eye(3)
        # mesh generator
        self._jlmg = jlm.JntLnksMesh(self)
        self._jlikslvr = jlik.JntLnksIK(self)

    def _initjntlnks(self):
        """
        init joints and links
        there are two lists of dictionaries where the first one is joints, the second one is links
        links: a list of dictionaries with each dictionary holding the properties of a link
        joints: a list of dictionaries with each dictionary holding the properties of a joint
        njoints is assumed to be equal to nlinks+1
        joint i connects link i-1 and link i
        :return:
        author: weiwei
        date: 20161202tsukuba, 20190328toyonaka, 20200330toyonaka
        """
        links = [dict() for i in range(self.ndof + 1)]
        joints = [dict() for i in range(self.ndof + 2)]
        # l_ for local, g_ for global
        for id in range(self.ndof + 1):
            links[id]['name'] = 'link0'
            links[id]['l_pos'] = np.array([0, 0, 0])
            links[id]['l_rotmat'] = rm.rotmat_from_euler(0, 0, 0)
            links[id]['com'] = np.zeros(3)
            links[id]['inertia'] = np.eye(3)
            links[id]['mass'] = 0  # the visual adjustment is ignored for simplisity
            links[id]['meshfile'] = None
            links[id]['collisionmodel'] = None
            links[id]['rgba'] = np.array([.7, .7, .7, 1])
        for id in range(self.ndof + 2):
            joints[id]['parent'] = id - 1
            joints[id]['child'] = id + 1
            joints[id]['l_pos'] = np.array([0, 0, .1])
            joints[id]['l_rotmat'] = np.eye(3)
            joints[id]['type'] = 'revolute'
            joints[id]['l_mtnax'] = np.array([0, 1, 0])  # l_mtnax: rot axis for rev joint, linear axis for pris joint
            joints[id]['rngmin'] = -(math.pi - self.jntrngsafemargin)
            joints[id]['rngmax'] = +(math.pi - self.jntrngsafemargin)
            joints[id]['movement'] = 0
        joints[0]['l_pos'] = self.position
        joints[0]['l_rotmat'] = self.rotmat
        joints[0]['type'] = 'end'
        joints[self.ndof + 1]['l_pos'] = np.array([0, 0, 0])
        joints[self.ndof + 1]['child'] = -1
        joints[self.ndof + 1]['type'] = 'end'
        return links, joints

    def _updatefk(self):
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
            pjid = self.joints[id]['parent']
            if pjid == -1:
                self.joints[id]['g_pos0'] = self.joints[id]['l_pos']
                self.joints[id]['g_rotmat0'] = self.joints[id]['l_rotmat']
            else:
                self.joints[id]['g_pos0'] = self.joints[pjid]['g_posq'] + np.dot(self.joints[pjid]['g_rotmatq'],
                                                                                 self.joints[id]['l_pos'])
                self.joints[id]['g_rotmat0'] = np.dot(self.joints[pjid]['g_rotmatq'], self.joints[id]['l_rotmat'])
            self.joints[id]['g_mtnax'] = np.dot(self.joints[id]['g_rotmat0'], self.joints[id]['l_mtnax'])
            if self.joints[id]['type'] == "end":
                self.joints[id]['g_rotmatq'] = self.joints[id]['g_rotmat0']
                self.joints[id]['g_posq'] = self.joints[id]['g_pos0']
            elif self.joints[id]['type'] == "revolute":
                self.joints[id]['g_rotmatq'] = np.dot(self.joints[id]['g_rotmat0'],
                                                      rm.rotmat_from_axangle(self.joints[id]['l_mtnax'],
                                                                             self.joints[id]['movement']))
                self.joints[id]['g_posq'] = self.joints[id]['g_pos0']
            elif self.joints[id]['type'] == "prismatic":
                self.joints[id]['g_rotmatq'] = self.joints[id]['g_rotmat0']
                self.joints[id]['g_posq'] = self.joints[id]['g_pos0'] + self.joints[id]['l_mtnax'] * self.joints[id][
                    'movement']
            # update link values, child link id = id
            if id < self.ndof + 1:
                self.links[id]['g_pos'] = np.dot(self.joints[id]['g_rotmatq'], self.links[id]['l_pos']) + \
                                          self.joints[id]['g_posq']
                self.links[id]['g_rotmat'] = np.dot(self.joints[id]['g_rotmatq'], self.links[id]['l_rotmat'])
            id = self.joints[id]['child']
        return self.links, self.joints

    @property
    def initconf(self):
        return np.array([self._initallvalues[i] for i in self.tgtjnts])

    @property
    def zeroconf(self):
        return np.array([self._zeroallvalues[i] for i in self.tgtjnts])

    def setinitvalues(self, jntvalues=None):
        """
        :param jntvalues:
        :return:
        """
        if jntvalues is None:
            jntvalues = np.zeros(self.ndof)
        if len(jntvalues) == self.ndof:
            self._initallvalues[1:self.ndof + 1] = jntvalues
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
        self.gotoinitconf()
        self._jlmg = jlm.JntLnksMesh(self)

    def settcp(self, tcp_jntid=None, tcp_localpos=None, tcp_localrotmat=None):
        if tcp_jntid is not None:
            self.tcp_jntid = tcp_jntid
        if tcp_localpos is not None:
            self.tcp_localpos = tcp_localpos
        if tcp_localrotmat is not None:
            self.tcp_localrotmat = tcp_localrotmat

    def get_globaltcp(self, tcp_jntid=None, tcp_localpos=None, tcp_localrotmat=None):
        """
        tcp_jntid, tcp_localpos, tcp_localrotmat are the tool center pose parameters. They are
        used for temporary computation, the self.tcp_xxx parameters will not be changed
        in case None is provided, the self.tcp_jntid, self.tcp_localpos, self.tcp_localrotmat will be used
        :param jntid:
        :param localpos:
        :param localrotmat:
        :return:
        """
        return self._jlikslvr.get_globaltcp(tcp_jntid, tcp_localpos, tcp_localrotmat)

    def jacobian(self):  # TODO: merge with jlik
        """
        compute the jacobian matrix
        :param rjlinstanceect: a RotJntLnks object
        :return: jmat, a 6xn nparray
        author: weiwei
        date: 20161202, 20200331, 20200705
        """
        jmat = np.zeros((6, len(self.tgtjnts)))
        counter = 0
        for id in self.tgtjnts:
            if self.joints[id]["type"] == "revolute":
                grax = self.joints[id]["g_mtnax"]
                jmat[:, counter] = np.append(
                    np.cross(grax, self.joints[self.tgtjnts[-1]]["g_posq"] - self.joints[id]["g_posq"]), grax)
            elif self.joints[id]["type"] == "prismatic":
                jmat[:, counter] = np.append(self.joints[id]["g_mtnax"], np.zeros(3))
            counter += 1
        return jmat

    def getjntranges(self):
        """
        get jntsrnage
        :return: [[jnt0min, jnt0max], [jnt1min, jnt1max], ...]
        date: 20180602, 20200704osaka
        author: weiwei
        """
        jointlimits = []
        for id in self.tgtjnts:
            jointlimits.append([self.joints[id]['rngmin'], self.joints[id]['rngmax']])
        return jointlimits

    def fk(self, jntvalues=None):
        """
        move the joints using forward kinematics
        :param jntvalues: a 1xn ndarray where each element indicates the value of a joint (in radian or meter)
        :return
        author: weiwei
        date: 20161205, 20201009osaka
        """
        if jntvalues is not None:
            counter = 0
            for id in self.tgtjnts:
                self.joints[id]['movement'] = jntvalues[counter]
                counter += 1
        self._updatefk()

    def gotoinitconf(self):
        """
        move the robot to initial pose
        :return: null
        author: weiwei
        date: 20161211osaka
        """
        self.fk(self.initconf)

    def gotozeroconf(self):
        """
        move the robot to initial pose
        :return: null
        author: weiwei
        date: 20161211osaka
        """
        self.fk(self.zeroconf)

    def getjntvalues(self):
        """
        get the current joint values
        :return: jntvalues: a 1xn ndarray
        author: weiwei
        date: 20161205tsukuba
        """
        jntvalues = np.zeros(len(self.tgtjnts))
        counter = 0
        for id in self.tgtjnts:
            jntvalues[counter] = self.joints[id]['movement']
            counter += 1
        return jntvalues

    def chkrng(self, jntvalues):  # TODO: merge with jlik
        """
        check if the given jntvalues is inside the oeprating range
        this function doesn't check the waist
        :param jntvalues: a 1xn ndarray
        :return: True or False indicating inside the range or not
        author: weiwei
        date: 20161205
        """
        counter = 0
        for id in self.tgtjnts:
            if jntvalues[counter] < self.joints[id]["rngmin"] or jntvalues[counter] > self.joints[id]["rngmax"]:
                print("Joint " + str(id) + " of the arm is out of range!")
                print("Value is " + str(jntvalues[counter]) + " .")
                print("Range is (" + str(self.joints[id]["rngmin"]) + ", " + str(self.joints[id]["rngmax"]) + ").")
                return False
            counter += 1

        return True

    def chkrngdrag(self, jntvalues):  # TODO: merge with jlik
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
        for id in self.tgtjnts:
            if self.joints[id]["type"] == "revolute":
                if self.joints[id]["rngmax"] - self.joints[id]["rngmin"] < math.pi * 2:
                    if jntvalues[counter] < self.joints[id]["rngmin"]:
                        isdragged[counter] = 1
                        jntvaluesdragged[counter] = self.joints[id]["rngmin"]
                    elif jntvalues[counter] > self.joints[id]["rngmax"]:
                        isdragged[counter] = 1
                        jntvaluesdragged[counter] = self.joints[id]["rngmax"]
            elif self.joints[id]["type"] == "prismatic":  # prismatic
                if jntvalues[counter] < self.joints[id]["rngmin"]:
                    isdragged[counter] = 1
                    jntvaluesdragged[counter] = self.joints[id]["rngmin"]
                elif jntvalues[counter] > self.joints[id]["rngmax"]:
                    isdragged[counter] = 1
                    jntvaluesdragged[counter] = self.joints[id]["rngmax"]
            counter += 1

        return isdragged, jntvaluesdragged

    def randompose(self):
        """
        generate a random pose
        :return: jntvalues: a 1xn numpy ndarray
        author: weiwei
        date: 20200326
        """
        jntvalues = np.zeros(len(self.tgtjnts))
        counter = 0
        for i in self.tgtjnts:
            jntvalues[counter] = np.random.uniform(self.joints[i]["rngmin"], self.joints[i]["rngmax"])
            counter += 1
        return jntvalues

    def numik(self, tgtpos, tgtrot, startconf=None, tcp_jntid=None, tcp_localpos=None, tcp_localrotmat=None,
              localminima="accept", toggledebug=False):
        """
        Numerical IK
        NOTE1: in the numik function of rotjntlinksik,
        tcp_jntid, tcp_localpos, tcp_localrotmat are the tool center pose parameters. They are
        used for temporary computation, the self.tcp_xxx parameters will not be changed
        in case None is provided, the self.tcp_jntid, self.tcp_localpos, self.tcp_localrotmat will be used
        NOTE2: if list, len(tgtpos)=len(tgtrot) < len(tcp_jntid)=len(tcp_localpos)=len(tcp_localrotmat)
        :param tgtpos: 1x3 nparray, single value or list
        :param tgtrot: 3x3 nparray, single value or list
        :param startconf: the starting configuration used in the numerical iteration
        :param tcp_jntid: a joint ID in the self.tgtjnts
        :param tcp_localpos: 1x3 nparray, decribed in the local frame of self.joints[tcp_jntid], single value or list
        :param tcp_localrotmat: 3x3 nparray, decribed in the local frame of self.joints[tcp_jntid], single value or list
        :param localminima:
        :return:
        """
        return self._jlikslvr.numik(tgtpos, tgtrot, startconf, tcp_jntid, tcp_localpos, tcp_localrotmat, localminima,
                                    toggledebug=toggledebug)

    def getworldpose(self, relpos, relrot, tcp_jntid=None, tcp_localpos=None, tcp_localrotmat=None):
        """
        given a relative pos and relative rot with respective to the ith jntlnk,
        get the world pos and world rot
        :param relpos: nparray 1x3
        :param relrot: nparray 3x3
        :return:
        author: weiwei
        date: 20190312
        """
        if tcp_jntid is None:
            tcp_jntid = self.tgtjnts[-1]
        objpos = self.joints[tcp_jntid]['g_posq'] + np.dot(self.joints[tcp_jntid]['g_rotmatq'], relpos)
        objrot = np.dot(self.joints[tcp_jntid]['g_rotmat'], relrot)
        return [objpos, objrot]

    def getrelpose(self, worldpos, worldrot, tcp_jntid=None, tcp_localpos=None, tcp_localrotmat=None):
        """
        given a world pos and world rot
        get the relative pos and relative rot with respective to the ith jntlnk
        :param worldpos: nparray 1x3
        :param worldrot: nparray 3x3
        :return:
        author: weiwei
        date: 20190312
        """
        if tcp_jntid is None:
            tcp_jntid = self.tgtjnts[-1]
        relpos, relrot = rm.relpose(self.joints[tcp_jntid]['g_posq'], self.joints[tcp_jntid]['g_rotmatq'], worldpos,
                                    worldrot)
        return [relpos, relrot]

    def gen_meshmodel(self, tcp_jntid=None, tcp_localpos=None, tcp_localrotmat=None, toggletcpcs=True,
                      togglejntscs=False, name='robotmesh', drawhand=True, rgbargt=None, rgbalft=None):
        return self._jlmg.gen_meshmodel(tcp_jntid=tcp_jntid, tcp_localpos=tcp_localpos, tcp_localrotmat=tcp_localrotmat,
                                        toggletcpcs=toggletcpcs, togglejntscs=togglejntscs, name=name,
                                        drawhand=drawhand, rgbargt=rgbargt, rgbalft=rgbalft)

    def gen_stickmodel(self, rgba=np.array([.5, 0, 0, 1]), thickness=.01, jointratio=1.62, linkratio=.62,
                       tcp_jntid=None, tcp_localpos=None, tcp_localrotmat=None,
                       toggletcpcs=True, togglejntscs=False, togglecntjnt=False, name='robotstick'):
        return self._jlmg.gen_stickmodel(rgba=rgba, thickness=thickness, jointratio=jointratio, linkratio=linkratio,
                                         tcp_jntid=tcp_jntid, tcp_localpos=tcp_localpos,
                                         tcp_localrotmat=tcp_localrotmat, toggletcpcs=toggletcpcs,
                                         togglejntscs=togglejntscs, togglecntjnt=togglecntjnt, name=name)

    def gen_endsphere(self):
        return self._jlmg.gen_endsphere()


if __name__ == "__main__":
    import time
    import visualization.panda.world as wd
    import robotsim._kinematics.jntlnksmesh as jlm
    import modeling.geometricmodel as gm

    base = wd.World(camp=[3, 0, 3], lookatpos=[0, 0, 0])
    gm.gen_frame().attach_to(base)

    jlinstance = JntLnks(initconf=np.array([0, 0, 0, 0, 0, 0, 0, 0]))
    # rjlinstance.settcp(tcp_jntid=rjlinstance.tgtjnts[-3], tcp_localpos=np.array([0,0,30]))
    # jlinstance.joints[4]['type'] = 'prismatic'
    # jlinstance.joints[4]['l_mtnax'] = np.array([1, 0, 0])
    # jlinstance.joints[4]['movement'] = 0
    # jlinstance.joints[4]['rngmax'] = 1
    # jlinstance.joints[4]['rngmin'] = -1
    jlinstance.fk()

    tgtpos0 = np.array([.45, 0, .3])
    tgtrot0 = np.eye(3)
    tgtpos1 = np.array([.1, 0, 0])
    tgtrot1 = np.eye(3)
    tgtposlist = [tgtpos0, tgtpos1]
    tgtrotlist = [tgtrot0, tgtrot1]
    gm.gen_mycframe(pos=tgtpos0, rotmat=tgtrot0, length=.15, thickness=.01).attach_to(base)
    gm.gen_mycframe(pos=tgtpos1, rotmat=tgtrot1, length=.15, thickness=.01).attach_to(base)

    tcp_jntidlist = [jlinstance.tgtjnts[-1], jlinstance.tgtjnts[-6]]
    tcp_localposlist = [np.array([.03, 0, -.05]), np.array([.03, 0, .0])]
    tcp_localrotmatlist = [np.eye(3), np.eye(3)]
    #
    # tgtposlist = tgtposlist[0]
    # tgtrotlist = tgtrotlist[0]
    # tcp_jntidlist = tcp_jntidlist[0]
    # tcp_localposlist = tcp_localposlist[0]
    # tcp_localrotmatlist = tcp_localrotmatlist[0]

    tic = time.time()
    jntvalues = jlinstance.numik(tgtposlist, tgtrotlist, startconf=None, tcp_jntid=tcp_jntidlist,
                                 tcp_localpos=tcp_localposlist, tcp_localrotmat=tcp_localrotmatlist,
                                 localminima="accept", toggledebug=True)
    toc = time.time()
    print("ik cost: ", toc - tic, jntvalues)
    jlinstance.fk(jntvalues)
    jlinstance.gen_stickmodel(tcp_jntid=tcp_jntidlist, tcp_localpos=tcp_localposlist,
                              tcp_localrotmat=tcp_localrotmatlist, togglejntscs=True).reparent_to(base)
    base.run()
