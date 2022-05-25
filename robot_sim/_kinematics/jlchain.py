import math
import copy
import numpy as np
import basis.robot_math as rm
import robot_sim._kinematics.jlchain_mesh as jlm
import robot_sim._kinematics.jlchain_ik as jlik


class JLChain(object):
    """
    Joint Link Chain, no branches allowed
    Usage:
    1. Inherit this class and overwrite self._initjntlnks()/self.tgtjnts to define new joint links
    2. Define multiple instances of this class to compose a complicated structure
    Notes:
    The joint types include "revolute", "prismatic", "end"; One JlChain object alwyas has two "end" joints
    """

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 homeconf=np.zeros(6),
                 name='jlchain',
                 cdprimitive_type='box',
                 cdmesh_type='triangles'):
        """
        initialize a manipulator
        naming rules
        allvalues -- all values: values at all joints including the fixed ones at the base and the end (both are 0)
        conf -- configuration: target joint values
        :param pos:
        :param rotmat:
        :param homeconf: number of joints
        :param name:
        :param cdprimitive_type: 'aabb', 'obb', 'convex_hull', 'triangulation
        :param cdmesh_type:
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
        self._tgtjnts = list(range(1, self.ndof + 1))
        self._jnt_ranges = self._get_jnt_ranges()
        self.goto_homeconf()
        # default tcp
        self.tcp_jnt_id = -1
        self.tcp_loc_pos = np.zeros(3)
        self.tcp_loc_rotmat = np.eye(3)
        # collision primitives
        # mesh generator
        self.cdprimitive_type = cdprimitive_type
        self.cdmesh_type = cdmesh_type
        self._mt = jlm.JLChainMesh(self, cdprimitive_type=cdprimitive_type, cdmesh_type=cdmesh_type)  # t = tool
        self._ikt = jlik.JLChainIK(self)  # t = tool

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
            lnks[id]['mesh_file'] = None
            lnks[id]['collision_model'] = None
            lnks[id]['cdprimit_childid'] = -1  # id of the CollisionChecker.np.Child
            lnks[id]['scale'] = [1, 1, 1]  # 3 list
            lnks[id]['rgba'] = [.57, .57, .57, 1]  # 4 list
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
            jnts[id]['motion_rng'] = [-math.pi, math.pi]  # min, max
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
            if self.jnts[id]['type'] == "end" or self.jnts[id]['type'] == "fixed":
                self.jnts[id]['gl_rotmatq'] = self.jnts[id]['gl_rotmat0']
                self.jnts[id]['gl_posq'] = self.jnts[id]['gl_pos0']
            elif self.jnts[id]['type'] == "revolute":
                self.jnts[id]['gl_rotmatq'] = np.dot(self.jnts[id]['gl_rotmat0'],
                                                     rm.rotmat_from_axangle(self.jnts[id]['loc_motionax'],
                                                                            self.jnts[id]['motion_val']))
                self.jnts[id]['gl_posq'] = self.jnts[id]['gl_pos0']
            elif self.jnts[id]['type'] == "prismatic":
                self.jnts[id]['gl_rotmatq'] = self.jnts[id]['gl_rotmat0']
                tmp_translation = np.dot(self.jnts[id]['gl_rotmatq'],
                                         self.jnts[id]['loc_motionax'] * self.jnts[id]['motion_val'])
                self.jnts[id]['gl_posq'] = self.jnts[id]['gl_pos0'] + tmp_translation
            # update link values, child link id = id
            if id < self.ndof + 1:
                self.lnks[id]['gl_pos'] = np.dot(self.jnts[id]['gl_rotmatq'], self.lnks[id]['loc_pos']) + \
                                          self.jnts[id]['gl_posq']
                self.lnks[id]['gl_rotmat'] = np.dot(self.jnts[id]['gl_rotmatq'], self.lnks[id]['loc_rotmat'])
                # self.lnks[id]['cdprimit_cache'][0] = True
            id = self.jnts[id]['child']
        return self.lnks, self.jnts

    @property
    def homeconf(self):
        return np.array([self._homeconf[i - 1] for i in self.tgtjnts])

    @property
    def zeroconf(self):
        return np.array([self._zeroconf[i - 1] for i in self.tgtjnts])

    @property
    def tgtjnts(self):
        return self._tgtjnts

    @property
    def jnt_ranges(self):
        return self._jnt_ranges

    @tgtjnts.setter
    def tgtjnts(self, values):
        self._tgtjnts = values
        self._jnt_ranges = self._get_jnt_ranges()
        self._ikt = jlik.JLChainIK(self)

    def _get_jnt_ranges(self):
        """
        get jntsrnage
        :return: [[jnt0min, jnt0max], [jnt1min, jnt1max], ...]
        date: 20180602, 20200704osaka
        author: weiwei
        """
        if self.tgtjnts:
            jnt_limits = []
            for id in self.tgtjnts:
                jnt_limits.append(self.jnts[id]['motion_rng'])
            return np.asarray(jnt_limits)
        else:
            return np.empty((0, 2))

    def fix_to(self, pos, rotmat, jnt_values=None):
        # fix the connecting end of the jlchain to the given pos and rotmat
        self.pos = pos
        self.rotmat = rotmat
        return self.fk(jnt_values=jnt_values)

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
            print('The given values must have enough dof!')
            raise Exception

    def reinitialize(self, cdprimitive_type=None, cdmesh_type=None):
        """
        reinitialize jntlinks by updating fk and reconstructing jntlnkmesh
        :return:
        author: weiwei
        date: 20201126
        """
        self._jnt_ranges = self._get_jnt_ranges()
        self.goto_homeconf()
        if cdprimitive_type is None:  # use previously set values if none
            cdprimitive_type = self.cdprimitive_type
        if cdmesh_type is None:
            cdmesh_type = self.cdmesh_type
        self._mg = jlm.JLChainMesh(self, cdprimitive_type, cdmesh_type)
        self._ikt = jlik.JLChainIK(self)

    def set_tcp(self, tcp_jnt_id=None, tcp_loc_pos=None, tcp_loc_rotmat=None):
        if tcp_jnt_id is not None:
            self.tcp_jnt_id = tcp_jnt_id
        if tcp_loc_pos is not None:
            self.tcp_loc_pos = tcp_loc_pos
        if tcp_loc_rotmat is not None:
            self.tcp_loc_rotmat = tcp_loc_rotmat

    def get_gl_tcp(self,
                   tcp_jnt_id=None,
                   tcp_loc_pos=None,
                   tcp_loc_rotmat=None):
        """
        tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat are the tool center pose parameters. They are
        used for temporary computation, the self.tcp_xxx parameters will not be changed
        in case None is provided, the self.tcp_jnt_id, self.tcp_loc_pos, self.tcp_loc_rotmat will be used
        :param tcp_jnt_id:
        :param tcp_loc_pos:
        :param tcp_loc_rotmat:
        :return:
        """
        return self._ikt.get_gl_tcp(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat)

    def is_jnt_values_in_ranges(self, jnt_values):
        """
        check if the given jnt_values
        :param jnt_values:
        :return:
        author: weiwei
        date: 20220326toyonaka
        """
        jnt_values = np.asarray(jnt_values)
        if np.all(self.jnt_ranges[:, 0] <= jnt_values) and np.all(jnt_values <= self.jnt_ranges[:, 1]):
            return True
        else:
            return False

    def fk(self, jnt_values=None):
        """
        move the joints using forward kinematics
        :param jnt_values: a 1xn ndarray where each element indicates the value of a joint (in radian or meter)
        :return
        author: weiwei
        date: 20161205, 20201009osaka
        """
        status = "succ"  # "succ" or "out_of_rng"
        if jnt_values is not None:
            counter = 0
            for id in self.tgtjnts:
                if jnt_values[counter] < self.jnts[id]["motion_rng"][0] or jnt_values[counter] > \
                        self.jnts[id]["motion_rng"][1]:
                    status = "out_of_rng"
                self.jnts[id]['motion_val'] = jnt_values[counter]
                counter += 1
        self._update_fk()
        return status

    def goto_homeconf(self):
        """
        move the robot_s to initial pose
        :return: null
        author: weiwei
        date: 20161211osaka
        """
        self.fk(jnt_values=self.homeconf)

    def goto_zeroconf(self):
        """
        move the robot_s to initial pose
        :return: null
        author: weiwei
        date: 20161211osaka
        """
        self.fk(jnt_values=self.zeroconf)

    def get_jnt_values(self):
        """
        get the current joint values
        :return: jnt_values: a 1xn ndarray
        author: weiwei
        date: 20161205tsukuba
        """
        jnt_values = np.zeros(len(self.tgtjnts))
        counter = 0
        for id in self.tgtjnts:
            jnt_values[counter] = self.jnts[id]['motion_val']
            counter += 1
        return jnt_values

    def rand_conf(self):
        """
        generate a random configuration
        author: weiwei
        date: 20200326
        """
        jnt_values = np.zeros(len(self.tgtjnts))
        counter = 0
        for i in self.tgtjnts:
            jnt_values[counter] = np.random.uniform(self.jnts[i]['motion_rng'][0], self.jnts[i]['motion_rng'][1])
            counter += 1
        return jnt_values

    def ik(self,
           tgt_pos,
           tgt_rotmat,
           seed_jnt_values=None,
           tcp_jnt_id=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           max_niter=100,
           local_minima="accept",
           toggle_debug=False):
        """
        Numerical IK
        NOTE1: in the numik function of rotjntlinksik,
        tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat are the tool center pose parameters. They are
        used for temporary computation, the self.tcp_xxx parameters will not be changed
        in case None is provided, the self.tcp_jnt_id, self.tcp_loc_pos, self.tcp_loc_rotmat will be used
        NOTE2: if list, len(tgtpos)=len(tgtrot) < len(tcp_jnt_id)=len(tcp_loc_pos)=len(tcp_loc_rotmat)
        :param tgt_pos: 1x3 nparray, single value or list
        :param tgt_rotmat: 3x3 nparray, single value or list
        :param seed_jnt_values: the starting configuration used in the numerical iteration
        :param tcp_jnt_id: a joint ID in the self.tgtjnts
        :param tcp_loc_pos: 1x3 nparray, decribed in the local frame of self.jnts[tcp_jnt_id], single value or list
        :param tcp_loc_rotmat: 3x3 nparray, decribed in the local frame of self.jnts[tcp_jnt_id], single value or list
        :param max_niter
        :param local_minima: what to do at local minima: "accept", "randomrestart", "end"
        :return:
        """
        return self._ikt.num_ik(tgt_pos=tgt_pos,
                                tgt_rot=tgt_rotmat,
                                seed_jnt_values=seed_jnt_values,
                                max_niter=max_niter,
                                tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                local_minima=local_minima,
                                toggle_debug=toggle_debug)

    def manipulability(self,
                       tcp_jnt_id,
                       tcp_loc_pos,
                       tcp_loc_rotmat):
        tcp_jnt_id = self.tcp_jnt_id if tcp_jnt_id is None else tcp_jnt_id
        tcp_loc_pos = self.tcp_loc_pos if tcp_loc_pos is None else tcp_loc_pos
        tcp_loc_rotmat = self.tcp_loc_rotmat if tcp_loc_rotmat is None else tcp_loc_rotmat
        return self._ikt.manipulability(tcp_jnt_id=tcp_jnt_id,
                                        tcp_loc_pos=tcp_loc_pos,
                                        tcp_loc_rotmat=tcp_loc_rotmat)

    def manipulability_axmat(self,
                             tcp_jnt_id,
                             tcp_loc_pos,
                             tcp_loc_rotmat,
                             type="translational"):
        return self._ikt.manipulability_axmat(tcp_jnt_id=tcp_jnt_id,
                                              tcp_loc_pos=tcp_loc_pos,
                                              tcp_loc_rotmat=tcp_loc_rotmat,
                                              type=type)

    def jacobian(self,
                 tcp_jnt_id,
                 tcp_loc_pos,
                 tcp_loc_rotmat):
        tcp_jnt_id = self.tcp_jnt_id if tcp_jnt_id is None else tcp_jnt_id
        tcp_loc_pos = self.tcp_loc_pos if tcp_loc_pos is None else tcp_loc_pos
        tcp_loc_rotmat = self.tcp_loc_rotmat if tcp_loc_rotmat is None else tcp_loc_rotmat
        return self._ikt.jacobian(tcp_jnt_id=tcp_jnt_id,
                                  tcp_loc_pos=tcp_loc_pos,
                                  tcp_loc_rotmat=tcp_loc_rotmat)

    def cvt_loc_tcp_to_gl(self,
                          loc_pos=np.zeros(3),
                          loc_rotmat=np.eye(3),
                          tcp_jnt_id=None,
                          tcp_loc_pos=None,
                          tcp_loc_rotmat=None):
        """
        given a relative pos and relative rot with respective to the ith jntlnk,
        get the world pos and world rot
        :param loc_pos: nparray 1x3
        :param loc_romat: nparray 3x3
        :return:
        author: weiwei
        date: 20190312, 20210609
        """
        if tcp_jnt_id is None:
            tcp_jnt_id = self.tcp_jnt_id
        if tcp_loc_pos is None:
            tcp_loc_pos = self.tcp_loc_pos
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.tcp_loc_rotmat
        tcp_gl_pos = self.jnts[tcp_jnt_id]['gl_posq'] + self.jnts[tcp_jnt_id]['gl_rotmatq'].dot(tcp_loc_pos)
        tcp_gl_rotmat = self.jnts[tcp_jnt_id]['gl_rotmatq'].dot(tcp_loc_rotmat)
        gl_pos = tcp_gl_pos + tcp_gl_rotmat.dot(loc_pos)
        gl_rot = tcp_gl_rotmat.dot(loc_rotmat)
        return [gl_pos, gl_rot]

    def cvt_gl_to_loc_tcp(self,
                          gl_pos,
                          gl_rotmat,
                          tcp_jnt_id=None,
                          tcp_loc_pos=None,
                          tcp_loc_rotmat=None):
        """
        given a world pos and world rot
        get the relative pos and relative rot with respective to the ith jntlnk
        :param gl_pos: 1x3 nparray
        :param gl_rotmat: 3x3 nparray
        :param tcp_jnt_id: id of the joint in which the tool center point is defined
        :param tcp_loc_pos: 1x3 nparray, local pose of the tool center point in the frame of the given tcp_jnt_id
        :param tcp_loc_rotmat: 3x3 nparray, local rotmat of the tool center point
        :return:
        author: weiwei
        date: 20190312
        """
        if tcp_jnt_id is None:
            tcp_jnt_id = self.tcp_jnt_id
        if tcp_loc_pos is None:
            tcp_loc_pos = self.tcp_loc_pos
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.tcp_loc_rotmat
        tcp_gloc_pos = self.jnts[tcp_jnt_id]['gl_posq'] + self.jnts[tcp_jnt_id]['gl_rotmatq'].dot(tcp_loc_pos)
        tcp_gloc_rotmat = self.jnts[tcp_jnt_id]['gl_rotmatq'].dot(tcp_loc_rotmat)
        loc_pos, loc_romat = rm.rel_pose(tcp_gloc_pos, tcp_gloc_rotmat, gl_pos, gl_rotmat)
        return [loc_pos, loc_romat]

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=True,
                      toggle_jntscs=False,
                      rgba=None,
                      name='jlcmesh'):
        return self._mt.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                      tcp_loc_pos=tcp_loc_pos,
                                      tcp_loc_rotmat=tcp_loc_rotmat,
                                      toggle_tcpcs=toggle_tcpcs,
                                      toggle_jntscs=toggle_jntscs,
                                      name=name, rgba=rgba)

    def gen_stickmodel(self,
                       rgba=np.array([.5, 0, 0, 1]),
                       thickness=.01,
                       joint_ratio=1.62,
                       link_ratio=.62,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=True,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='jlcstick'):
        return self._mt.gen_stickmodel(rgba=rgba,
                                       thickness=thickness,
                                       joint_ratio=joint_ratio,
                                       link_ratio=link_ratio,
                                       tcp_jnt_id=tcp_jnt_id,
                                       tcp_loc_pos=tcp_loc_pos,
                                       tcp_loc_rotmat=tcp_loc_rotmat,
                                       toggle_tcpcs=toggle_tcpcs,
                                       toggle_jntscs=toggle_jntscs,
                                       toggle_connjnt=toggle_connjnt,
                                       name=name)

    def gen_endsphere(self):
        return self._mt.gen_endsphere()

    def copy(self):
        return copy.deepcopy(self)


if __name__ == "__main__":
    import time
    import visualization.panda.world as wd
    import robot_sim._kinematics.jlchain_mesh as jlm
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[3, 0, 3], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)

    jlinstance = JLChain(homeconf=np.array([0, 0, 0, 0, 0, 0, 0, 0]))
    # rjlinstance.settcp(tcp_jnt_id=rjlinstance.tgtjnts[-3], tcp_loc_pos=np.array([0,0,30]))
    # jlinstance.jnts[4]['type'] = 'prismatic'
    # jlinstance.jnts[4]['loc_motionax'] = np.array([1, 0, 0])
    # jlinstance.jnts[4]['motion_val'] = .2
    # jlinstance.jnts[4]['rngmax'] = 1
    # jlinstance.jnts[4]['rngmin'] = -1
    jlinstance.fk()
    jlinstance.gen_stickmodel().attach_to(base)
    base.run()

    tgt_pos0 = np.array([.45, 0, 0])
    tgt_rotmat0 = np.eye(3)
    tgt_pos1 = np.array([.1, 0, 0])
    tgt_rotmat1 = np.eye(3)
    tgt_pos_list = [tgt_pos0, tgt_pos1]
    tgt_rotmat_list = [tgt_rotmat0, tgt_rotmat1]
    gm.gen_mycframe(pos=tgt_pos0, rotmat=tgt_rotmat0, length=.15, thickness=.01).attach_to(base)
    gm.gen_mycframe(pos=tgt_pos1, rotmat=tgt_rotmat1, length=.15, thickness=.01).attach_to(base)

    tcp_jnt_id_list = [jlinstance.tgtjnts[-1], jlinstance.tgtjnts[-6]]
    tcp_loc_poslist = [np.array([.03, 0, .0]), np.array([.03, 0, .0])]
    tcp_loc_rotmatlist = [np.eye(3), np.eye(3)]
    # tgt_pos_list = tgt_pos_list[0]
    # tgt_rotmat_list = tgt_rotmat_list[0]
    # tcp_jnt_id_list = tcp_jnt_id_list[0]
    # tcp_loc_poslist = tcp_loc_poslist[0]
    # tcp_loc_rotmatlist = tcp_loc_rotmatlist[0]

    tic = time.time()
    jnt_values = jlinstance.ik(tgt_pos_list,
                               tgt_rotmat_list,
                               seed_jnt_values=None,
                               tcp_jnt_id=tcp_jnt_id_list,
                               tcp_loc_pos=tcp_loc_poslist,
                               tcp_loc_rotmat=tcp_loc_rotmatlist,
                               local_minima="accept",
                               toggle_debug=True)
    toc = time.time()
    print('ik cost: ', toc - tic, jnt_values)
    jlinstance.fk(jnt_values=jnt_values)
    jlinstance.gen_stickmodel(tcp_jnt_id=tcp_jnt_id_list,
                              tcp_loc_pos=tcp_loc_poslist,
                              tcp_loc_rotmat=tcp_loc_rotmatlist,
                              toggle_jntscs=True).attach_to(base)

    jlinstance2 = jlinstance.copy()
    jlinstance2.fix_to(pos=np.array([1, 1, 0]), rotmat=rm.rotmat_from_axangle([0, 0, 1], math.pi / 2))
    jlinstance2.gen_stickmodel(tcp_jnt_id=tcp_jnt_id_list,
                               tcp_loc_pos=tcp_loc_poslist,
                               tcp_loc_rotmat=tcp_loc_rotmatlist,
                               toggle_jntscs=True).attach_to(base)
    base.run()
