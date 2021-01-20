import math
import numpy as np
import basis.robot_math as rm


class Jnt(object):
    def __init__(self, name="joint",
                 type="revolute",
                 loc_pos=np.array([0, 0, .1]),
                 loc_rotmat=np.eye(3),
                 loc_motionax=np.array([0, 0, 1]),
                 gl_pos0=np.zeros(3),  # to be updated by fk
                 gl_rotmat0=np.eye(3),  # to be updated by fk
                 gl_motionax=np.zeros(3),  # to be updated by fk
                 gl_posq=np.zeros(3),  # to be updated by fk
                 gl_rotmatq=np.eye(3),  # to be updated by fk
                 rng_min=-math.pi,
                 rng_max=+math.pi,
                 motion_val=.0,
                 p_name=None,
                 chd_name_list=[],
                 lnk_name_dict={}):
        self.name = name
        self.type = type
        self.loc_pos = loc_pos
        self.loc_rotmat = loc_rotmat
        self.loc_motionax = loc_motionax
        self.gl_pos0 = gl_pos0
        self.gl_rotmat0 = gl_rotmat0
        self.gl_motionax = gl_motionax
        self.gl_posq = gl_posq
        self.gl_rotmatq = gl_rotmatq
        self.rng_min = rng_min
        self.rng_max = rng_max
        self.motion_val = motion_val
        self.p_name = p_name  # a single value
        self.chd_name_list = chd_name_list  # a list, may include only a single value
        self.lnk_name_dict = lnk_name_dict  # a dictionary, {chd_name: lnk_name, ...}


class Lnk(object):
    def __init__(self, name="link",
                 refjnt_name=None,
                 loc_pos=np.zeros(3),
                 loc_rotmat=np.eye(3),
                 gl_pos=np.zeros(3),  # to be updated by fk
                 gl_rotmat=np.eye(3),  # to be updated by fk
                 com=np.zeros(3),
                 intertia=np.eye(3),
                 mass=.0,
                 meshfile=None,
                 collisionmodel=None,
                 rgba=np.array([.7, .7, .7, 1])):
        self.name = name
        self.loc_pos = loc_pos
        self.loc_rotmat = loc_rotmat
        self.gl_pos = gl_pos
        self.gl_rotmat = gl_rotmat
        self.com = com
        self.intertia = intertia
        self.mass = mass
        self.meshfile = meshfile
        self.collisionmodel = collisionmodel
        self.rgba = rgba


class JLTree(object):
    """
    Define Joint Links using Networkx DiGraph
    """

    def __init__(self, position=np.zeros(3), rotmat=np.eye(3), initconf=np.zeros(6), name='manipulator'):
        """
        :param position:
        :param rotmat:
        :param initconf:
        :param name:
        """
        self.name = name
        self.position = np.array(position)
        self.rotmat = np.array(rotmat)
        self.ndof = initconf.shape[0]
        self.jntrng_safemargin = 0
        # base: an nx1 list of compactly connected nodes, comprises at least two jnt_names
        self.jnt_collection, self.lnk_collection, self._base = self._initjntlnks()
        self.tgtjnt_ids = ['jnt' + str(id) for id in range(self.ndof)]

    def _initgraph(self):
        """
        :return:  jnt_collection, lnk_collection,
        """
        # links
        lnk_collection = {}
        lnk_name = self.name + '_lnk_f0j0'
        lnk_collection[lnk_name] = Lnk()
        for id in range(self.ndof - 1):
            lnk_name = self.name + '_link_j' + str(id) + 'j' + str(id + 1)
            lnk_collection[lnk_name] = Lnk()
        lnk_name = self.name + '_link_j' + str(self.ndof - 1) + 'f1'
        lnk_collection[lnk_name] = Lnk()
        # joints
        jnt_collection = {}
        jnt_name = self.name + '_f0'
        jnt_collection[jnt_name] = Jnt(type='fixed')
        jnt_collection[jnt_name].p_name = None
        jnt_collection[jnt_name].chd_name_list = [self.name + '_j0']
        jnt_collection[jnt_name].lnk_name_dict[self.name + '_j0'] = [self.name + '_lnk_f0j0']
        for id in range(self.ndof):
            jnt_name = self.name + '_j' + str(id)
            jnt_collection[jnt_name] = Jnt()
            if id == 0:
                jnt_collection[jnt_name].p_name = self.name + '_f0'
                jnt_collection[jnt_name].chd_name_list = [self.name + '_j1']
                jnt_collection[jnt_name].lnk_name_dict[self.name + '_j1'] = [self.name + '_lnk_f0j1']
            elif id == self.ndof - 1:
                jnt_collection[jnt_name].p_name = self.name + '_j' + str(id - 1)
                jnt_collection[jnt_name].chd_name_list = [self.name + '_f1']
                jnt_collection[jnt_name].lnk_name_dict[self.name + '_f1'] = [self.name + '_lnk_j' + str(id) + 'f1']
            else:
                jnt_collection[jnt_name].p_name = self.name + '_j' + str(id - 1)
                jnt_collection[jnt_name].chd_name_list = [self.name + '_j' + str(id + 1)]
                jnt_collection[jnt_name].lnk_name_dict[self.name + '_j' + str(id + 1)] = [self.name + '_lnk_j' + str(id) + 'j' + str(id + 1)]
        jnt_name = self.name + '_f1'
        jnt_collection[jnt_name] = Jnt(type='fixed')
        jnt_collection[jnt_name].p_name = self.name + '_joint' + str(self.ndof - 1)
        jnt_collection[jnt_name].chd_name_list = []
        jnt_collection[jnt_name].lnk_name_dict = {}
        return jnt_collection, lnk_collection, [self.name + '_fixed0', self.name + '_joint0']

    def _update_jnt_fk(self, jnt_name):
        """
        update fk tree recursively
        author: weiwei
        date: 20201204osaka
        """
        p_jnt_name = self.jnt_collection[jnt_name].p_name
        cur_jnt = self.jnt_collection[jnt_name]
        # update gl_pos0 and gl_rotmat0
        if p_jnt_name is None:
            cur_jnt.gl_pos0 = cur_jnt.loc_pos
            cur_jnt.gl_rotmat0 = cur_jnt.loc_rotmat
        else:
            p_jnt = self.jnt_collection[p_jnt_name]
            curjnt_loc_pos = np.dot(p_jnt.gl_rotmatq, cur_jnt.loc_pos)
            cur_jnt.gl_pos0 = p_jnt.gl_posq + curjnt_loc_pos
            cur_jnt.gl_rotmat0 = np.dot(p_jnt.gl_rotmatq, cur_jnt.loc_rotmat)
            cur_jnt.gl_motionax = np.dot(cur_jnt.gl_rotmat0, cur_jnt.loc_motionax)
        # update gl_pos_q and gl_rotmat_q
        if cur_jnt.type == "dummy":
            cur_jnt.gl_posq = cur_jnt.gl_pos0
            cur_jnt.gl_rotmatq = cur_jnt.gl_rotmat0
        elif cur_jnt.type == "revolute":
            cur_jnt.gl_posq = cur_jnt.gl_pos0
            curjnt_loc_rotmat = rm.rotmat_from_axangle(cur_jnt.loc_motionax, cur_jnt.motion_val)
            cur_jnt.gl_rotmatq = np.dot(cur_jnt.gl_rotmat0, curjnt_loc_rotmat)
        elif cur_jnt.type == "prismatic":
            cur_jnt.gl_posq = cur_jnt.gl_pos0 + cur_jnt.motion_val * cur_jnt.loc_motionax
            cur_jnt.gl_rotmatq = cur_jnt.gl_rotmat0
        else:
            raise ValueError("The given joint type is not available!")
        for each_jnt_name in cur_jnt.chd_name_list:
            self._update_jnt_fk(each_jnt_name)

    def _update_jnt_fk_faster(self, jnt_name):
        """
        is this fastedr than _update_jnt_fk?
        author: weiwei
        date: 20201203osaka
        """
        while jnt_name is not None:
            cur_jnt = self.jnt_collection[jnt_name]
            p_jnt_name = cur_jnt.p_name
            # update gl_pos0 and gl_rotmat0
            if p_jnt_name is None:
                cur_jnt.gl_pos0 = cur_jnt.loc_pos
                cur_jnt.gl_rotmat0 = cur_jnt.loc_rotmat
            else:
                p_jnt = self.jnt_collection[p_jnt_name]
                curjnt_loc_pos = np.dot(p_jnt.gl_rotmatq, cur_jnt.loc_pos)
                cur_jnt.gl_pos0 = p_jnt.gl_posq + curjnt_loc_pos
                cur_jnt.gl_rotmat0 = np.dot(p_jnt.gl_rotmatq, cur_jnt.loc_rotmat)
                cur_jnt.gl_motionax = np.dot(cur_jnt.gl_rotmat0, cur_jnt.loc_motionax)
            # update gl_pos_q and gl_rotmat_q
            if cur_jnt.type == "dummy":
                cur_jnt.gl_posq = cur_jnt.gl_pos0
                cur_jnt.gl_rotmatq = cur_jnt.gl_rotmat0
            elif cur_jnt.type == "revolute":
                cur_jnt.gl_posq = cur_jnt.gl_pos0
                curjnt_loc_rotmat = rm.rotmat_from_axangle(cur_jnt.loc_motionax, cur_jnt.motion_val)
                cur_jnt.gl_rotmatq = np.dot(cur_jnt.gl_rotmat0, curjnt_loc_rotmat)
            elif cur_jnt.type == "prismatic":
                cur_jnt.gl_posq = cur_jnt.gl_pos0 + cur_jnt.motion_val * cur_jnt.loc_motionax
                cur_jnt.gl_rotmatq = cur_jnt.gl_rotmat0
            else:
                raise ValueError("The given joint type is not available!")
            for each_jnt_name in cur_jnt.chd_name_list:
                self._update_jnt_fk(each_jnt_name)

    def _update_lnk_fk(self, base=None):
        """
        note:
        author: weiwei
        date: 20201203osaka
        """
        for lnk_name in self.lnk_collection.keys():
            cur_lnk = self.lnk_collection[lnk_name]
            ref_jnt = self.jnt_collection[cur_lnk.refjnt_name]
            cur_lnk.gl_pos = np.dot(ref_jnt.gl_rotmatq, cur_lnk.loc_pos) + ref_jnt.gl_posq
            cur_lnk.gl_rotmat = np.dot(ref_jnt.gl_rotmatq, cur_lnk.loc_rotmat)

    def _update_fk(self, base=None):
        if base is None:
            base = self._base
        for jnt_name in base:
            for chd_name in self.jnt_collection[jnt_name].chd_name_list:
                if chd_name not in base:
                    self._update_jnt_fk(jnt_name=chd_name)
        self._update_lnk_fk()

    def fk(self, root=None, jnt_motion_vals=None):
        """
        move the joints using forward kinematics
        :param jnt_motion_vals: a nx1 list, each element indicates the value of a joint (in radian or meter);
                                Dictionary, {jnt_name: motion_val, ...}
        :return
        author: weiwei
        date: 20161205, 20201009osaka
        """
        if isinstance(jnt_motion_vals, list):
            if len(jnt_motion_vals) != len(self.tgtjnt_names):
                raise ValueError("The number of given joint motion values must be coherent with self.tgtjnt_names!")
            counter = 0
            for jnt_name in self.tgtjnt_names:
                self.jnt_collection[jnt_name].motion_val = jnt_motion_vals[counter]
                counter += 1
        if isinstance(jnt_motion_vals, dict):
            pass
        self._update_fk()

    def change_base(self, base):
        """
        two joints determine a base
        notes:
        # if a joint is a child, the family flow below it alway goes downward
        # if a joint has a parent, the family flow is not clear. The original base might be above it at some where
        # thus, we only need to reverse the parent
        # when a list jnt_names is set to be a base, the family relation among them is ignored but kept
        # they will be udpated when another list is set as the base
        :param base: an nx1 list of compactly connected nodes, comprises at least two jnt_names
        :return:
        author: weiwei
        date: 20201203
        """
        self._base = base
        for jnt_name in base:
            cur_jnt = self.jnt_collection[jnt_name]
            cur_jnt.loc_pos = cur_jnt.gl_posq
            cur_jnt.loc_rotmat = cur_jnt.gl_posq
            if cur_jnt.p_name not in base:  # each joint has only one parent joint
                p_jnt_name = cur_jnt.p_name
                cur_jnt.p_name = None
                while p_jnt_name is not None:
                    # reverse all parents of cur_jnt
                    p_jnt = self.jnt_collection[p_jnt_name]
                    p_jnt.p_name = jnt_name
                    cur_jnt.chd_name_list = cur_jnt.chd_name_list + [p_jnt_name]
                    p_jnt.chd_name_list.remove(jnt_name)
                    # reverse lnk
                    lnk_name = p_jnt.lnk_name_dict[jnt_name]
                    cur_jnt.lnk_name_dict[p_jnt_name] = lnk_name
                    p_lnk = self.lnk_collection[lnk_name]
                    p_lnk.loc_pos = p_jnt.loc_pos + np.dot(p_jnt.loc_rotmat, p_lnk.loc_pos)
                    p_lnk.loc_rotmat = np.dot(p_jnt.loc_rotmat, p_lnk.loc_rotmat)
                    # TODO
                    # p_lnk.com = com
                    # p_lnk.intertia = intertia
                    # p_lnk.mass = mass
                    # reverse rotation
                    p_jnt.loc_motionax = -p_jnt.loc_motionax
                    # iteration
                    p_jnt.lnk_name_dict.pop[jnt_name]
                    cur_jnt = p_jnt
                    p_jnt_name = cur_jnt.p_name
                    cur_jnt.p_name = None


if __name__ == '__main__':
    jl = JntLnks()
