import math
import numpy as np
import basis.robotmath as rm


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
                 chd_name=None):
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
        self.p_name = p_name # a single value
        self.chd_name = chd_name # a single value or a list


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
        self.refjnt_name = refjnt_name
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


class JntLnks(object):
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
        self.jnt_collection, self.lnk_collection = self._initjntlnks()
        self.tgtjnt_ids = ['joint' + str(id) for id in range(self.ndof)]
        self._root = self.name + 'fixed0'

    def _initgraph(self):
        # joints
        jnt_collection = {}
        jnt_name = self.name + '_fixed0'
        jnt_collection[jnt_name] = Jnt()
        jnt_collection[jnt_name].p_name = None
        jnt_collection[jnt_name].chd_name = self.name + '_joint0'
        for id in range(self.ndof):
            jnt_name = self.name + '_joint' + str(id)
            jnt_collection[jnt_name] = Jnt()
            if id == 0:
                jnt_collection[jnt_name].p_name = self.name + '_fixed0'
                jnt_collection[jnt_name].chd_name = self.name + '_joint1'
            elif id == self.ndof - 1:
                jnt_collection[jnt_name].p_name = self.name + '_joint' + str(id - 1)
                jnt_collection[jnt_name].chd_name = self.name + '_fixed1'
            else:
                jnt_collection[jnt_name].p_name = self.name + '_joint' + str(id - 1)
                jnt_collection[jnt_name].chd_name = self.name + '_joint' + str(id + 1)
        jnt_collection[self.name + '_fixed1'] = Jnt()
        jnt_collection[self.name + '_fixed1'].p_name = self.name + '_joint' + str(self.ndof - 1)
        jnt_collection[self.name + '_fixed1'].chd_name = None
        # links
        lnk_collection = {}
        lnk_name = self.name + '_link_base'
        lnk_collection[lnk_name] = Lnk()
        lnk_collection[lnk_name].refjnt_name = self.name + '_fixed0'
        for id in range(self.ndof):
            name = self.name + '_link' + str(id)
            lnk_collection[name] = Lnk()
            lnk_collection[name].refjnt_name = self.name + '_joint' + str(id)
        return jnt_collection, lnk_collection

    def _update_jnt_fk(self, jnt_name):
        """
        author: weiwei
        date: 20201203osaka
        """
        while jnt_name is not None:
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
            jnt_name = cur_jnt.chd_name
            if isinstance(jnt_name, list):
                for each_jnt_name in jnt_name:
                    self._updatefk(each_jnt_name)

    def _update_lnk_fk(self):
        """
        author: weiwei
        date: 20201203osaka
        """
        for lnk_name in self.lnk_collection.keys():
            cur_lnk = self.lnk_collection[lnk_name]
            ref_jnt = self.jnt_collection[cur_lnk.refjnt_name]
            cur_lnk.gl_pos = np.dot(ref_jnt.gl_rotmatq, cur_lnk.loc_pos) + ref_jnt.gl_posq
            cur_lnk.gl_rotmat = np.dot(ref_jnt.gl_rotmatq, cur_lnk.loc_rotmat)

    def fk(self, jnt_motion_vals=None):
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
        self._updatefk()

    def change_base(self, jnt_name):
        """
        change the base of the manipulator to the given joint_name
        :param jnt_name: str
        :return:
        author: weiwei
        date: 20201203
        """

        # successor
        try:
            successor_iter = self.graph.successors(jnt_name)
            for successor in successor_iter:
                self.graph.successors(successor)
        except NetworkXError:
            pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    jlg = JLGraph()
    nx.draw(jlg.graph, with_labels=True, arrows=True)
    plt.title('draw_networkx')
    plt.show()
