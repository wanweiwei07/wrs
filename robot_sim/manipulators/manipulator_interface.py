import copy
import numpy as np
import robot_sim._kinematics.collision_checker as cc


class ManipulatorInterface(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='yumi_gripper'):
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        # jlc
        self.jlc = None
        # collision detection
        self.cc = None

    @property
    def jnts(self):
        return self.jlc.jnts

    @property
    def lnks(self):
        return self.jlc.lnks

    @property
    def tgtjnts(self):
        return self.jlc.tgtjnts

    @property
    def ndof(self):
        return self.jlc.n_dof

    @property
    def homeconf(self):
        return self.jlc.homeconf

    @property
    def tcp_jnt_id(self):
        return self.jlc.tcp_jnt_id

    @property
    def tcp_loc_pos(self):
        return self.jlc.tcp_loc_pos

    @property
    def tcp_loc_rotmat(self):
        return self.jlc.tcp_loc_rotmat

    @tcp_jnt_id.setter
    def tcp_jnt_id(self, tcp_jnt_id):
        self.jlc.tcp_jnt_id = tcp_jnt_id

    @tcp_loc_pos.setter
    def tcp_loc_pos(self, tcp_loc_pos):
        self.jlc.tcp_loc_pos = tcp_loc_pos

    @tcp_loc_rotmat.setter
    def tcp_loc_rotmat(self, tcp_loc_rotmat):
        self.jlc.tcp_loc_rotmat = tcp_loc_rotmat

    def set_homeconf(self, jnt_values):
        self.jlc.set_home(jnt_values=jnt_values)

    def set_tcp(self, tcp_jnt_id=None, tcp_loc_pos=None, tcp_loc_rotmat=None):
        if tcp_jnt_id is not None:
            self.jlc.tcp_jnt_id = tcp_jnt_id
        if tcp_loc_pos is not None:
            self.jlc.tcp_loc_pos = tcp_loc_pos
        if tcp_loc_rotmat is not None:
            self.jlc.tcp_loc_rotmat = tcp_loc_rotmat

    def get_gl_tcp(self,
                   tcp_jnt_id=None,
                   tcp_loc_pos=None,
                   tcp_loc_rotmat=None):
        return self.jlc.get_gl_tcp(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat)

    def get_jnt_ranges(self):
        return self.jlc.jnt_rngs

    def goto_homeconf(self):
        self.jlc.fk(joint_values=self.jlc.homeconf)

    def goto_zeroconf(self):
        self.jlc.fk(joint_values=self.jlc.zeroconf)

    def fix_to(self, pos, rotmat, jnt_values=None):
        return self.jlc.fix_to(pos=pos, rotmat=rotmat, jnt_values=jnt_values)

    def is_jnt_values_in_ranges(self, jnt_values):
        return self.jlc.are_joint_values_in_ranges(jnt_values)

    def fk(self, jnt_values):
        if jnt_values is None:
            raise Exception("Joint values are None!")
        return self.jlc.fk(joint_values=jnt_values)

    def get_jnt_values(self):
        return self.jlc.get_joint_values()

    def rand_conf(self):
        return self.jlc.rand_conf()

    def ik(self,
           tgt_pos: np.ndarray,
           tgt_rotmat: np.ndarray,
           tcp_loc_pos: np.ndarray = None,
           tcp_loc_rotmat: np.ndarray = None,
           tcp_jnt_id=None,
           seed_jnt_values=None,
           max_niter=100,
           local_minima="accept",
           toggle_debug=False):
        """
        by default the function calls the numerical implementation of jlc
        override this function in case of analytical IK; ignore the unrequired parameters
        :param tgt_pos:
        :param tgt_rotmat:
        :param tcp_loc_pos:
        :param tcp_loc_rotmat:
        :param tcp_jnt_id:
        :param seed_jnt_values:
        :param max_niter:
        :param local_minima:
        :param toggle_debug:
        :return:
        """
        return self.jlc.ik(tgt_pos=tgt_pos,
                           tgt_rotmat=tgt_rotmat,
                           tcp_loc_pos=tcp_loc_pos,
                           tcp_loc_rotmat=tcp_loc_rotmat,
                           tcp_joint_id=tcp_jnt_id,
                           seed_jnt_vals=seed_jnt_values,
                           max_n_iter=max_niter,
                           local_minima=local_minima,
                           toggle_dbg=toggle_debug)

    def manipulability(self,
                       tcp_jnt_id,
                       tcp_loc_pos,
                       tcp_loc_rotmat):
        return self.jlc.manipulability(tcp_jnt_id=tcp_jnt_id,
                                       tcp_loc_pos=tcp_loc_pos,
                                       tcp_loc_rotmat=tcp_loc_rotmat)

    def manipulability_axmat(self,
                             tcp_jnt_id,
                             tcp_loc_pos,
                             tcp_loc_rotmat, type="translational"):
        return self.jlc.manipulability_axmat(tcp_jnt_id=tcp_jnt_id,
                                             tcp_loc_pos=tcp_loc_pos,
                                             tcp_loc_rotmat=tcp_loc_rotmat,
                                             type=type)

    def jacobian(self,
                 tcp_jnt_id,
                 tcp_loc_pos,
                 tcp_loc_rotmat):
        return self.jlc.jacobian(tcp_joint_id=tcp_jnt_id,
                                 tcp_loc_pos=tcp_loc_pos,
                                 tcp_loc_rotmat=tcp_loc_rotmat)

    def cvt_loc_tcp_to_gl(self,
                          loc_pos=np.zeros(3),
                          loc_rotmat=np.eye(3),
                          tcp_jnt_id=None,
                          tcp_loc_pos=None,
                          tcp_loc_rotmat=None):
        return self.jlc.cvt_loc_tcp_to_gl(loc_pos=loc_pos,
                                          loc_rotmat=loc_rotmat,
                                          tcp_jnt_id=tcp_jnt_id,
                                          tcp_loc_pos=tcp_loc_pos,
                                          tcp_loc_rotmat=tcp_loc_rotmat)

    def cvt_gl_to_loc_tcp(self,
                          gl_pos,
                          gl_rotmat,
                          tcp_jnt_id=None,
                          tcp_loc_pos=None,
                          tcp_loc_rotmat=None):
        return self.jlc.cvt_gl_to_loc_tcp(gl_pos=gl_pos,
                                          gl_rotmat=gl_rotmat,
                                          tcp_jnt_id=tcp_jnt_id,
                                          tcp_loc_pos=tcp_loc_pos,
                                          tcp_loc_rotmat=tcp_loc_rotmat)

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        :param obstacle_list:
        :param otherrobot_list:
        :return:
        author: weiwei
        date: 20201223
        """
        return self.cc.is_collided(obstacle_list=obstacle_list,
                                   otherrobot_list=otherrobot_list)

    def show_cdprimit(self):
        self.cc.show_cdprimit()

    def unshow_cdprimit(self):
        self.cc.unshow_cdprimit()

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=True,
                      toggle_jntscs=False,
                      rgba=None,
                      name='manipulator_mesh'):
        return self.jlc._mt.gen_mesh_model(tcp_jnt_id=tcp_jnt_id,
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
        return self.jlc._mt.gen_stickmodel(rgba=rgba,
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
        return self.jlc._mt.gen_endsphere()

    def enable_cc(self):
        self.cc = cc.CollisionChecker("collision_checker")

    def disable_cc(self):
        """
        clear pairs and pdndp
        :return:
        """
        for cdelement in self.cc.all_cd_elements:
            cdelement['cdprimit_childid'] = -1
        self.cc = None
        # self.cc.all_cd_elements = []
        # for child in self.cc.np.getChildren():
        #     child.removeNode()
        # self.cc.nbitmask = 0

    def copy(self):
        self_copy = copy.deepcopy(self)
        # deepcopying colliders are problematic, I have to update it manually
        if self.cc is not None:
            for child in self_copy.cc.np.getChildren():  # empty NodePathCollection if the np does not have a child
                self_copy.cc.cd_trav.addCollider(child, self_copy.cc.cd_handler)
        return self_copy
