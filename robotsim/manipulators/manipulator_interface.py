import copy
import numpy as np
import robotsim._kinematics.collisionchecker as cc


class ManipulatorInterface(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='yumi_gripper'):
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        # jlc
        self.jlc = None
        # collision detection
        self.cc = cc.CollisionChecker("collision_checker")

    @property
    def jnts(self):
        return self.jlc.jnts

    @property
    def lnks(self):
        return self.jlc.lnks

    @property
    def is_fk_updated(self):
        return self.jlc.is_fk_updated

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        :param obstacle_list:
        :param otherrobot_list:
        :return:
        author: weiwei
        date: 20201223
        """
        return self.cc.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list,
                                   need_update=self.is_fk_updated)

    def set_homeconf(self, jnt_values):
        self.jlc.set_homeconf(jnt_values=jnt_values)

    def set_tcp(self, tcp_jntid=None, tcp_loc_pos=None, tcp_loc_rotmat=None):
        if tcp_jntid is not None:
            self.jlc.tcp_jntid = tcp_jntid
        if tcp_loc_pos is not None:
            self.jlc.tcp_loc_pos = tcp_loc_pos
        if tcp_loc_rotmat is not None:
            self.jlc.tcp_loc_rotmat = tcp_loc_rotmat

    def get_gl_tcp(self,
                   tcp_jntid=None,
                   tcp_loc_pos=None,
                   tcp_loc_rotmat=None):
        return self.jlc.get_gl_tcp(tcp_jnt_id=tcp_jntid,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat)

    def get_jntranges(self):
        return self.jlc.get_jntranges()

    def goto_homeconf(self):
        self.jlc.fk(jnt_values=self.jlc.homeconf)

    def goto_zeroconf(self):
        self.jlc.fk(jnt_values=self.jlc.zeroconf)

    def get_jntvalues(self):
        return self.jlc.get_jntvalues()

    def rand_conf(self):
        return self.jlc.rand_conf()

    def num_ik(self,
               tgt_pos,
               tgt_rot,
               start_conf=None,
               tcp_jntid=None,
               tcp_loc_pos=None,
               tcp_loc_rotmat=None,
               local_minima="accept",
               toggle_debug=False):
        return self.jlc.numik(tgt_pos=tgt_pos,
                              tgt_rot=tgt_rot,
                              start_conf=start_conf,
                              tcp_jntid=tcp_jntid,
                              tcp_loc_pos=tcp_loc_pos,
                              tcp_loc_rotmat=tcp_loc_rotmat,
                              local_minima=local_minima,
                              toggle_debug=toggle_debug)

    def get_gl_pose(self,
                    loc_pos=np.zeros(3),
                    loc_rotmat=np.eye(3),
                    tcp_jntid=None,
                    tcp_loc_pos=None,
                    tcp_loc_rotmat=None):
        return self.jlc.get_gl_pose(loc_pos=loc_pos,
                                    loc_rotmat=loc_rotmat,
                                    tcp_jntid=tcp_jntid,
                                    tcp_loc_pos=tcp_loc_pos,
                                    tcp_loc_rotmat=tcp_loc_rotmat)

    def get_loc_pose(self,
                     gl_pos,
                     gl_rotmat,
                     tcp_jntid=None,
                     tcp_loc_pos=None,
                     tcp_loc_rotmat=None):
        return self.jlc.get_loc_pose(gl_pos=gl_pos,
                                     gl_rotmat=gl_rotmat,
                                     tcp_jntid=tcp_jntid,
                                     tcp_loc_pos=tcp_loc_pos,
                                     tcp_loc_rotmat=tcp_loc_rotmat)

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        return self.cc.is_collided(obstacle_list=obstacle_list,
                                   otherrobot_list=otherrobot_list,
                                   need_update = self.is_fk_updated)

    def fix_to(self, pos, rotmat):
        self.jlc.fix_to(pos=pos, rotmat=rotmat)

    def fk(self, jnt_values):
        return self.jlc.fk(jnt_values=jnt_values)

    def show_cdprimit(self):
        self.cc.show_cdprimit(need_update=self.is_fk_updated)

    def unshow_cdprimit(self):
        self.cc.unshow_cdprimit()

    def gen_meshmodel(self,
                      tcp_jntid=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=True,
                      toggle_jntscs=False,
                      rgba=None,
                      name='manipulator_mesh'):
        return self.jlc._mt.gen_meshmodel(tcp_jntid=tcp_jntid,
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
                       tcp_jntid=None,
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
                                           tcp_jntid=tcp_jntid,
                                           tcp_loc_pos=tcp_loc_pos,
                                           tcp_loc_rotmat=tcp_loc_rotmat,
                                           toggle_tcpcs=toggle_tcpcs,
                                           toggle_jntscs=toggle_jntscs,
                                           toggle_connjnt=toggle_connjnt,
                                           name=name)

    def gen_endsphere(self):
        return self.jlc._mt.gen_endsphere()

    def disable_cc(self):
        """
        clear pairs and nodepath
        :return:
        """
        for cdelement in self.cc.all_cdelements:
            cdelement['cdprimit_childid'] = -1
        self.cc.all_cdelements = []
        for child in self.cc.np.getChildren():
            child.removeNode()
        self.cc.nbitmask = 0

    def copy(self):
        self_copy = copy.deepcopy(self)
        # deepcopying colliders are problematic, I have to update it manually
        for child in self_copy.cc.np.getChildren(): # empty NodePathCollection if the np does not have a child
            self_copy.cc.ctrav.addCollider(child, self_copy.cc.chan)
        return self_copy