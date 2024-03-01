import copy
import numpy as np
import modeling.model_collection as mmc
import modeling.geometric_model as mgm
import robot_sim._kinematics.jlchain as jl
import robot_sim._kinematics.TBD_collision_checker as cc
import robot_sim._kinematics.model_generator as rkmg


class ManipulatorInterface(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), home_conf=np.zeros(6), name='manipulator'):
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        # jlc
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, n_dof=len(home_conf), name=name)
        self.jlc.home = home_conf
        # flange
        self.loc_flange_pos = np.zeros(3)
        self.loc_flange_rotmat = np.eye(3)
        # collision detection
        self.cc = None

    @property
    def jnts(self):
        return self.jlc.jnts

    # @property
    # def tgtjnts(self):
    #     return self.jlc.tgtjnts

    @property
    def n_dof(self):
        return self.jlc.n_dof

    @property
    def home_conf(self):
        return self.jlc.home

    @property
    def tcp_jnt_id(self):
        return self.jlc.functional_jnt_id

    @property
    def loc_tcp_pos(self):
        return self.jlc.loc_tcp_pos

    @property
    def loc_tcp_rotmat(self):
        return self.jlc.loc_tcp_rotmat

    @property
    def jnt_ranges(self):
        return self.jlc.jnt_ranges

    @tcp_jnt_id.setter
    def tcp_jnt_id(self, tcp_jnt_id):
        self.jlc.functional_jnt_id = tcp_jnt_id

    @loc_tcp_pos.setter
    def loc_tcp_pos(self, loc_tcp_pos):
        self.jlc.loc_tcp_pos = loc_tcp_pos

    @loc_tcp_rotmat.setter
    def loc_tcp_rotmat(self, loc_tcp_rotmat):
        self.jlc.loc_tcp_rotmat = loc_tcp_rotmat

    @home_conf.setter
    def home_conf(self, home_conf):
        self.jlc.home = home_conf

    def set_tcp(self, loc_tcp_pos=None, loc_tcp_rotmat=None):
        self.jlc.set_tcp(loc_tcp_pos=loc_tcp_pos, loc_tcp_rotmat=loc_tcp_rotmat)

    def get_gl_tcp(self):
        return self.jlc.get_gl_tcp()

    def get_gl_flange(self):
        gl_flange_pos = self.jnts[self.tcp_jnt_id].gl_pos_q + self.jnts[
            self.tcp_jnt_id].gl_rotmat_q @ self.loc_flange_pos
        gl_flange_rotmat = self.jnts[self.tcp_jnt_id].gl_rotmat_q @ self.loc_flange_rotmat
        return gl_flange_pos, gl_flange_rotmat

    def goto_given_conf(self, jnt_values):
        return self.jlc.go_given_conf(jnt_values=jnt_values)

    def goto_home_conf(self):
        return self.jlc.go_home()

    def goto_zero_conf(self):
        return self.jlc.go_zero()

    def fix_to(self, pos, rotmat, jnt_values=None):
        return self.jlc.fix_to(pos=pos, rotmat=rotmat, jnt_values=jnt_values)

    def are_jnts_in_ranges(self, jnt_values):
        return self.jlc.are_jnts_in_ranges(jnt_values)

    def fk(self, jnt_values, toggle_jacobian=True):
        return self.jlc.fk(jnt_values=jnt_values, toggle_jacobian=toggle_jacobian, update=False)

    def get_jnt_values(self):
        return self.jlc.get_jnt_values()

    def rand_conf(self):
        return self.jlc.rand_conf()

    def ik(self,
           tgt_pos: np.ndarray,
           tgt_rotmat: np.ndarray,
           seed_jnt_values=None,
           toggle_dbg=False):
        """
        by default the function calls the numerical implementation of jlc
        override this function in case of analytical IK; ignore the unrequired parameters
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param toggle_dbg:
        :return:
        """
        return self.jlc.ik(tgt_pos=tgt_pos,
                           tgt_rotmat=tgt_rotmat,
                           seed_jnt_values=seed_jnt_values,
                           toggle_dbg=toggle_dbg)

    def manipulability_val(self):
        return self.jlc.manipulability_val()

    def manipulability_mat(self):
        return self.jlc.manipulability_mat()

    def jacobian(self, jnt_values=None):
        return self.jlc.jacobian(jnt_values=jnt_values)

    def cvt_loc_tcp_to_gl(self):
        return self.jlc.cvt_loc_tcp_to_gl()

    def cvt_gl_pose_to_tcp(self, gl_pos, gl_rotmat):
        return self.jlc.cvt_gl_pose_to_tcp(gl_pos=gl_pos,
                                           gl_rotmat=gl_rotmat)

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
        self.cc.show_cdprim()

    def unshow_cdprimit(self):
        self.cc.unshow_cdprim()

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=True,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name="manipulator_mesh"):
        return rkmg.gen_jlc_mesh(self.jlc,
                                 rgb=rgb,
                                 alpha=alpha,
                                 toggle_tcp_frame=toggle_tcp_frame,
                                 toggle_jnt_frames=toggle_jnt_frames,
                                 toggle_flange_frame=toggle_flange_frame,
                                 toggle_cdprim=toggle_cdprim,
                                 toggle_cdmesh=toggle_cdmesh,
                                 name=name)

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name="manipulator_stickmodel"):
        return rkmg.gen_jlc_stick(self.jlc,
                                  toggle_jnt_frames=toggle_jnt_frames,
                                  toggle_tcp_frame=toggle_tcp_frame,
                                  toggle_flange_frame=toggle_flange_frame,
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
        for cdelement in self.cc.cce_dict:
            cdelement['cdprimit_childid'] = -1
        self.cc = None
        # self.cc.cce_dict = []
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
