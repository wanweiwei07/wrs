import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim._kinematics.collision_checker as cc


class ManipulatorInterface(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), home_conf=np.zeros(6), name='manipulator', enable_cc=False):
        self.name = name
        # jlc
        self.jlc = rkjlc.JLChain(pos=pos, rotmat=rotmat, n_dof=len(home_conf), name=name)
        self.jlc.home = home_conf
        # tcp is defined locally in flange
        self._loc_tcp_pos = np.zeros(3, dtype=float)
        self._loc_tcp_rotmat = np.eye(3, dtype=float)
        self._is_gl_tcp_delayed = True
        self._gl_tcp_pos = np.zeros(3, dtype=float)
        self._gl_tcp_rotmat = np.eye(3, dtype=float)
        # TODO self.ooflange = []
        # collision detection
        if enable_cc:
            self.cc = cc.CollisionChecker("collision_checker")
        else:
            self.cc = None
        # backup
        self.jnt_values_bk = []

    # delays
    @staticmethod
    def delay_gl_tcp_decorator(method):
        def wrapper(self, *args, **kwargs):
            self._is_gl_tcp_delayed = True
            return method(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def update_gl_tcp_decorator(method):
        def wrapper(self, *args, **kwargs):
            if self._is_gl_tcp_delayed:
                self._gl_tcp_pos, self._gl_tcp_rotmat = self.compute_gl_tcp()
                self._is_gl_tcp_delayed = False
            return method(self, *args, **kwargs)

        return wrapper

    @property
    def pos(self):
        return self.jlc.pos

    @property
    def rotmat(self):
        return self.jlc.rotmat

    @property
    def jnts(self):
        return self.jlc.jnts

    @property
    def n_dof(self):
        return self.jlc.n_dof

    @property
    def home_conf(self):
        return self.jlc.home

    @home_conf.setter
    def home_conf(self, conf):
        self.jlc.home = conf
        self.goto_home_conf()

    @property
    def loc_tcp_pos(self):
        return self._loc_tcp_pos

    @loc_tcp_pos.setter
    @delay_gl_tcp_decorator
    def loc_tcp_pos(self, loc_tcp_pos):
        self._loc_tcp_pos = loc_tcp_pos

    @property
    def loc_tcp_rotmat(self):
        return self._loc_tcp_rotmat

    @loc_tcp_rotmat.setter
    @delay_gl_tcp_decorator
    def loc_tcp_rotmat(self, loc_tcp_rotmat):
        self._loc_tcp_rotmat = loc_tcp_rotmat

    @property
    @update_gl_tcp_decorator
    def gl_tcp_pos(self):
        return self._gl_tcp_pos

    @property
    @update_gl_tcp_decorator
    def gl_tcp_rotmat(self):
        return self._gl_tcp_rotmat

    @property
    def gl_flange_pos(self):
        return self.jlc.gl_flange_pos

    @property
    def gl_flange_rotmat(self):
        return self.jlc.gl_flange_rotmat

    @property
    def jnt_ranges(self):
        return self.jlc.jnt_ranges

    def _is_jnt_in_range(self, jnt_id, jnt_value):
        """

        :param jnt_id:
        :param jnt_value:
        :return:
        author: weiwei
        date: 20230801
        """
        if jnt_value < self.jlc.jnts[jnt_id].motion_range[0] or jnt_value > self.jlc.jnts[jnt_id].motion_range[1]:
            print(f"Error: Joint {jnt_id} is out of range!")
            return False
        else:
            return True

    def backup_state(self):
        self.jnt_values_bk.append(self.jlc.get_jnt_values())

    def restore_state(self):
        self.goto_given_conf(jnt_values=self.jnt_values_bk.pop())

    def clear_cc(self):
        if self.cc is None:
            print("The cc is currently unavailable. Nothing to clear.")
        else:
            # create a new cc and delete the original one
            self.cc = cc.CollisionChecker("collision_checker")

    def compute_gl_tcp(self):
        gl_tcp_pos = self.jlc.gl_flange_pos + self.jlc.gl_flange_rotmat @ self._loc_tcp_pos
        gl_tcp_rotmat = self.jlc.gl_flange_rotmat @ self._loc_tcp_rotmat
        return (gl_tcp_pos, gl_tcp_rotmat)

    def goto_given_conf(self, jnt_values):
        return self.fk(jnt_values=jnt_values, toggle_jacobian=False, update=True)

    def goto_home_conf(self):
        return self.goto_given_conf(self.home_conf)

    def fix_to(self, pos, rotmat, jnt_values=None):
        self.jlc.fix_to(pos=pos, rotmat=rotmat, jnt_values=jnt_values)
        self._gl_tcp_pos, self._gl_tcp_rotmat = self.compute_gl_tcp()
        return self.gl_tcp_pos, self.gl_tcp_rotmat

    def fk(self, jnt_values, toggle_jacobian=False, update=False):
        jlc_result = self.jlc.fk(jnt_values=jnt_values, toggle_jacobian=toggle_jacobian, update=update)
        gl_flange_pos = jlc_result[0]
        gl_flange_rotmat = jlc_result[1]
        gl_tcp_pos = gl_flange_pos + gl_flange_rotmat @ self.loc_tcp_pos
        gl_tcp_rotmat = gl_flange_rotmat @ self.loc_tcp_rotmat
        if update:
            self._gl_tcp_pos = gl_tcp_pos
            self._gl_tcp_rotmat = gl_tcp_rotmat
        return (gl_tcp_pos, gl_tcp_rotmat)

    def are_jnts_in_ranges(self, jnt_values):
        return self.jlc.are_jnts_in_ranges(jnt_values)

    def get_jnt_values(self):
        return self.jlc.get_jnt_values()

    def rand_conf(self):
        return self.jlc.rand_conf()

    def ik(self,
           tgt_pos: np.ndarray,
           tgt_rotmat: np.ndarray,
           seed_jnt_values=None,
           option="empty",
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
        tgt_rotmat = tgt_rotmat @ self.loc_tcp_rotmat.T
        tgt_pos = tgt_pos - tgt_rotmat @ self.loc_tcp_pos
        return self.jlc.ik(tgt_pos=tgt_pos,
                           tgt_rotmat=tgt_rotmat,
                           seed_jnt_values=seed_jnt_values,
                           toggle_dbg=toggle_dbg)

    def jacobian(self, jnt_values=None):
        if jnt_values is None:
            j_mat = np.zeros((6, self.jlc.n_dof))
            for i in range(self.jlc.flange_jnt_id + 1):
                if self.jlc.jnts[i].type == rkjlc.const.JntType.REVOLUTE:
                    j2t_vec = self.gl_tcp_pos - self.jlc.jnts[i].gl_pos_q
                    j_mat[:3, i] = np.cross(self.jlc.jnts[i].gl_motion_ax, j2t_vec)
                    j_mat[3:6, i] = self.jlc.jnts[i].gl_motion_ax
                if self.jlc.jnts[i].type == rkjlc.const.JntType.PRISMATIC:
                    j_mat[:3, i] = self.jlc.jnts[i].gl_motion_ax
            return j_mat
        else:
            homomat = self.jlc.anchor.gl_flange_homomat
            jnt_pos = np.zeros((self.jlc.n_dof, 3))
            jnt_motion_ax = np.zeros((self.jlc.n_dof, 3))
            for i in range(self.jlc.flange_jnt_id + 1):
                jnt_pos[i, :] = homomat[:3, 3] + homomat[:3, :3] @ self.jlc.jnts[i].loc_pos
                homomat = homomat @ self.jlc.jnts[i].get_motion_homomat(motion_value=jnt_values[i])
                jnt_motion_ax[i, :] = homomat[:3, :3] @ self.jlc.jnts[i].loc_motion_ax
            gl_flange_homomat = homomat @ self.jlc.loc_flange_homomat
            gl_flange_pos = gl_flange_homomat[:3, 3]
            gl_flange_rotmat = gl_flange_homomat[:3, :3]
            gl_tcp_pos = gl_flange_pos + gl_flange_rotmat @ self.loc_tcp_pos
            j_mat = np.zeros((6, self.jlc.n_dof))
            for i in range(self.jlc.flange_jnt_id + 1):
                if self.jlc.jnts[i].type == rkjlc.const.JntType.REVOLUTE:
                    j2t_vec = gl_tcp_pos - jnt_pos[i, :]
                    j_mat[:3, i] = np.cross(jnt_motion_ax[i, :], j2t_vec)
                    j_mat[3:6, i] = jnt_motion_ax[i, :]
                if self.jlc.jnts[i].type == rkjlc.const.JntType.PRISMATIC:
                    j_mat[:3, i] = jnt_motion_ax[i, :]
            return j_mat

    def manipulability_val(self):
        j_mat = self.jacobian()
        return np.sqrt(np.linalg.det(j_mat @ j_mat.T))

    def manipulability_mat(self):
        j_mat = self.jacobian()
        # linear ellipsoid
        linear_j_dot_jt = j_mat[:3, :] @ j_mat[:3, :].T
        eig_values, eig_vecs = np.linalg.eig(linear_j_dot_jt)
        linear_ellipsoid_mat = np.eye(3)
        linear_ellipsoid_mat[:, 0] = np.sqrt(eig_values[0]) * eig_vecs[:, 0]
        linear_ellipsoid_mat[:, 1] = np.sqrt(eig_values[1]) * eig_vecs[:, 1]
        linear_ellipsoid_mat[:, 2] = np.sqrt(eig_values[2]) * eig_vecs[:, 2]
        # angular ellipsoid
        angular_j_dot_jt = j_mat[3:, :] @ j_mat[3:, :].T
        eig_values, eig_vecs = np.linalg.eig(angular_j_dot_jt)
        angular_ellipsoid_mat = np.eye(3)
        angular_ellipsoid_mat[:, 0] = np.sqrt(eig_values[0]) * eig_vecs[:, 0]
        angular_ellipsoid_mat[:, 1] = np.sqrt(eig_values[1]) * eig_vecs[:, 1]
        angular_ellipsoid_mat[:, 2] = np.sqrt(eig_values[2]) * eig_vecs[:, 2]
        return (linear_ellipsoid_mat, angular_ellipsoid_mat)

    def cvt_pose_in_tcp_to_gl(self, loc_pos=np.zeros(3), loc_rotmat=np.eye(3)):
        gl_pos = self.gl_tcp_pos + self.gl_tcp_rotmat.dot(loc_pos)
        gl_rotmat = self.gl_tcp_rotmat.dot(loc_rotmat)
        return (gl_pos, gl_rotmat)

    def cvt_gl_pose_to_tcp(self, gl_pos, gl_rotmat):
        return rm.rel_pose(self.gl_tcp_pos, self.gl_tcp_rotmat, gl_pos, gl_rotmat)

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=True,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name="manipulator_mesh"):
        m_col = self.jlc.gen_meshmodel(rgb=rgb,
                                       alpha=alpha,
                                       toggle_jnt_frames=toggle_jnt_frames,
                                       toggle_flange_frame=toggle_flange_frame,
                                       toggle_cdprim=toggle_cdprim,
                                       toggle_cdmesh=toggle_cdmesh,
                                       name=name)
        if toggle_tcp_frame:
            rkjlc.rkmg.gen_indicated_frame(spos=self.jlc.gl_flange_pos, gl_pos=self.gl_tcp_pos,
                                           gl_rotmat=self.gl_tcp_rotmat).attach_to(m_col)
        return m_col

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name="manipulator_stickmodel"):
        m_col = self.jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                        toggle_flange_frame=toggle_flange_frame,
                                        name=name)
        if toggle_tcp_frame:
            rkjlc.rkmg.gen_indicated_frame(spos=self.jlc.gl_flange_pos, gl_pos=self.gl_tcp_pos,
                                           gl_rotmat=self.gl_tcp_rotmat).attach_to(m_col)
        return m_col

    def gen_endsphere(self):
        return mgm.gen_sphere(pos=self.gl_tcp_pos)

    ## member functions for cdprimit collisions

    def is_collided(self, obstacle_list=[], otherrobot_list=[], toggle_contacts=False):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        :param obstacle_list:
        :param otherrobot_list:
        :param toggle_contacts: if True, returns collision_result, contact_points; if False, only collision_reuls
        :return:
        author: weiwei
        date: 20201223
        """
        if self.cc is None:  # TODO assertion decorator
            raise ValueError("Collision checker is not enabled!")
        return self.cc.is_collided(obstacle_list=obstacle_list,
                                   other_robot_list=otherrobot_list, toggle_contacts=toggle_contacts)

    def show_cdprim(self):
        """
        draw cdprim to base, you can use this function to double check if tf was correct
        :return:
        """
        if self.cc is None:
            raise ValueError("Collision checker is not enabled!")
        self.cc.show_cdprim()

    def unshow_cdprim(self):
        if self.cc is None:
            raise ValueError("Collision checker is not enabled!")
        self.cc.unshow_cdprim()
