import math
import copy
import numpy as np

import basis.constant
import basis.constant as bc
import basis.robot_math as rm
import modeling.collision_model as cm
import robot_sim.kinematics.constant as rkc
import robot_sim.kinematics.jl as rkjl
import robot_sim.kinematics.numik_solver as rkn
import robot_sim.kinematics.optik_solver as rko
import robot_sim.kinematics.tracik_solver as rkt
import basis.constant as cst


class JLChain(object):
    """
    Joint Link Chain, no branches allowed
    Usage:
    1. Create a JLChain instance with a given n_dof and update its parameters for particular definition
    2. Define multiple instances of this class to compose a complicated structure
    3. Use mimic for underactuated or coupled mechanism
    """

    def __init__(self,
                 name="auto",
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 n_dof=6,
                 cdprimitive_type=cm.CDPrimitiveType.BOX,
                 cdmesh_type=cm.CDMeshType.DEFAULT):
        """
        conf -- configuration: target joint values
        :param name:
        :param pos:
        :param rotmat:
        :param home: number of joints
        :param cdprimitive_type:
        :param cdmesh_type:
        :param name:
        """
        self.name = name
        self.n_dof = n_dof
        self.home = np.zeros(self.n_dof)  # self.n_dof+1 joints in total, the first joint is a anchor joint
        # initialize joints and links
        self.anchor = rkjl.Anchor(name, pos=pos, rotmat=rotmat)
        self.joints = [rkjl.create_joint_with_link(joint_name=f"j{i}", link_name=f"l{i}") for i in range(self.n_dof)]
        self._joint_ranges = self._get_joint_ranges()
        # default tcp
        self._tcp_joint_id = self.n_dof - 1
        self.tcp_loc_pos = np.zeros(3)
        self.tcp_loc_rotmat = np.eye(3)
        # initialize
        self.go_home()
        # collision primitives
        # mesh generator
        self.cdprimitive_type = cdprimitive_type
        self.cdmesh_type = cdmesh_type
        # ik solver
        self._tracik_solver = rkt.TracIKSolver(self)

    @property
    def joint_ranges(self):
        return self._joint_ranges

    @property
    def tcp_joint_id(self):
        return self._tcp_joint_id

    @tcp_joint_id.setter
    def tcp_joint_id(self, value):
        self._tcp_joint_id = value

    @property
    def tcp_loc_homomat(self):
        return rm.homomat_from_posrot(pos=self.tcp_loc_pos, rotmat=self.tcp_loc_rotmat)

    @property
    def pos(self):
        return self.anchor.pos

    @property
    def rotmat(self):
        return self.anchor.rotmat

    def _get_joint_ranges(self):
        """
        get jntsrnage
        :return: [[jnt1min, jnt1max], [jnt2min, jnt2max], ...]
        date: 20180602, 20200704osaka
        author: weiwei
        """
        jnt_limits = []
        for i in range(self.n_dof):
            jnt_limits.append(self.joints[i].motion_range)
        return np.asarray(jnt_limits)

    def forward_kinematics(self, joint_values, toggle_jacobian=True, update=False, toggle_debug=False):
        """
        This function will update the global parameters
        :param joint_values: a 1xn ndarray where each element indicates the value of a joint (in radian or meter)
        :param update if True, update internal values
        :return: True (succ), False (failure)
        author: weiwei
        date: 20161202, 20201009osaka, 20230823
        """
        if not update:
            homomat = self.anchor.homomat
            j_pos = np.zeros((self.n_dof, 3))
            j_axis = np.zeros((self.n_dof, 3))
            for i in range(self.tcp_joint_id + 1):
                j_axis[i, :] = homomat[:3, :3] @ self.joints[i].loc_motion_axis
                if self.joints[i].type == rkc.JointType.REVOLUTE:
                    j_pos[i, :] = homomat[:3, 3] + homomat[:3, :3] @ self.joints[i].loc_pos
                homomat = homomat @ self.joints[i].get_motion_homomat(motion_value=joint_values[i])
            tcp_gl_homomat = homomat @ self.tcp_loc_homomat
            tcp_gl_pos = tcp_gl_homomat[:3, 3]
            tcp_gl_rotmat = tcp_gl_homomat[:3, :3]
            if toggle_jacobian:
                j_mat = np.zeros((6, self.n_dof))
                for i in range(self.tcp_joint_id + 1):
                    if self.joints[i].type == rkc.JointType.REVOLUTE:
                        vec_jnt2tcp = tcp_gl_pos - j_pos[i, :]
                        j_mat[:3, i] = np.cross(j_axis[i, :], vec_jnt2tcp)
                        j_mat[3:6, i] = j_axis[i, :]
                        if toggle_debug:
                            gm.gen_arrow(spos=j_pos[i, :],
                                         epos=j_pos[i, :] + .2 * j_axis[i, :],
                                         rgba=bc.black).attach_to(base)
                    if self.joints[i].type == rkc.JointType.PRISMATIC:
                        j_mat[:3, i] = j_axis[i, :]
                return tcp_gl_pos, tcp_gl_rotmat, j_mat
            else:
                return tcp_gl_pos, tcp_gl_rotmat
        else:
            pos = self.anchor.pos
            rotmat = self.anchor.rotmat
            for i in range(self.n_dof):
                motion_value = joint_values[i]
                self.joints[i].update_globals(pos=pos, rotmat=rotmat, motion_value=motion_value)
                pos = self.joints[i].gl_pos_q
                rotmat = self.joints[i].gl_rotmat_q
        tcp_gl_pos, tcp_gl_rotmat = self.cvt_tcp_loc_to_gl()
        if toggle_jacobian:
            j_mat = np.zeros((6, self.n_dof))
            for i in range(self.tcp_joint_id + 1):
                if self.joints[i].type == rkc.JointType.REVOLUTE:
                    vec_jnt2tcp = tcp_gl_pos - self.joints[i].gl_pos_q
                    j_mat[:3, i] = np.cross(self.joints[i].gl_motion_axis, vec_jnt2tcp)
                    j_mat[3:6, i] = self.joints[i].gl_motion_axis
                    if toggle_debug:
                        gm.gen_arrow(spos=self.joints[i].gl_pos_q,
                                     epos=self.joints[i].gl_pos_q + .3 * self.joints[i].gl_motion_axis,
                                     rgba=bc.black).attach_to(base)
                if self.joints[i].type == rkc.JointType.PRISMATIC:
                    j_mat[:3, i] = self.joints[i].gl_motion_axis
            return tcp_gl_rotmat, tcp_gl_rotmat, j_mat
        else:
            return tcp_gl_rotmat, tcp_gl_rotmat

    def jacobian(self, joint_values=None):
        """
        compute the jacobian matrix; use internal values if joint_values is None
        :param joint_values:
        :param update:
        :return:
        author :weiwei
        date: 20230829
        """
        if joint_values is None:  # use internal, ignore update
            _, _, j_mat = self.forward_kinematics(joint_values=self.get_joint_values(),
                                                  toggle_jacobian=True,
                                                  update=False)
        else:
            _, _, j_mat = self.forward_kinematics(joint_values=joint_values,
                                                  toggle_jacobian=True,
                                                  update=False)
        return j_mat

    def manipulability_vol(self, joint_values=None):
        """
        compute the yoshikawa manipulability
        :param tcp_joint_id:
        :param tcp_loc_pos:
        :param tcp_loc_rotmat:
        :return:
        author: weiwei
        date: 20200331
        """
        j_mat = self.jacobian(joint_values=joint_values)
        return np.sqrt(np.linalg.det(j_mat @ j_mat.T))

    def manipulability_mat(self, joint_values=None):
        """
        compute the axes of the manipulability ellipsoid
        :param tcp_joint_id:
        :param tcp_loc_pos:
        :param tcp_loc_rotmat:
        :return: (linear ellipsoid matrix, angular ellipsoid matrix)
        """
        j_mat = self.jacobian(joint_values=joint_values)
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

    def fix_to(self, pos, rotmat):
        self.anchor.pos = pos
        self.anchor.rotmat = rotmat
        return self.go_given_conf(joint_values=self.get_joint_values())

    def reinitialize(self):
        """
        :return:
        author: weiwei
        date: 20201126
        """
        self._joint_ranges = self._get_joint_ranges()
        self.go_home()
        self._tracik_solver = rkt.TracIKSolver(self)

    def set_tcp(self, tcp_joint_id=None, tcp_loc_pos=None, tcp_loc_rotmat=None):
        if tcp_joint_id is not None:
            self.tcp_joint_id = tcp_joint_id
        if tcp_loc_pos is not None:
            self.tcp_loc_pos = tcp_loc_pos
        if tcp_loc_rotmat is not None:
            self.tcp_loc_rotmat = tcp_loc_rotmat

    def get_gl_tcp(self):
        tcp_gl_pos, tcp_gl_rotmat = self.cvt_tcp_loc_to_gl()
        return tcp_gl_pos, tcp_gl_rotmat

    def cvt_tcp_loc_to_gl(self):
        gl_pos = self.joints[self.tcp_joint_id].gl_pos_q + self.joints[self.tcp_joint_id].gl_rotmat_q @ self.tcp_loc_pos
        gl_rotmat = self.joints[self.tcp_joint_id].gl_rotmat_q @ self.tcp_loc_rotmat
        return (gl_pos, gl_rotmat)

    def cvt_posrot_in_tcp_to_gl(self,
                                pos_in_loc_tcp=np.zeros(3),
                                rotmat_in_loc_tcp=np.eye(3)):
        """
        given a loc pos and rotmat in loc_tcp, convert it to global frame
        if the last three parameters are given, the code will use them as loc_tcp instead of the internal member vars.
        :param pos_in_loc_tcp: nparray 1x3
        :param rotmat_in_loc_tcp: nparray 3x3
        :param
        :return:
        author: weiwei
        date: 20190312, 20210609
        """
        tcp_gl_pos, tcp_gl_rotmat = self.cvt_tcp_loc_to_gl()
        cvted_gl_pos = tcp_gl_pos + tcp_gl_rotmat.dot(pos_in_loc_tcp)
        cvted_gl_rotmat = tcp_gl_rotmat.dot(rotmat_in_loc_tcp)
        return (cvted_gl_pos, cvted_gl_rotmat)

    def cvt_gl_posrot_to_tcp(self, gl_pos, gl_rotmat):
        """
        given a world pos and world rotmat
        get the relative pos and relative rotmat with respective to the ith jntlnk
        :param gl_pos: 1x3 nparray
        :param gl_rotmat: 3x3 nparray
        :return:
        author: weiwei
        date: 20190312
        """
        tcp_gl_pos, tcp_gl_rotmat = self.cvt_tcp_loc_to_gl()
        return rm.rel_pose(tcp_gl_pos, tcp_gl_rotmat, gl_pos, gl_rotmat)

    def are_joint_values_in_ranges(self, joint_values):
        """
        check if the given joint_values
        :param joint_values:
        :return:
        author: weiwei
        date: 20220326toyonaka
        """
        if len(joint_values) == self.n_dof:
            raise ValueError('The given joint values do not match n_dof')
        joint_values = np.asarray(joint_values)
        if np.any(joint_values < self.joint_ranges[:, 0]) or np.any(joint_values > self.joint_ranges[:, 1]):
            return False
        else:
            return True

    def go_given_conf(self, joint_values):
        """
        move the robot_s to the given pose
        :return: null
        author: weiwei
        date: 20230927osaka
        """
        return self.forward_kinematics(joint_values=joint_values, toggle_jacobian=False, update=True)

    def go_home(self):
        """
        move the robot_s to initial pose
        :return: null
        author: weiwei
        date: 20161211osaka
        """
        return self.go_given_conf(joint_values=self.home)

    def go_zero(self):
        """
        move the robot_s to initial pose
        :return: null
        author: weiwei
        date: 20161211osaka
        """
        return self.go_given_conf(joint_values=np.zeros(self.n_dof))

    def get_joint_values(self):
        """
        get the current joint values
        :return: joint_values: a 1xn ndarray
        author: weiwei
        date: 20161205tsukuba
        """
        jnt_values = np.zeros(self.n_dof)
        for i in range(self.n_dof):
            jnt_values[i] = self.joints[i].motion_value
        return jnt_values

    def rand_conf(self):
        """
        generate a random configuration
        author: weiwei
        date: 20200326
        """
        return np.multiply(np.random.rand(self.n_dof),
                           (self.joint_ranges[:, 1] - self.joint_ranges[:, 0])) + self.joint_ranges[:, 0]

    def ik(self,
           tgt_pos: np.ndarray,
           tgt_rotmat: np.ndarray,
           seed_joint_values=None,
           max_n_iter=100):
        """
        trac ik solver runs num_ik and opt_ik in parallel, and return the faster result
        :param tgt_pos: 1x3 nparray, single value or list
        :param tgt_rotmat: 3x3 nparray, single value or list
        :param seed_joint_values: the starting configuration used in the numerical iteration
        :param max_n_iter
        :return:
        """
        tic = time.time()
        jnt_values = self._tracik_solver.ik(tgt_pos=tgt_pos,
                                            tgt_rotmat=tgt_rotmat,
                                            seed_jnt_vals=seed_joint_values,
                                            max_n_iter=max_n_iter)
        toc = time.time()
        print("trac ik time ", toc - tic)
        return jnt_values

    def copy(self):
        return copy.deepcopy(self)


if __name__ == "__main__":
    import time
    import visualization.panda.world as wd
    import robot_sim.kinematics.model_generator as rkmg
    import robot_sim.kinematics.constant as rkc
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[1.25, .75, .75], lookat_pos=[0, 0, .3])
    gm.gen_frame().attach_to(base)

    jlc = JLChain(n_dof=6)
    jlc.joints[0].loc_pos = np.array([0, 0, 0])
    jlc.joints[0].loc_motion_axis = np.array([0, 0, 1])
    jlc.joints[0].motion_range = np.array([-np.pi / 2, np.pi / 2])
    # jlc.joints[1].change_type(rkc.JointType.PRISMATIC)
    jlc.joints[1].loc_pos = np.array([0, 0, .05])
    jlc.joints[1].loc_motion_axis = np.array([0, 1, 0])
    jlc.joints[1].motion_range = np.array([-np.pi / 2, np.pi / 2])
    jlc.joints[2].loc_pos = np.array([0, 0, .2])
    jlc.joints[2].loc_motion_axis = np.array([0, 1, 0])
    jlc.joints[2].motion_range = np.array([-np.pi, np.pi])
    jlc.joints[3].loc_pos = np.array([0, 0, .2])
    jlc.joints[3].loc_motion_axis = np.array([0, 0, 1])
    jlc.joints[3].motion_range = np.array([-np.pi / 2, np.pi / 2])
    jlc.joints[4].loc_pos = np.array([0, 0, .1])
    jlc.joints[4].loc_motion_axis = np.array([0, 1, 0])
    jlc.joints[4].motion_range = np.array([-np.pi / 2, np.pi / 2])
    jlc.joints[5].loc_pos = np.array([0, 0, .05])
    jlc.joints[5].loc_motion_axis = np.array([0, 0, 1])
    jlc.joints[5].motion_range = np.array([-np.pi / 2, np.pi / 2])
    jlc.tcp_loc_pos = np.array([0, 0, .01])
    jlc.reinitialize()
    # joint_values = np.array(
    #     [np.radians(90), np.radians(-30), np.radians(120), np.radians(30), np.radians(30), np.radians(30)])
    # tcp_pos_physical, tcp_rotmat_physical = jlc.go_given_conf(joint_values=joint_values)
    # rkmg.gen_jlc_stick(jlc, stick_rgba=basis.constant.navy_blue, toggle_tcp_frame=True,
    #                    toggle_joint_frame=True).attach_to(base)
    # print(jlc.forward_kinematics(joint_values=None, toggle_jacobian=True, update=False)[2])
    # print(jlc.forward_kinematics(joint_values=jlc.get_joint_values(), toggle_jacobian=True, update=False)[2])
    #
    # base.run()
    # linear_ellipsoid_mat, _ = jlc.manipulability_mat()
    # print(linear_ellipsoid_mat)
    # gm.gen_ellipsoid(pos=tcp_pos_physical, axes_mat=linear_ellipsoid_mat).attach_to(base)
    # base.run()

    # tgt_pos0 = np.array([.2, 0, 0])
    # tgt_rotmat0 = rm.rotmat_from_euler(np.radians(90), np.radians(90), 0)
    # tgt_pos1 = np.array([.2, .2, .2])
    # tgt_rotmat1 = np.eye(3)
    # gm.gen_myc_frame(pos=tgt_pos0, rotmat=tgt_rotmat0).attach_to(base)
    # # gm.gen_myc_frame(pos=tgt_pos1, rotmat=tgt_rotmat1).attach_to(base)
    # joint_values = jlc.ik(tgt_pos=tgt_pos0, tgt_rotmat=tgt_rotmat0, toggle_debug=False)
    # print(joint_values)
    seed_joint_values = jlc.get_joint_values()
    # tgt_pos0 = np.array([.3, .2, .1])
    # tgt_rotmat0 = rm.rotmat_from_euler(np.radians(-90), np.radians(0), np.radians(0))
    # gm.gen_frame(pos=tgt_pos0, rotmat=tgt_rotmat0).attach_to(base)
    # joint_values = jlc.ik(tgt_pos=tgt_pos0,
    #                       tgt_rotmat=tgt_rotmat0,
    #                       seed_joint_values=seed_joint_values)
    # print("initial ik is done!")
    # print(joint_values)
    # time.sleep(.1)

    for i in range(7):
        tgt_pos0 = np.array([.1 + .02 * i, -.1, .3])
        tgt_rotmat0 = rm.rotmat_from_euler(np.radians(90), np.radians(90), np.radians(0))
        gm.gen_frame(pos=tgt_pos0, rotmat=tgt_rotmat0).attach_to(base)
        joint_values = jlc.ik(tgt_pos=tgt_pos0,
                              tgt_rotmat=tgt_rotmat0,
                              seed_joint_values=seed_joint_values)
        print(f"{i}th ik is done! {joint_values}")
        time.sleep(.1)
        if joint_values is not None:
            jlc.forward_kinematics(joint_values=joint_values, update=True)
            seed_joint_values = joint_values
            rkmg.gen_jlc_stick(jlc).attach_to(base)
    base.run()

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
                               seed_joint_values=None,
                               tcp_joint_id=tcp_jnt_id_list,
                               tcp_loc_pos=tcp_loc_poslist,
                               tcp_loc_rotmat=tcp_loc_rotmatlist,
                               local_minima="accept",
                               toggle_debug=True)
    toc = time.time()
    print('ik cost: ', toc - tic, jnt_values)
    jlinstance.fk(joint_values=jnt_values)
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
