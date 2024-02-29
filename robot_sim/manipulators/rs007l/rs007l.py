import os
import math
import numpy as np
import basis.robot_math as rm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.manipulator_interface as mi


class RS007L(mi.ManipulatorInterface):
    """
    author: weiwei
    date: 20230728
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(6), name='khi_rs007l', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, home_conf=homeconf, name=name)
        # six joints, n_jnts = 6+2 (tgt ranges from 1-6), nlinks = 6+1
        self.jlc.jnts[1]['pos_in_loc_tcp'] = np.array([0, 0, 0.36])
        self.jlc.jnts[1]['loc_motionax'] = np.array([0, 0, -1])
        self.jlc.jnts[1]['motion_range'] = [-3.14159265359, 3.14159265359]  # -180, 180
        self.jlc.jnts[2]['pos_in_loc_tcp'] = np.array([0, 0, 0.0])
        self.jlc.jnts[2]['gl_rotmat'] = rm.rotmat_from_euler(0, np.radians(-90), 0)
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[2]['motion_range'] = [-2.35619449019, 2.35619449019]  # -135, 135
        self.jlc.jnts[3]['pos_in_loc_tcp'] = np.array([0.455, 0, 0])
        self.jlc.jnts[3]['loc_motionax'] = np.array([0, 0, -1])
        self.jlc.jnts[3]['motion_range'] = [-2.74016692563, 2.74016692563]  # -157, 157
        self.jlc.jnts[4]['pos_in_loc_tcp'] = np.array([0.0925, 0, 0])
        self.jlc.jnts[4]['gl_rotmat'] = rm.rotmat_from_euler(0, np.radians(90), 0)
        self.jlc.jnts[4]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[4]['motion_range'] = [-3.49065850399, 3.49065850399]  # -200, 200
        self.jlc.jnts[5]['pos_in_loc_tcp'] = np.array([0, 0, 0.3825])
        self.jlc.jnts[5]['gl_rotmat'] = rm.rotmat_from_euler(0, np.radians(-90), 0)
        self.jlc.jnts[5]['loc_motionax'] = np.array([0, 0, -1])
        self.jlc.jnts[5]['motion_range'] = [-2.18166156499, 2.18166156499]  # -125, 125
        self.jlc.jnts[6]['pos_in_loc_tcp'] = np.array([0.078, 0, 0])
        self.jlc.jnts[6]['gl_rotmat'] = rm.rotmat_from_euler(0, np.radians(90), 0)
        self.jlc.jnts[6]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[6]['motion_range'] = [-6.28318530718, 6.28318530718]  # -360, 360
        # links
        self.jlc.lnks[0]['name'] = "base"
        self.jlc.lnks[0]['pos_in_loc_tcp'] = np.zeros(3)
        self.jlc.lnks[0]['mass'] = 11
        self.jlc.lnks[0]['com'] = np.array([0, 0, 0])
        self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "joint0.stl")
        self.jlc.lnks[0]['rgba'] = [.7, .7, .7, 1.0]
        self.jlc.lnks[1]['name'] = "l1"
        self.jlc.lnks[1]['pos_in_loc_tcp'] = np.zeros(3)
        self.jlc.lnks[1]['com'] = np.array([0, 0, 0])
        self.jlc.lnks[1]['mass'] = 8.188
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "joint1.stl")
        self.jlc.lnks[1]['rgba'] = [.7, .7, .7, 1.0]
        self.jlc.lnks[2]['name'] = "l2"
        self.jlc.lnks[2]['pos_in_loc_tcp'] = np.array([0, 0, 0])
        self.jlc.lnks[2]['gl_rotmat'] = rm.rotmat_from_euler(0, np.radians(90), 0)
        self.jlc.lnks[2]['com'] = np.array([0, 0, 0])
        self.jlc.lnks[2]['mass'] = 6.826
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "joint2.stl")
        self.jlc.lnks[2]['rgba'] = [.7, .7, .7, 1]
        self.jlc.lnks[3]['name'] = "l3"
        self.jlc.lnks[3]['pos_in_loc_tcp'] = np.array([0, 0, 0])
        self.jlc.lnks[3]['gl_rotmat'] = rm.rotmat_from_euler(0, np.radians(90), 0)
        self.jlc.lnks[3]['com'] = np.array([0, 0, 0])
        self.jlc.lnks[3]['mass'] = 5.236
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "joint3.stl")
        self.jlc.lnks[4]['name'] = "l4"
        self.jlc.lnks[4]['pos_in_loc_tcp'] = np.array([0, 0, 0.3825])
        self.jlc.lnks[4]['com'] = np.array([0, 0, 0])
        self.jlc.lnks[4]['mass'] = 5.066
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "joint4.stl")
        self.jlc.lnks[4]['rgba'] = [.7, .7, .7, 1.0]
        self.jlc.lnks[5]['name'] = "l5"
        self.jlc.lnks[5]['pos_in_loc_tcp'] = np.array([0, 0, 0])
        self.jlc.lnks[5]['gl_rotmat'] = rm.rotmat_from_euler(0, np.radians(90), 0)
        self.jlc.lnks[5]['com'] = np.array([0, 0, 0])
        self.jlc.lnks[5]['mass'] = 1.625
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "joint5.stl")
        self.jlc.lnks[5]['rgba'] = [.7, .7, .7, 1]
        self.jlc.lnks[6]['name'] = "l6"
        self.jlc.lnks[6]['pos_in_loc_tcp'] = np.array([0, 0, 0])
        self.jlc.lnks[6]['com'] = np.array([.0, .0, 0])
        self.jlc.lnks[6]['mass'] = 0.625
        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "joint6.stl")
        self.jlc.lnks[6]['rgba'] = [.7, .7, .7, 1.0]
        self.jlc.finalize()
        # collision detection
        if enable_cc:
            self.enable_cc()

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 6])
        activelist = [self.jlc.lnks[0],
                      self.jlc.lnks[1],
                      self.jlc.lnks[2],
                      self.jlc.lnks[3],
                      self.jlc.lnks[4],
                      self.jlc.lnks[5],
                      self.jlc.lnks[6]]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.jlc.lnks[0],
                    self.jlc.lnks[1]]
        intolist = [self.jlc.lnks[4],
                    self.jlc.lnks[5],
                    self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)

    def ik(self,
           tgt_pos: np.ndarray,
           tgt_rotmat: np.ndarray,
           tcp_loc_pos: np.ndarray = None,
           tcp_loc_rotmat: np.ndarray = None,
           **kwargs):
        """
        analytical ik sovler,
        the parameters in kwargs will be ignored
        tcp_joint_id is always jlc.tcp_joint_id (-1),
        :param tgt_pos:
        :param tgt_rotmat:
        :param tcp_loc_pos:
        :param tcp_loc_rotmat:
        :return:
        author: weiwei
        date: 20230728
        """
        if tcp_loc_pos is None:
            tcp_loc_pos = self.jlc.loc_tcp_pos
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.jlc.loc_tcp_rotmat
        flange_rotmat = tgt_rotmat @ tcp_loc_rotmat.T
        flange_pos = tgt_pos - flange_rotmat @ tcp_loc_pos
        rrr_pos = flange_pos - flange_rotmat[:, 2] * np.linalg.norm(self.jlc.jnts[6]['pos_in_loc_tcp'])
        rrr_x, rrr_y, rrr_z = ((rrr_pos - self.jlc.pos) @ self.rotmat).tolist()
        j1_value = math.atan2(rrr_x, rrr_y)
        if not self._is_jnt_in_range(1, jnt_value=j1_value):
            return None
        # assume a, b, c are the axis_length of shoulders and bottom of the big triangle formed by the robot arm
        c = math.sqrt(rrr_x ** 2 + rrr_y ** 2 + (rrr_z - self.jlc.jnts[1]['pos_in_loc_tcp'][2]) ** 2)
        a = self.jlc.jnts[3]['pos_in_loc_tcp'][0]
        b = self.jlc.jnts[4]['pos_in_loc_tcp'][0] + self.jlc.jnts[5]['pos_in_loc_tcp'][2]
        tmp_acos_target = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
        if tmp_acos_target > 1 or tmp_acos_target < -1:
            print("The triangle formed by the robot arm is violated!")
            return None
        j3_value = math.acos(tmp_acos_target) - math.pi
        if not self._is_jnt_in_range(3, jnt_value=j3_value):
            return None
        tmp_acos_target = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
        if tmp_acos_target > 1 or tmp_acos_target < -1:
            print("The triangle formed by the robot arm is violated!")
            return None
        j2_value_upper = math.acos(tmp_acos_target)
        # assume d, c, e are the edges of the lower triangle formed with the ground
        d = self.jlc.jnts[1]['pos_in_loc_tcp'][2]
        e = math.sqrt(rrr_x ** 2 + rrr_y ** 2 + rrr_z ** 2)
        tmp_acos_target = (d ** 2 + c ** 2 - e ** 2) / (2 * d * c)
        if tmp_acos_target > 1 or tmp_acos_target < -1:
            print("The triangle formed with the ground is violated!")
            return None
        j2_value_lower = math.acos(tmp_acos_target)
        j2_value = math.pi - (j2_value_lower + j2_value_upper)
        if not self._is_jnt_in_range(2, jnt_value=j2_value):
            return None
        # RRR
        j0_gl_rotmatq = self.jlc.rotmat
        j1_gl_rotmat0 = j0_gl_rotmatq @ self.jlc.jnts[1]['gl_rotmat']
        j1_gl_rotmatq = j1_gl_rotmat0 @ rm.rotmat_from_axangle(self.jlc.jnts[1]['loc_motionax'], j1_value)
        j2_gl_rotmat0 = j1_gl_rotmatq @ self.jlc.jnts[2]['gl_rotmat']
        j2_gl_rotmatq = j2_gl_rotmat0 @ rm.rotmat_from_axangle(self.jlc.jnts[2]['loc_motionax'], j2_value)
        j3_gl_rotmat0 = j2_gl_rotmatq @ self.jlc.jnts[3]['gl_rotmat']
        j3_gl_rotmatq = j3_gl_rotmat0 @ rm.rotmat_from_axangle(self.jlc.jnts[3]['loc_motionax'], j3_value)
        j4_gl_rotmatq = j3_gl_rotmatq @ self.jlc.jnts[4]['gl_rotmat']
        j5_gl_rotmatq = j4_gl_rotmatq @ self.jlc.jnts[5]['gl_rotmat']
        rrr_g_rotmat = j5_gl_rotmatq @ self.jlc.jnts[6]['gl_rotmat']
        j4_value, j5_value, j6_value = rm.rotmat_to_euler(rrr_g_rotmat.T @ flange_rotmat, "rzxz").tolist()
        if not (self._is_jnt_in_range(4, jnt_value=j4_value) and \
                self._is_jnt_in_range(5, jnt_value=j5_value) and \
                self._is_jnt_in_range(6, jnt_value=j6_value)):
            return None
        return np.array([j1_value, j2_value, j3_value, j4_value, j5_value, j6_value])

    def _is_jnt_in_range(self, jnt_id: int, jnt_value: float):
        """

        :param jnt_id:
        :param jnt_value:
        :return:
        author: weiwei
        date: 20230801
        """
        if jnt_value < self.jlc.jnts[jnt_id]['motion_range'][0] or jnt_value > self.jlc.jnts[jnt_id]['motion_range'][1]:
            print(f"Error: Joint {jnt_id} is out of range!")
            return False
        else:
            return True


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[5, 0, 3], lookat_pos=[0, 0, .7])
    gm.gen_frame(axis_length=.3, axis_radius=.01).attach_to(base)
    manipulator_instance = RS007L(pos=np.array([0, 0, 0.2]),
                                  rotmat=rm.rotmat_from_euler(np.radians(30), np.radians(-30), 0), enable_cc=True)
    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    # manipulator_meshmodel.show_cdprimit()
    # manipulator_instance.gen_stickmodel(toggle_joint_frame=True).attach_to(base)
    # tic = time.time()
    # print(manipulator_instance.is_collided())
    # toc = time.time()
    # print(toc - tic)

    # tgt_pos = np.array([.5, 0, .3])
    # tgt_rotmat = rm.rotmat_from_euler(np.radians(30), np.radians(120), np.radians(130))
    # mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    tic = time.time()
    # loc_tcp_pos = np.array([0, .1, 0.1])
    # loc_tcp_rotmat = rm.rotmat_from_euler(0, np.radians(30), 0)
    tgt_pos = np.array([.25, .2, .15])
    tgt_rotmat = rm.rotmat_from_euler(np.radians(130), np.radians(40), np.radians(180))
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    tcp_loc_pos = np.array([0, 0, .1645])
    tcp_loc_rotmat = np.eye(3)
    j_values = manipulator_instance.ik(tgt_pos=tgt_pos,
                                       tgt_rotmat=tgt_rotmat,
                                       tcp_loc_pos=tcp_loc_pos,
                                       tcp_loc_rotmat=tcp_loc_rotmat)
    toc = time.time()
    print("ik cost: ", toc - tic)
    manipulator_instance.fk(jnt_values=j_values)
    manipulator_instance.gen_stickmodel(toggle_tcpcs=True, toggle_jntscs=True).attach_to(base)
    manipulator_instance.gen_meshmodel(tcp_loc_pos=tcp_loc_pos, tcp_loc_rotmat=tcp_loc_rotmat).attach_to(base)

    # base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0,0,0])
    # mgm.GeometricModel("./meshes/base.dae").attach_to(base)
    # mgm.gen_frame().attach_to(base)
    base.run()
