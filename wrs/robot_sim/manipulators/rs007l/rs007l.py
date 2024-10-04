import os
import math
import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim.manipulators.manipulator_interface as mi


class RS007L(mi.ManipulatorInterface):
    """
    author: weiwei
    date: 20230728
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='khi_rs007l', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, home_conf=np.zeros(6), enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "joint0.stl"))
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # first joint and link
        self.jlc.jnts[0].loc_pos = np.array([0.0, 0.0, 0.36])
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, -1])
        self.jlc.jnts[0].motion_range = np.array([-3.14159265359, 3.14159265359])  # -180, 180
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "joint1.stl"))
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # second joint and link
        self.jlc.jnts[1].loc_pos = np.array([0.0, 0.0, 0.0])
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(0, np.radians(-90), 0)
        self.jlc.jnts[1].motion_range = np.array([-2.35619449019, 2.35619449019])  # -135, 135
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "joint2.stl"))
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        self.jlc.jnts[1].lnk.loc_rotmat = rm.rotmat_from_euler(0, np.radians(90), 0)
        # third joint and link
        self.jlc.jnts[2].loc_pos = np.array([0.455, 0.0, 0.0])
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 0, -1])
        self.jlc.jnts[2].motion_range = np.array([-2.74016692563, 2.74016692563])  # -157, 157
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "joint3.stl"))
        self.jlc.jnts[2].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        self.jlc.jnts[2].lnk.loc_rotmat = rm.rotmat_from_euler(0, np.radians(90), 0)
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = np.array([0.0925, 0.0, 0.0])
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(0, np.radians(90), 0)
        self.jlc.jnts[3].motion_range = np.array([-3.49065850399, 3.49065850399])  # -200, 200
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "joint4.stl"))
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        self.jlc.jnts[3].lnk.loc_pos = np.array([0.0, 0.0, 0.3852])
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = np.array([0, 0, 0.3825])
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 0, -1])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(0, np.radians(-90), 0)
        self.jlc.jnts[4].motion_range = np.array([-2.18166156499, 2.18166156499])  # -125, 125
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "joint5.stl"))
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        self.jlc.jnts[4].lnk.loc_rotmat = rm.rotmat_from_euler(0, np.radians(90), 0)
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = np.array([0.078, 0, 0])
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(0, np.radians(90), 0)
        self.jlc.jnts[5].motion_range = np.array([-6.28318530718, 6.28318530718])  # -360, 360
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "joint6.stl"))
        self.jlc.jnts[5].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        self.jlc.finalize(ik_solver=None, identifier_str=name)
        # tcp
        self.loc_tcp_pos = np.array([0, 0, 0])
        self.loc_tcp_rotmat = np.eye(3)
        # set up cc
        if self.cc is not None:
            self.enable_cc()

    def enable_cc(self):
        lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)
        from_list = [l3, l4, l5]
        into_list = [lb, l0]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def ik(self,
           tgt_pos: np.ndarray,
           tgt_rotmat: np.ndarray,
           **kwargs):
        """
        analytical ik sovler, slover than ddik
        the parameters in kwargs will be ignored
        :param tgt_pos:
        :param tgt_rotmat:
        :return:
        author: weiwei
        date: 20230728
        """
        tcp_loc_pos = self.loc_tcp_pos
        tcp_loc_rotmat = self.loc_tcp_rotmat
        tgt_flange_rotmat = tgt_rotmat @ tcp_loc_rotmat.T
        tgt_flange_pos = tgt_pos - tgt_flange_rotmat @ tcp_loc_pos
        rrr_pos = tgt_flange_pos - tgt_flange_rotmat[:, 2] * np.linalg.norm(self.jlc.jnts[5].loc_pos)
        rrr_x, rrr_y, rrr_z = ((rrr_pos - self.pos) @ self.rotmat).tolist()  # in base coordinate system
        j0_value = math.atan2(rrr_x, rrr_y)
        if not self._is_jnt_in_range(jnt_id=0, jnt_value=j0_value):
            return None
        # assume a, b, c are the axis_length of shoulders and bottom of the big triangle formed by the robot arm
        c = math.sqrt(rrr_x ** 2 + rrr_y ** 2 + (rrr_z - self.jlc.jnts[0].loc_pos[2]) ** 2)
        a = self.jlc.jnts[2].loc_pos[0]
        b = self.jlc.jnts[3].loc_pos[0] + self.jlc.jnts[4].loc_pos[2]
        tmp_acos_target = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
        if tmp_acos_target > 1 or tmp_acos_target < -1:
            print("The triangle formed by the robot arm is violated!")
            return None
        j2_value = math.acos(tmp_acos_target) - math.pi
        if not self._is_jnt_in_range(jnt_id=2, jnt_value=j2_value):
            # ignore reversed elbow
            # j2_value = math.acos(tmp_acos_target)
            # if not self._is_jnt_in_range(jnt_id=2, jnt_value=j2_value):
            return None
        tmp_acos_target = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
        if tmp_acos_target > 1 or tmp_acos_target < -1:
            print("The triangle formed by the robot arm is violated!")
            return None
        j1_value_upper = math.acos(tmp_acos_target)
        # assume d, c, e are the edges of the lower triangle formed with the ground
        d = self.jlc.jnts[0].loc_pos[2]
        e = math.sqrt(rrr_x ** 2 + rrr_y ** 2 + rrr_z ** 2)
        tmp_acos_target = (d ** 2 + c ** 2 - e ** 2) / (2 * d * c)
        if tmp_acos_target > 1 or tmp_acos_target < -1:
            print("The triangle formed with the ground is violated!")
            return None
        j1_value_lower = math.acos(tmp_acos_target)
        j1_value = math.pi - (j1_value_lower + j1_value_upper)
        if not self._is_jnt_in_range(jnt_id=1, jnt_value=j1_value):
            return None
        # RRR
        anchor_gl_rotmatq = self.rotmat
        j0_gl_rotmat0 = anchor_gl_rotmatq @ self.jlc.jnts[0].loc_rotmat
        j0_gl_rotmatq = j0_gl_rotmat0 @ rm.rotmat_from_axangle(self.jlc.jnts[0].loc_motion_ax, j0_value)
        j1_gl_rotmat0 = j0_gl_rotmatq @ self.jlc.jnts[1].loc_rotmat
        j1_gl_rotmatq = j1_gl_rotmat0 @ rm.rotmat_from_axangle(self.jlc.jnts[1].loc_motion_ax, j1_value)
        j2_gl_rotmat0 = j1_gl_rotmatq @ self.jlc.jnts[2].loc_rotmat
        j2_gl_rotmatq = j2_gl_rotmat0 @ rm.rotmat_from_axangle(self.jlc.jnts[2].loc_motion_ax, j2_value)
        rrr_g_rotmat = (j2_gl_rotmatq @ self.jlc.jnts[3].loc_rotmat @
                        self.jlc.jnts[4].loc_rotmat @ self.jlc.jnts[5].loc_rotmat)
        j3_value, j4_value, j5_value = rm.rotmat_to_euler(rrr_g_rotmat.T @ tgt_flange_rotmat, order='rzxz').tolist()
        if not (self._is_jnt_in_range(jnt_id=3, jnt_value=j3_value) and
                self._is_jnt_in_range(jnt_id=4, jnt_value=j4_value) and
                self._is_jnt_in_range(jnt_id=5, jnt_value=j5_value)):
            return None
        return np.array([j0_value, j1_value, j2_value, j3_value, j4_value, j5_value])


if __name__ == '__main__':
    import time
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[5, 0, 3], lookat_pos=[0, 0, .7])
    mcm.mgm.gen_frame(ax_length=.3, ax_radius=.01).attach_to(base)
    arm = RS007L(pos=np.array([0, 0, 0.2]),
                 rotmat=rm.rotmat_from_euler(np.radians(30), np.radians(-30), 0), enable_cc=True)
    arm.gen_meshmodel(alpha=.3).attach_to(base)
    arm.gen_stickmodel(toggle_jnt_frames=True).attach_to(base)
    tgt_pos = np.array([.35, .3, 1])
    tgt_rotmat = rm.rotmat_from_euler(np.radians(130), np.radians(40), np.radians(180))
    mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    tic = time.time()
    j_values = arm.ik(tgt_pos=tgt_pos,
                      tgt_rotmat=tgt_rotmat)
    toc = time.time()
    print("ik cost: ", toc - tic)
    print(j_values)
    arm.goto_given_conf(jnt_values=j_values)
    arm.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    arm.gen_meshmodel().attach_to(base)
    base.run()
