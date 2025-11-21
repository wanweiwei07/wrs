import os
import math
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface as mi
import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm


def wrap_to_range(value, jnt_range):
    low, high = jnt_range
    while value < low:  value += 2 * math.pi
    while value > high: value -= 2 * math.pi
    return value


class XArmLite6(mi.ManipulatorInterface):
    """
    Definition for XArm Lite 6
    Author: Chen Hao (chen960216@gmail.com), Updated by Weiwei
    Date: 20220909osaka, 20240318
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='xarm_lite6', enable_cc=False):
        home_conf = np.array([0., 0.173311, 0.555015, 0., 0.381703, 0.])
        super().__init__(pos=pos, rotmat=rotmat, home_conf=home_conf, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "base.stl"), name="xarm_lite6_base",
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED, ex_radius=.005, userdef_cdprim_fn=self._base_cdprim)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.const.tab20_list[15]
        # first joint and link
        self.jlc.jnts[0].loc_pos = np.array([.0, .0, .2433])
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-math.pi, math.pi])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link1.stl"), name="xarm_lite6_link1")
        self.jlc.jnts[0].lnk.cmodel.rgba = rm.const.tab20_list[15]
        # second joint and link
        self.jlc.jnts[1].loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(1.5708, -1.5708, 3.1416)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[1].motion_range = np.array([-2.61799, 2.61799])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link2.stl"), name="xarm_lite6_link2",
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED, ex_radius=.005, userdef_cdprim_fn=self._link2_cdprim)
        self.jlc.jnts[1].lnk.cmodel.rgba = rm.const.tab20_list[15]
        # third joint and link
        self.jlc.jnts[2].loc_pos = np.array([.2, .0, .0])
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(-3.1416, 0., 1.5708)
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[2].motion_range = np.array([-0.061087, 5.235988])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link3.stl"), name="xarm_lite6_link3")
        self.jlc.jnts[2].lnk.cmodel.rgba = rm.const.tab20_list[15]
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = np.array([.087, -.2276, .0])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(1.5708, 0., 0.)
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[3].motion_range = np.array([-math.pi, math.pi])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link4.stl"), name="xarm_lite6_link4",
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED, ex_radius=.005, userdef_cdprim_fn=self._link4_cdprim)
        self.jlc.jnts[3].lnk.cmodel.rgba = rm.const.tab20_list[15]
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(1.5708, 0., 0.)
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[4].motion_range = np.array([-2.1642, 2.1642])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link5.stl"), name="xarm_lite6_link5")
        self.jlc.jnts[4].lnk.cmodel.rgba = rm.const.tab20_list[15]
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = np.array([.0, .0615, .0])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(-1.5708, 0., 0.)
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[5].motion_range = np.array([-math.pi, math.pi])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link6.stl"), name="xarm_lite6_link6")
        self.jlc.jnts[5].lnk.cmodel.rgba = rm.const.tab20_list[15]
        self.jlc.finalize(ik_solver='a', identifier_str=name)
        # tcp
        self.loc_tcp_pos = np.array([0, 0, 0])
        self.loc_tcp_rotmat = np.eye(3)
        # set up cc
        if self.cc is not None:
            self.setup_cc()

    # self-defined collison model for the base link
    @staticmethod
    def _base_cdprim(name="auto", ex_radius=None):
        pdcnd = CollisionNode(name + "_cnode")
        collision_primitive_c0 = CollisionBox(Point3(-0.008, 0, 0.0375),
                                              x=.07 + ex_radius, y=.065 + ex_radius, z=0.0375 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0, 0, .124),
                                              x=.043 + ex_radius, y=.043 + ex_radius, z=.049 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        cdprim = NodePath(name + "_cdprim")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    @staticmethod
    def _link2_cdprim(name="auto", ex_radius=None):
        pdcnd = CollisionNode(name + "_cnode")
        collision_primitive_c0 = CollisionBox(Point3(0, 0, 0.1065),
                                              x=.041 + ex_radius, y=.042 + ex_radius, z=0.0315 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.100, 0, 0.1065),
                                              x=.059 + ex_radius, y=.042 + ex_radius, z=0.0315 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(.2, 0, 0.0915),
                                              x=.041 + ex_radius, y=.042 + ex_radius, z=0.0465 + ex_radius)
        pdcnd.addSolid(collision_primitive_c2)
        cdprim = NodePath(name + "_cdprim")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    @staticmethod
    def _link4_cdprim(name="auto", ex_radius=None):
        pdcnd = CollisionNode(name + "_cnode")
        collision_primitive_c0 = CollisionBox(Point3(0, 0, -0.124009),
                                              x=.041 + ex_radius, y=.042 + ex_radius, z=0.0682075 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0, -0.063315, -0.0503),
                                              x=.041 + ex_radius, y=.021315 + ex_radius, z=.087825 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        cdprim = NodePath(name + "_cdprim")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    def setup_cc(self):
        lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)
        from_list = [l3, l4, l5]
        into_list = [lb, l0, l1]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def ik(self,
           tgt_pos: np.ndarray,
           tgt_rotmat: np.ndarray,
           seed_jnt_values=None,
           option="single",
           toggle_dbg=False):
        """
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :return:
        """
        solutions = []
        tcp_loc_pos = self.loc_tcp_pos
        tcp_loc_rotmat = self.loc_tcp_rotmat
        tgt_flange_rotmat = tgt_rotmat @ tcp_loc_rotmat.T
        tgt_flange_pos = tgt_pos - tgt_flange_rotmat @ tcp_loc_pos
        rrr_pos = tgt_flange_pos - tgt_flange_rotmat[:, 2] * np.linalg.norm(self.jlc.jnts[5].loc_pos)
        rrr_x, rrr_y, rrr_z = ((rrr_pos - self.pos) @ self.rotmat).tolist()  # in base coordinate system

        j0_value_candidates = [math.atan2(rrr_y, rrr_x)]
        for j0_value in j0_value_candidates:
            if not self._is_jnt_in_range(jnt_id=0, jnt_value=j0_value):
                continue
            # assume a, b, c are the axis_length of shoulders and bottom of the big triangle formed by the robot arm
            c = math.sqrt(rrr_x ** 2 + rrr_y ** 2 + (rrr_z - self.jlc.jnts[0].loc_pos[2]) ** 2)
            a = self.jlc.jnts[2].loc_pos[0]
            b = np.linalg.norm(self.jlc.jnts[3].loc_pos)
            tmp_acos_target = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
            if tmp_acos_target > 1 or tmp_acos_target < -1:
                print("Analytical IK Failure: The triangle formed by the robot arm is violated!")
                continue
            j2_value_candidates = []
            j2_value = math.acos(tmp_acos_target)
            j2_initial_offset = math.atan(abs(self.jlc.jnts[3].loc_pos[0] / self.jlc.jnts[3].loc_pos[1]))
            j2_value_candidates.append(j2_value - j2_initial_offset)
            j2_value_candidates.append(wrap_to_range(-(j2_value + j2_initial_offset), self.jnt_ranges[2]))
            for id, j2_value in enumerate(j2_value_candidates):
                if not self._is_jnt_in_range(jnt_id=2, jnt_value=j2_value):
                    # ignore reversed elbow
                    # j2_value = math.acos(tmp_acos_target) - math.pi
                    # if not self._is_jnt_in_range(jnt_id=2, jnt_value=j2_value):
                    continue
                tmp_acos_target = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
                if tmp_acos_target > 1 or tmp_acos_target < -1:
                    print("Analytical IK Failure: The triangle formed by the robot arm is violated!")
                    continue
                j1_value_upper = math.acos(tmp_acos_target)
                # assume d, c, e are the edges of the lower triangle formed with the ground
                d = self.jlc.jnts[0].loc_pos[2]
                e = math.sqrt(rrr_x ** 2 + rrr_y ** 2 + rrr_z ** 2)
                tmp_acos_target = (d ** 2 + c ** 2 - e ** 2) / (2 * d * c)
                if tmp_acos_target > 1 or tmp_acos_target < -1:
                    print("Analytical IK Failure: The triangle formed with the ground is violated!")
                    continue
                j1_value_lower = math.acos(tmp_acos_target)
                if id == 0:
                    j1_value = math.pi - (j1_value_lower + j1_value_upper)
                elif id == 1:
                    j1_value = j1_value_upper + math.pi - j1_value_lower
                if not self._is_jnt_in_range(jnt_id=1, jnt_value=j1_value):
                    continue
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
                j3_value, j4_value, j5_value = rm.rotmat_to_euler(rrr_g_rotmat.T @ tgt_flange_rotmat,
                                                                  order='rzyz').tolist()
                j4_value = -j4_value
                solutions.append(np.array([j0_value, j1_value, j2_value, j3_value, j4_value, j5_value]))
                if self._is_jnt_in_range(jnt_id=4, jnt_value=j4_value) and \
                        self._is_jnt_in_range(jnt_id=4, jnt_value=-j4_value):
                    j4_value = -j4_value
                    j3_value -= np.pi
                    j5_value -= np.pi
                    solutions.append(np.array([j0_value, j1_value, j2_value, j3_value, j4_value, j5_value]))
        if option == "single":
            if len(solutions) == 0:
                return None
            else:
                return solutions[0]
        return solutions


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    from tqdm import tqdm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot = XArmLite6(enable_cc=True)

    success = 0
    num_trials = 10000
    pos_err_list = []
    rot_err_list = []
    visualize = False

    for i in tqdm(range(num_trials)):
        jnt = robot.rand_conf()
        # jnt = [ 1.85502727, -2.26425354,  3.87582289,  0.87367316,  1.27778191,
        # 2.33416951]
        pos, rot = robot.fk(jnt)
        print("Random jnt:", repr(jnt))
        robot.goto_given_conf(jnt)
        if visualize:
            mgm.gen_frame(pos=pos, rotmat=rot).attach_to(base)
            robot.gen_meshmodel(rgb=[1, 0, 0], alpha=0.5).attach_to(base)
        tgt_pos, tgt_rotmat = robot.fk(jnt)
        jnt_values = robot.ik(tgt_pos, tgt_rotmat, option="multiple")

        if jnt_values is not None:
            success += 1
            print("IK jnt values:", repr(jnt_values))

            if len(jnt_values) > 1:
                for jnt_value in jnt_values:
                    pos, rot = robot.fk(jnt_value)
                    pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos * 1000, tgt_rotmat, pos * 1000, rot)
                    pos_err_list.append(pos_err)
                    rot_err_list.append(rot_err)
                    if visualize:
                        mgm.gen_frame(pos=pos, rotmat=rot).attach_to(base)
                        robot.goto_given_conf(jnt_value)
                        robot.gen_meshmodel(rgb=[0, 0, 1], alpha=0.5).attach_to(base)
                if visualize: base.run()
    print(f"position error (mm): mean {np.mean(pos_err_list)}, std {np.std(pos_err_list)}")
    print(f"rotation error (deg): mean {np.mean(rot_err_list)}, std {np.std(rot_err_list)}")
    print(f"success rate: {success} / {num_trials} * 100% = {success / num_trials * 100}%")