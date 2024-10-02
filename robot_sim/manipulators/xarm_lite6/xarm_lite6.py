import os
import math
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import basis.robot_math as rm
import modeling.collision_model as mcm
import robot_sim.manipulators.manipulator_interface as mi


class XArmLite6(mi.ManipulatorInterface):
    """
    Definition for XArm Lite 6
    Author: Chen Hao (chen960216@gmail.com), Updated by Weiwei
    Date: 20220909osaka, 20240318
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),
                 home_conf=np.array([0., 0.173311, 0.555015, 0., 0.381703, 0.]),
                 name='xarm_lite6', enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=home_conf, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "base.stl"),
                                                                cdprim_type=mcm.mc.CDPType.USER_DEFINED, ex_radius=.005,
                                                                userdef_cdprim_fn=self._base_cdprim)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.bc.tab20_list[15]
        # first joint and link
        self.jlc.jnts[0].loc_pos = np.array([.0, .0, .2433])
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-2 * math.pi, 2 * math.pi])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link1.stl"))
        self.jlc.jnts[0].lnk.cmodel.rgba = rm.bc.tab20_list[15]
        # second joint and link
        self.jlc.jnts[1].loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(1.5708, -1.5708, 3.1416)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[1].motion_range = np.array([-2.61799, 2.61799])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link2.stl"),
                                                         cdprim_type=mcm.mc.CDPType.USER_DEFINED, ex_radius=.005,
                                                         userdef_cdprim_fn=self._link2_cdprim)
        self.jlc.jnts[1].lnk.cmodel.rgba = rm.bc.tab20_list[15]
        # third joint and link
        self.jlc.jnts[2].loc_pos = np.array([.2, .0, .0])
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(-3.1416, 0., 1.5708)
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[2].motion_range = np.array([-0.061087, 5.235988])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link3.stl"))
        self.jlc.jnts[2].lnk.cmodel.rgba = rm.bc.tab20_list[15]
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = np.array([.087, -.2276, .0])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(1.5708, 0., 0.)
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[3].motion_range = np.array([-2 * math.pi, 2 * math.pi])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link4.stl"),
                                                         cdprim_type=mcm.mc.CDPType.USER_DEFINED, ex_radius=.005,
                                                         userdef_cdprim_fn=self._link4_cdprim)
        self.jlc.jnts[3].lnk.cmodel.rgba = rm.bc.tab20_list[15]
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(1.5708, 0., 0.)
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[4].motion_range = np.array([-2.1642, 2.1642])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link5.stl"))
        self.jlc.jnts[4].lnk.cmodel.rgba = rm.bc.tab20_list[15]
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = np.array([.0, .0615, .0])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(-1.5708, 0., 0.)
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[5].motion_range = np.array([-2 * math.pi, 2 * math.pi])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link6.stl"))
        self.jlc.jnts[5].lnk.cmodel.rgba = rm.bc.tab20_list[15]
        self.jlc.finalize(ik_solver='a', identifier_str=name)
        self.jlc.finalize()
        # tcp
        self.loc_tcp_pos = np.array([0, 0, 0])
        self.loc_tcp_rotmat = np.eye(3)
        # set up cc
        if self.cc is not None:
            self.setup_cc()

    # self-defined collison model for the base link
    @staticmethod
    def _base_cdprim(ex_radius):
        pdcnd = CollisionNode("base")
        collision_primitive_c0 = CollisionBox(Point3(-0.008, 0, 0.0375),
                                              x=.07 + ex_radius, y=.065 + ex_radius, z=0.0375 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0, 0, .124),
                                              x=.043 + ex_radius, y=.043 + ex_radius, z=.049 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        cdprim = NodePath("user_defined_base")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    @staticmethod
    def _link2_cdprim(ex_radius):
        pdcnd = CollisionNode("link2")
        collision_primitive_c0 = CollisionBox(Point3(0, 0, 0.1065),
                                              x=.041 + ex_radius, y=.042 + ex_radius, z=0.0315 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.100, 0, 0.1065),
                                              x=.059 + ex_radius, y=.042 + ex_radius, z=0.0315 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(.2, 0, 0.0915),
                                              x=.041 + ex_radius, y=.042 + ex_radius, z=0.0465 + ex_radius)
        pdcnd.addSolid(collision_primitive_c2)
        cdprim = NodePath("user_defined_link2")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    @staticmethod
    def _link4_cdprim(ex_radius):
        pdcnd = CollisionNode("link4")
        collision_primitive_c0 = CollisionBox(Point3(0, 0, -0.124009),
                                              x=.041 + ex_radius, y=.042 + ex_radius, z=0.0682075 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0, -0.063315, -0.0503),
                                              x=.041 + ex_radius, y=.021315 + ex_radius, z=.087825 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        cdprim = NodePath("user_defined_link4")
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
        Modified D-H from base to tcp
        T01 = sympy.Matrix([[c1, -s1, 0, 0],
                            [s1, c1, 0, 0],
                            [0, 0, 1, d1],
                            [0, 0, 0, 1]])
        T12 = sympy.Matrix([[s2, c2, 0, 0],
                            [0, 0, 1, 0],
                            [c2, -s2, 0, 0],
                            [0, 0, 0, 1]])
        T23 = sympy.Matrix([[s3, c3, 0, a3],
                            [c3, -s3, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])
        T34 = sympy.Matrix([[c4, -s4, 0, a4],
                            [0, 0, -1, -d4],
                            [s4, c4, 0, 0],
                            [0, 0, 0, 1]])
        T45 = sympy.Matrix([[c5, -s5, 0, 0],
                            [0, 0, -1, 0],
                            [s5, c5, 0, 0],
                            [0, 0, 0, 1]])
        T56 = sympy.Matrix([[c6, -s6, 0, 0],
                            [0, 0, 1, 0],
                            [-s6, -c6, 0, 0],
                            [0, 0, 0, 1]])
        T6t = sympy.Matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, d_hnd + d6],
                            [0, 0, 0, 1]])
        T0t=T01*T12*T23*T34*T45*T56*T6t
        """
        # relative to base
        rel_pos, rel_rotmat = rm.rel_pose(self.jlc.pos, self.jlc.rotmat, tgt_pos, tgt_rotmat)
        # target
        tgt_rotmat = rel_rotmat @ self.loc_tcp_rotmat.T
        tgt_pos = rel_pos - rel_rotmat @ self.loc_tcp_pos
        # lengths of link
        d1, d4, d6 = .2433, .2276, .0615
        a3, a4 = .2, .087
        # global position of joint 1
        pos_1 = np.array([0, 0, d1])
        # Joint 1
        pos_w = tgt_pos - d6 * tgt_rotmat[:, 2]
        theta1 = np.arctan2(pos_w[1], pos_w[0])
        # Joint 3
        d1w = np.sum((pos_w - pos_1) ** 2)
        num3_1 = 2 * a3 * a4
        num3_2 = abs(- a3 ** 4 + 2 * a3 ** 2 * a4 ** 2 + 2 * a3 ** 2 * d1w + 2 * a3 ** 2 * d4 ** 2 - a4 ** 4 +
                     2 * a4 ** 2 * d1w - 2 * a4 ** 2 * d4 ** 2 - d1w ** 2 + 2 * d1w * d4 ** 2 - d4 ** 4)
        den3 = a3 ** 2 + 2 * a3 * d4 + a4 ** 2 - d1w + d4 ** 2
        theta3_list = [-2 * np.arctan2((num3_1 - np.sqrt(num3_2)), den3),
                       -2 * np.arctan2((num3_1 + np.sqrt(num3_2)), den3)]
        theta2_4_5_6_list = []
        for theta3 in theta3_list:
            # Joint 2
            dxy = np.sqrt(pos_w[0] ** 2 + pos_w[1] ** 2)
            z_w = pos_w[2]
            num_c2 = -a3 * d1 + a3 * z_w - a4 * d1 * np.sin(theta3) + a4 * dxy * np.cos(theta3) + a4 * z_w * np.sin(
                theta3) + \
                     d1 * d4 * np.cos(theta3) + d4 * dxy * np.sin(theta3) - d4 * z_w * np.cos(theta3)
            num_s2 = a3 * dxy + a4 * d1 * np.cos(theta3) + a4 * dxy * np.sin(theta3) - a4 * z_w * np.cos(theta3) + \
                     d1 * d4 * np.sin(theta3) - d4 * dxy * np.cos(theta3) - d4 * z_w * np.sin(theta3)
            den_cs2 = a3 ** 2 + 2 * a3 * a4 * np.sin(theta3) - 2 * a3 * d4 * np.cos(theta3) + a4 ** 2 * np.sin(
                theta3) ** 2 + \
                      a4 ** 2 * np.cos(theta3) ** 2 + d4 ** 2 * np.sin(theta3) ** 2 + d4 ** 2 * np.cos(theta3) ** 2
            if den_cs2 < 0:
                num_s2 = -num_s2
                num_c2 = -num_c2
            theta2 = np.arctan2(num_s2, num_c2)
            # Joint 4,5,6
            U06 = rm.homomat_from_posrot(pos=pos_w, rotmat=tgt_rotmat)
            s1, c1, s2, c2, s3, c3 = np.sin(theta1), np.cos(theta1), np.sin(theta2), np.cos(theta2), np.sin(
                theta3), np.cos(theta3)
            T03 = np.array([[s2 * s3 * c1 + c1 * c2 * c3, s2 * c1 * c3 - s3 * c1 * c2, s1, a3 * s2 * c1],
                            [s1 * s2 * s3 + s1 * c2 * c3, s1 * s2 * c3 - s1 * s3 * c2, -c1, a3 * s1 * s2],
                            [-s2 * c3 + s3 * c2, s2 * s3 + c2 * c3, 0, a3 * c2 + d1],
                            [0, 0, 0, 1]])
            """
            U36 = 
            [[-sin(θ4)*sin(θ6) + cos(θ4)*cos(θ5)*cos(θ6), -sin(θ4)*cos(θ6) - sin(θ6)*cos(θ4)*cos(θ5), -sin(θ5)*cos(θ4), a4], 
             [-sin(θ5)*cos(θ6), sin(θ5)*sin(θ6), -cos(θ5), -d4], 
             [sin(θ4)*cos(θ5)*cos(θ6) + sin(θ6)*cos(θ4), -sin(θ4)*sin(θ6)*cos(θ5) + cos(θ4)*cos(θ6), -sin(θ4)*sin(θ5), 0], 
             [0, 0, 0, 1]]
            """
            U36 = (np.linalg.inv(T03)).dot(U06)
            theta6 = -np.arctan2(U36[1, 1], U36[1, 0])
            theta4 = np.arctan2(U36[2, 2], U36[0, 2])
            c5 = -U36[1, 2]
            if abs(U36[1, 0] / np.cos(theta6)) < abs(U36[1, 1] / np.sin(theta6)):
                s5 = -U36[1, 0] / np.cos(theta6)
            else:
                s5 = U36[1, 1] / np.sin(theta6)
            theta5 = np.arctan2(s5, c5)
            theta2_4_5_6_list.append((theta2, theta4, theta5, theta6))
        # Adapt to joint range
        if theta3_list[0] < .0:
            theta3_list[0] = 2. * np.pi + theta3_list[0]
        if theta3_list[1] < .0:
            theta3_list[1] = 2. * np.pi + theta3_list[1]
        candidate_jnt_values_list = [np.array([theta1, j2, j3, j4, j5, j6]) for j3, (j2, j4, j5, j6) in
                                     zip(theta3_list, theta2_4_5_6_list)]
        jnt_values_list = []
        for jnt_values in candidate_jnt_values_list:
            if ((self.jlc.jnt_ranges[:, 0] <= jnt_values) & (jnt_values <= self.jlc.jnt_ranges[:, 1])).all():
                jnt_values_list.append(jnt_values)
        # if not ((self.jlc.jnts[1].motion_range[0] <= jnt_values_list[1][1] <= self.jlc.jnts[1].motion_range[1]) and
        #         (self.jlc.jnts[2].motion_range[0] <= jnt_values_list[1][2] <= self.jlc.jnts[2].motion_range[1]) and
        #         (self.jlc.jnts[4].motion_range[0] <= jnt_values_list[1][4] <= self.jlc.jnts[4].motion_range[1])):
        #     jnt_values_list.pop(1)
        # if not ((self.jlc.jnts[1].motion_range[0] <= jnt_values_list[0][1] <= self.jlc.jnts[1].motion_range[1]) and
        #         (self.jlc.jnts[2].motion_range[0] <= jnt_values_list[0][2] <= self.jlc.jnts[2].motion_range[1]) and
        #         (self.jlc.jnts[4].motion_range[0] <= jnt_values_list[0][4] <= self.jlc.jnts[4].motion_range[1])):
        #     jnt_values_list.pop(0)
        if len(jnt_values_list) == 0:
            return None
        if option == "single" or option is None:
            if len(jnt_values_list) == 1:
                return jnt_values_list[0]
            else:
                # return joint values close to seed_jnt_values
                seed_jnt_values = np.zeros(6) if seed_jnt_values is None else seed_jnt_values
                if np.linalg.norm(jnt_values_list[0] - seed_jnt_values) < np.linalg.norm(
                        jnt_values_list[1] - seed_jnt_values):
                    return jnt_values_list[0]
                else:
                    return jnt_values_list[1]
        if option == "multiple":
            return jnt_values_list


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as mgm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot = XArmLite6(enable_cc=True)
    robot.gen_meshmodel(toggle_cdprim=True).attach_to(base)

    tgt_pos = np.array([.3, .3, .3])
    tgt_rotmat = rm.rotmat_from_euler(0, 0, 0)
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    jnt_values = robot.ik(tgt_pos, tgt_rotmat, option="single")
    robot.goto_given_conf(jnt_values=jnt_values)
    robot.gen_meshmodel().attach_to(base)
    base.run()
