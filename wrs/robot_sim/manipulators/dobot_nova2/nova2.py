import os
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim.manipulators.manipulator_interface as mi


class Nova2(mi.ManipulatorInterface):
    """
    Definition for Dobot Nova 2
    author: chen hao <chen960216@gmail.com>, 20230214osaka; weiwei20240611
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), ik_solver='d', name='nova2', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=np.zeros(6), name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "base_link0.stl"),
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
            userdef_cdprim_fn=self._base_cdprim)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # first joint and link
        self.jlc.jnts[0].loc_pos = np.array([.0, .0, .2234])
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, -1])
        self.jlc.jnts[0].motion_range = np.array([-np.pi, np.pi])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j1.stl"))
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # second joint and link
        self.jlc.jnts[1].loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(1.5708, 1.5708, 0)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[1].motion_range = np.array([-np.pi, np.pi])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j2.stl"))
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # third joint and link
        self.jlc.jnts[2].loc_pos = np.array([-.28, .0, .0])
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[2].motion_range = np.array([-2.79, 2.79])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j3.stl"))
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = np.array([-0.22501, .0, 0.1175])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(0, 0, -1.5708)
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[3].motion_range = np.array([-2 * np.pi, 2 * np.pi])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j4.stl"))
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = np.array([.0, -0.12, .0])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(1.5708, .0, .0)
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[4].motion_range = np.array([-2 * np.pi, 2 * np.pi])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j5.stl"))
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = np.array([0., 0.088004, 0])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(-1.5708, -np.pi, 0)
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[5].motion_range = np.array([-2 * np.pi, 2 * np.pi])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j6.stl"))
        self.jlc.finalize(ik_solver=ik_solver, identifier_str=name)
        # tcp
        self.loc_tcp_pos = np.array([0, 0, 0])
        self.loc_tcp_rotmat = np.eye(3)
        # set up cc
        if self.cc is not None:
            self.setup_cc()

    # self-defined collison model for the base link
    @staticmethod
    def _base_cdprim(ex_radius):
        pdcnd = CollisionNode("nova_base")
        collision_primitive_c0 = CollisionBox(Point3(-0.008, 0, 0.0375),
                                              x=.07 + ex_radius, y=.065 + ex_radius, z=0.0375 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0, 0, .124),
                                              x=.043 + ex_radius, y=.043 + ex_radius, z=.049 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        cdprim = NodePath("user_defined")
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
        into_list = [lb, l0]
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
        :param option: "single", "multiple", None -> "single"
        :param toggle_dbg:
        :return:
        author: weiwei, liang qin
        date: 20240708
        """
        # default parameters
        option = "single" if option is None else option
        # relative to base
        rel_pos, rel_rotmat = rm.rel_pose(self.jlc.pos, self.jlc.rotmat, tgt_pos, tgt_rotmat)
        # target
        tgt_rotmat = rel_rotmat @ self.loc_tcp_rotmat.T
        tgt_pos = rel_pos - rel_rotmat @ self.loc_tcp_pos
        # DH parameters of nova2
        a2 = -0.280
        a3 = -0.22501
        d1 = 0.2234
        d4 = 0.1175
        d5 = 0.120
        d6 = 0.088004
        n = tgt_rotmat[:, 0]
        o = tgt_rotmat[:, 1]
        a = tgt_rotmat[:, 2]
        p = tgt_pos
        q = np.zeros((8, 6))
        m1 = d6 * a[1] - p[1]
        n1 = d6 * a[0] - p[0]
        k = m1 ** 2 + n1 ** 2 - d4 ** 2
        if -1e-8 < k < 0:
            k = 0
        for index in range(4):
            q[index][0] = np.arctan2(m1, n1) - np.arctan2(d4, np.sqrt(k))
            q[index + 4][0] = np.arctan2(m1, n1) - np.arctan2(d4, -np.sqrt(k))
        for index in range(4):
            q5 = np.arccos(a[0] * np.sin(q[2 * index + 1][0]) - a[1] * np.cos(q[2 * index + 1][0]))
            if index % 2 == 0:
                q[2 * index][4] = q5
                q[2 * index + 1][4] = q5
            else:
                q[2 * index][4] = -q5
                q[2 * index + 1][4] = -q5
        for index in range(8):
            m6 = n[0] * np.sin(q[index][0]) - n[1] * np.cos(q[index][0])
            n6 = o[0] * np.sin(q[index][0]) - o[1] * np.cos(q[index][0])
            q[index][5] = np.arctan2(m6, n6) - np.arctan2(np.sin(q[index][4]), 0)
            m3 = d5 * (np.sin(q[index][5]) * (n[0] * np.cos(q[index][0]) + n[1] * np.sin(q[index][0]))
                       + np.cos(q[index][5]) * (o[0] * np.cos(q[index][0]) + o[1] * np.sin(q[index][0]))) \
                 + p[0] * np.cos(q[index][0]) + p[1] * np.sin(q[index][0]) - d6 * (
                         a[0] * np.cos(q[index][0]) + a[1] * np.sin(q[index][0]))
            n3 = p[2] - d1 - a[2] * d6 + d5 * (o[2] * np.cos(q[index][5]) + n[2] * np.sin(q[index][5]))
            k3 = (m3 ** 2 + n3 ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)
            if k3 - 1 > 1e-6 or k3 + 1 < -1e-6:
                q3 = np.nan
            elif 0 <= k3 - 1 <= 1e-6:
                q3 = 0
            elif 0 <= k3 + 1 < 1e-6:
                q3 = np.pi
            else:
                q3 = np.arccos(k3)
            q[index][2] = q3 if index % 2 == 0 else -q3
            s2 = ((a3 * np.cos(q[index][2]) + a2) * n3 - a3 * np.sin(q[index][2]) * m3) / \
                 (a2 ** 2 + a3 ** 2 + 2 * a2 * a3 * np.cos(q[index][2]))
            c2 = (m3 + a3 * np.sin(q[index][2]) * s2) / (a3 * np.cos(q[index][2]) + a2)
            q[index][1] = np.arctan2(s2, c2)
            s234 = -np.sin(q[index][5]) * (n[0] * np.cos(q[index][0]) + n[1] * np.sin(q[index][0])) - \
                   np.cos(q[index][5]) * (o[0] * np.cos(q[index][0]) + o[1] * np.sin(q[index][0]))
            c234 = o[2] * np.cos(q[index][5]) + n[2] * np.sin(q[index][5])
            q[index][3] = np.arctan2(s234, c234) - q[index][1] - q[index][2]
        # ur5 -> nova2
        q[:, 1] = q[:, 1] + np.ones(8) * np.pi / 2
        q[:, 3] = q[:, 3] + np.ones(8) * np.pi / 2
        q[:, 0] = -q[:, 0]
        for index_i in range(8):
            for index_j in range(6):
                if q[index_i][index_j] < self.jnt_ranges[index_j][0]:
                    q[index_i][index_j] += 2 * np.pi
                elif q[index_i][index_j] >= self.jnt_ranges[index_j][1]:
                    q[index_i][index_j] -= 2 * np.pi
        result = q[~np.isnan(q).any(axis=1)]
        if len(result) == 0:
            print("No valid solutions found")
            return None
        else:
            mask = np.all((result >= self.jnt_ranges[:, 0]) & (result <= self.jnt_ranges[:, 1]), axis=1)
            filtered_result = result[mask]
            if len(filtered_result) == 0:
                print("No valid solutions found")
                return None
            if seed_jnt_values is None:
                seed_jnt_values = self.home_conf
            if option == "single":
                return filtered_result[np.argmin(np.linalg.norm(filtered_result - seed_jnt_values, axis=1))]
            elif option == "multiple":
                return filtered_result[np.argsort(np.linalg.norm(filtered_result - seed_jnt_values, axis=1))]


if __name__ == '__main__':
    import time
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mcm.mgm.gen_frame().attach_to(base)
    arm = Nova2(ik_solver='d', enable_cc=True)
    # arm.jlc.test_ik_success_rate()
    # arm_meshmodel = arm.gen_meshmodel()
    # arm_meshmodel.attach_to(base)
    # base.run()

    random_conf = arm.rand_conf()
    tgt_pos, tgt_rotmat = arm.fk(random_conf)
    tic = time.time()
    jv_list = arm.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, option="multiple")
    toc = time.time()
    print(toc - tic)
    if jv_list is not None:
        for jnt_values in jv_list:
            arm.goto_given_conf(jnt_values=jnt_values)
            arm_mesh = arm.gen_meshmodel(alpha=.3)
            arm_mesh.attach_to(base)
    tmp_arm_stick = arm.gen_stickmodel(toggle_flange_frame=True)
    tmp_arm_stick.attach_to(base)
    base.run()
