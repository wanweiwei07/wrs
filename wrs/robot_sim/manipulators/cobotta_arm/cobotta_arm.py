import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim.manipulators.manipulator_interface as mi


class CobottaArm(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), ik_solver='d', name="cobotta_arm", enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=np.zeros(6), name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "base_link.dae"), name=self.name + "_base")
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # first joint and link
        self.jlc.jnts[0].loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-2.617994, 2.617994])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(initor=os.path.join(current_file_dir, "meshes", "j1.dae"),
                                                         name=self.name + "_link1")
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # second joint and link
        self.jlc.jnts[1].loc_pos = np.array([0, 0, 0.18])
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[1].motion_range = np.array([-1.047198, 1.745329])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(initor=os.path.join(current_file_dir, "meshes", "j2.dae"),
                                                         name=self.name + "_link2")
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # third joint and link
        self.jlc.jnts[2].loc_pos = np.array([0, 0, 0.165])
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[2].motion_range = np.array([0.3141593, 2.443461])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(initor=os.path.join(current_file_dir, "meshes", "j3.dae"),
                                                         name=self.name + "_link3")
        self.jlc.jnts[2].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = np.array([-0.012, 0.02, 0.088])
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[3].motion_range = np.array([-2.96706, 2.96706])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(initor=os.path.join(current_file_dir, "meshes", "j4.dae"),
                                                         name=self.name + "_link4")
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = np.array([0, -.02, .0895])
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[4].motion_range = np.array([-1.658063, 2.356194])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(initor=os.path.join(current_file_dir, "meshes", "j5.dae"),
                                                         name=self.name + "_link5")
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = np.array([0, -.0445, 0.042])
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[5].motion_range = np.array([-2.96706, 2.96706])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(initor=os.path.join(current_file_dir, "meshes", "j6.dae"),
                                                         name=self.name + "_link6")
        self.jlc.jnts[5].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        self.jlc.finalize(ik_solver=ik_solver, identifier_str=name)
        # tcp
        self.loc_tcp_pos = np.array([0, 0, 0])
        self.loc_tcp_rotmat = np.eye(3)
        # set up cc
        if self.cc is not None:
            self.setup_cc()

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


if __name__ == '__main__':
    import time
    from wrs import wd
    import wrs.robot_sim._kinematics.ikgeo.sp4_lib as sp4_lib
    import wrs.robot_sim._kinematics.ikgeo.sp1_lib as sp1_lib

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, .3])
    mcm.mgm.gen_frame().attach_to(base)
    arm = CobottaArm(enable_cc=True)

    tgt_pos = np.array([.25, .1, .1])
    tgt_rotmat = rm.rotmat_from_euler(0, np.pi, 0)
    mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

    candidate_jnt_values = []
    last_error = 100
    for q4 in np.linspace(arm.jlc.jnts[3].motion_range[0], arm.jlc.jnts[3].motion_range[1], 200):
        R34 = rm.rotmat_from_axangle(arm.jlc.jnts[3].loc_motion_ax, q4)
        p06 = tgt_pos
        h2 = arm.jlc.jnts[1].loc_motion_ax
        p45 = arm.jlc.jnts[4].loc_pos
        p23 = arm.jlc.jnts[2].loc_pos
        # subproblem 4 for q1
        h = h2
        p = p06
        d = h2.T.dot(p23) + h2.T.dot(R34).dot(p45)
        k = arm.jlc.jnts[0].loc_motion_ax
        q1_cadidates, is_ls = sp4_lib.sp4_run(p, k, h, d)
        if not is_ls:
            for q in q1_cadidates:
                if arm.jlc.jnts[0].motion_range[0] < q < arm.jlc.jnts[0].motion_range[1]:
                    q1 = -q
                    # subproblem 4 for q5
                    h6 = arm.jlc.jnts[5].loc_pos
                    R10 = rm.rotmat_from_axangle(arm.jlc.jnts[0].loc_motion_ax, -q1)
                    h = h2.T.dot(R34)
                    p = h6
                    d = h2.T.dot(R10).dot(tgt_rotmat).dot(h6)
                    q5_candidates, is_ls = sp4_lib.sp4_run(p, k, h, d)
                    if not is_ls:
                        for q in q5_candidates:
                            if arm.jlc.jnts[4].motion_range[0] < q < arm.jlc.jnts[4].motion_range[1]:
                                q5 = q
                                # subproblem 4 for q6
                                R45 = rm.rotmat_from_axangle(arm.jlc.jnts[4].loc_motion_ax, q5)
                                h = h2.T.dot(R34).dot(R45)
                                p = h2
                                d = h2.T.dot(R10).dot(tgt_rotmat).dot(h2)
                                q6_candidates, is_ls = sp4_lib.sp4_run(p, k, h, d)
                                if not is_ls:
                                    for q in q6_candidates:
                                        if arm.jlc.jnts[5].motion_range[0] < q < arm.jlc.jnts[5].motion_range[1]:
                                            q6 = q
                                            # 1d search
                                            R65 = rm.rotmat_from_axangle(arm.jlc.jnts[5].loc_motion_ax, -q6)
                                            e_q4 = h2.T.dot(R34.T).dot(R45.T).dot(R65).dot(R10).dot(tgt_rotmat).dot(h2) - 1
                                            old_last_error = last_error
                                            last_error = e_q4
                                            print(old_last_error, e_q4)
                                            if old_last_error >= 100:
                                                continue
                                            else:
                                                if np.sign(e_q4) != np.sign(old_last_error) or abs(e_q4)<1e-3:
                                                    # subproblem 1 for q2
                                                    R65 = rm.rotmat_from_axangle(arm.jlc.jnts[5].loc_motion_ax, -q6)
                                                    p1 = p23
                                                    p2 = R10.dot(tgt_pos) - R10.dot(tgt_rotmat).dot(R65).dot(R45.T).dot(p45)
                                                    k = arm.jlc.jnts[1].loc_motion_ax
                                                    q2, is_ls = sp1_lib.sp1_run(p1, p2, k)
                                                    # if not is_ls:
                                                    R12 = rm.rotmat_from_axangle(arm.jlc.jnts[1].loc_motion_ax, q2)
                                                    # subproblem 1 for q3
                                                    p1 = R34.dot(p45)
                                                    p2 = R10.dot(tgt_pos) - R12.dot(p23)
                                                    k = arm.jlc.jnts[2].loc_motion_ax
                                                    q3, is_ls = sp1_lib.sp1_run(p1, p2, k)
                                                    candidate_jnt_values.append([q1, q2, q3, q4, q5, q6])
                                                    # if not is_ls:
                                                    #     break
    print(len(candidate_jnt_values))
    for jnt_values in candidate_jnt_values:
        arm.goto_given_conf(jnt_values=np.array(jnt_values))
        arm.gen_meshmodel(alpha=.3).attach_to(base)
    base.run()
    jnt_values = np.array([q1, q2, q3, q4, q5, q6])
    arm.goto_given_conf(jnt_values=jnt_values)
    arm.gen_meshmodel(alpha=.3).attach_to(base)
    # base.run()

    # arm.jlc._ik_solver.test_success_rate()
    arm_mesh = arm.gen_meshmodel(alpha=.3)
    arm_mesh.attach_to(base)
    tmp_arm_stick = arm.gen_stickmodel(toggle_flange_frame=True)
    tmp_arm_stick.attach_to(base)
    # base.run()

    tgt_pos = np.array([.25, .1, .1])
    tgt_rotmat = rm.rotmat_from_euler(0, np.pi, 0)
    mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    tic = time.time()
    jnt_values = arm.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    print(jnt_values)
    toc = time.time()
    print(toc - tic)
    if jnt_values is not None:
        arm.goto_given_conf(jnt_values=jnt_values)
    arm_mesh = arm.gen_meshmodel(alpha=.3)
    arm_mesh.attach_to(base)
    tmp_arm_stick = arm.gen_stickmodel(toggle_flange_frame=True)
    tmp_arm_stick.attach_to(base)
    base.run()

    arm.goto_given_conf(jnt_values=np.array([0, np.pi / 2, np.pi * 3 / 4, 0, np.pi / 2, 0]))
    arm.show_cdprim()

    arm_mesh = arm.gen_meshmodel(alpha=.3)
    arm_mesh.attach_to(base)
    tmp_arm_stick = arm.gen_stickmodel(toggle_flange_frame=True)
    tmp_arm_stick.attach_to(base)

    box = mcm.gen_box(xyz_lengths=np.array([0.1, .1, .1]), pos=tgt_pos)
    box.attach_to(base)
    tic = time.time()
    result, contacts = arm.is_collided(obstacle_list=[box], toggle_contacts=True)
    toc = time.time()
    print(toc - tic)
    for pnt in contacts:
        mcm.mgm.gen_sphere(pnt).attach_to(base)
    base.run()
