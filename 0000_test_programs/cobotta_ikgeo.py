import time
from wrs import wd, mcm, rm
import wrs.robot_sim.manipulators.cobotta_arm.cobotta_arm as cbta
import wrs.robot_sim._kinematics.ikgeo.sp4_lib as sp4_lib
import wrs.robot_sim._kinematics.ikgeo.sp1_lib as sp1_lib

base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)
arm = cbta.CVR038(enable_cc=True)

tgt_pos = rm.vec(.25, .1, .1)
tgt_rotmat = rm.rotmat_from_euler(0, rm.pi, 0)
mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

candidate_jnt_values = []
last_error = 100
for q4 in rm.np.linspace(arm.jlc.jnts[3].motion_range[0], arm.jlc.jnts[3].motion_range[1], 100):
    p01 = arm.jlc.jnts[1].loc_pos
    R06 = tgt_rotmat
    p06 = tgt_pos - p01 - R06.dot(rm.np.array([0, 0, arm.jlc.jnts[5].loc_pos[2]]))
    R34 = rm.rotmat_from_axangle(arm.jlc.jnts[3].loc_motion_ax, q4)
    h2 = arm.jlc.jnts[1].loc_motion_ax
    p45 = arm.jlc.jnts[4].loc_pos + rm.np.array([0, arm.jlc.jnts[5].loc_pos[1], 0])
    p34 = arm.jlc.jnts[3].loc_pos
    p23 = arm.jlc.jnts[2].loc_pos
    # subproblem 4 for q1
    h = h2
    p = p06
    d = h2.T.dot(p23 + p34) + h2.T.dot(R34).dot(p45)
    k = -arm.jlc.jnts[0].loc_motion_ax
    q1_cadidates, is_ls = sp4_lib.sp4_run(p, k, h, d)
    if not is_ls:
        for q in q1_cadidates:
            if arm.jlc.jnts[0].motion_range[0] < q < arm.jlc.jnts[0].motion_range[1]:
                q1 = q
                # subproblem 4 for q5
                h6 = arm.jlc.jnts[5].loc_motion_ax
                R01 = rm.rotmat_from_axangle(arm.jlc.jnts[0].loc_motion_ax, q1)
                h = h2.T.dot(R34)
                p = h6
                d = h2.T.dot(R01.T).dot(R06).dot(p)
                k = arm.jlc.jnts[4].loc_motion_ax
                q5_candidates, is_ls = sp4_lib.sp4_run(p, k, h, d)
                if not is_ls:
                    for q in q5_candidates:
                        if arm.jlc.jnts[4].motion_range[0] < q < arm.jlc.jnts[4].motion_range[1]:
                            q5 = q
                            # subproblem 4 for q6
                            R45 = rm.rotmat_from_axangle(arm.jlc.jnts[4].loc_motion_ax, q5)
                            h = h2.T.dot(R34).dot(R45)
                            p = h2
                            d = h2.T.dot(R01.T).dot(R06).dot(p)
                            k = arm.jlc.jnts[5].loc_motion_ax
                            q6_candidates, is_ls = sp4_lib.sp4_run(p, k, h, d)
                            if not is_ls:
                                for q in q6_candidates:
                                    if arm.jlc.jnts[5].motion_range[0] < q < arm.jlc.jnts[5].motion_range[1]:
                                        q6 = q
                                        # # q2 and q
                                        # R56 = rm.rotmat_from_axangle(arm.jlc.jnts[5].loc_motion_ax, q6)
                                        # p1 = p23
                                        # R10066554=R01.T.dot(R06).dot(R56.T).dot(R45.T)
                                        # p2 = R01.T.dot(p06) - R10066554.dot(R34.T).dot(p34)-R10066554.dot(p45)
                                        # k = arm.jlc.jnts[1].loc_motion_ax
                                        # q2, is_ls = sp1_lib.sp1_run(p1, p2, k)
                                        # # print(q2, is_ls)
                                        # if not is_ls:
                                        #     R12 = rm.rotmat_from_axangle(arm.jlc.jnts[1].loc_motion_ax, q2)
                                        #     # sub-problem 1 for q3
                                        #     p1 = p34+R34.dot(p45)
                                        #     p2 = R12.T.dot(R01.T.dot(p06)) - p23
                                        #     k = arm.jlc.jnts[2].loc_motion_ax
                                        #     q3, is_ls = sp1_lib.sp1_run(p1, p2, k)
                                        #     candidate_jnt_values.append([q1, q2, q3, q4, q5, q6])
                                        # 1d search
                                        R56 = rm.rotmat_from_axangle(arm.jlc.jnts[5].loc_motion_ax, q6)
                                        e_q4 = h2.T.dot(R01.T).dot(R06).dot(R56.T).dot(R45.T).dot(R34.T).dot(h2) - 1
                                        # old_last_error = last_error
                                        # last_error = e_q4
                                        # print(old_last_error, e_q4)
                                        # if old_last_error == 100:
                                        #     continue
                                        # else:
                                        if abs(e_q4) < 1e-15:
                                            # sub-problem 1 for q2
                                            p1 = p23
                                            R10066554 = R01.T.dot(R06).dot(R56.T).dot(R45.T)
                                            p2 = R01.T.dot(p06) - R10066554.dot(R34.T).dot(p34) - R10066554.dot(p45)
                                            k = arm.jlc.jnts[1].loc_motion_ax
                                            q2, is_ls = sp1_lib.sp1_run(p1, p2, k)
                                            if arm.jlc.jnts[1].motion_range[0] < q2 < arm.jlc.jnts[1].motion_range[1]:
                                                # if not is_ls:
                                                R12 = rm.rotmat_from_axangle(arm.jlc.jnts[1].loc_motion_ax, q2)
                                                # sub-problem 1 for q3
                                                p1 = p34 + R34.dot(p45)
                                                p2 = R12.T.dot(R01).T.dot(p06)- p23
                                                k = arm.jlc.jnts[2].loc_motion_ax
                                                q3, is_ls = sp1_lib.sp1_run(p1, p2, k)
                                                if arm.jlc.jnts[2].motion_range[0] < q3 < arm.jlc.jnts[2].motion_range[1]:
                                                    candidate_jnt_values.append([q1, q2, q3, q4, q5, q6])
print(len(candidate_jnt_values))
for jnt_values in candidate_jnt_values:
    arm.goto_given_conf(jnt_values=rm.vec(*jnt_values))
    arm.gen_meshmodel(alpha=.3).attach_to(base)
base.run()
jnt_values = rm.vec(q1, q2, q3, q4, q5, q6)
arm.goto_given_conf(jnt_values=jnt_values)
arm.gen_meshmodel(alpha=.3).attach_to(base)
# base.run()

# arm.jlc._ik_solver.test_success_rate()
arm_mesh = arm.gen_meshmodel(alpha=.3)
arm_mesh.attach_to(base)
tmp_arm_stick = arm.gen_stickmodel(toggle_flange_frame=True)
tmp_arm_stick.attach_to(base)
# base.run()
