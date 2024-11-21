from wrs import wd, mcm, rm
import time
import wrs.robot_sim.manipulators.cobotta_arm.cobotta_arm as cbta
import wrs.robot_sim._kinematics.ikgeo.sp4_lib as sp4_lib
import wrs.robot_sim._kinematics.ikgeo.sp3_lib as sp3_lib
import wrs.robot_sim._kinematics.ikgeo.sp1_lib as sp1_lib
from scipy.optimize import minimize

base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)
arm = cbta.CVR038(enable_cc=True)

def finalize(q4, p06, R06):
    R34 = rm.rotmat_from_axangle(arm.jlc.jnts[3].loc_motion_ax, q4)
    h2 = arm.jlc.jnts[1].loc_motion_ax
    p45 = arm.jlc.jnts[4].loc_pos + rm.np.array([0, arm.jlc.jnts[5].loc_pos[1], 0])
    p34 = arm.jlc.jnts[3].loc_pos
    p23 = arm.jlc.jnts[2].loc_pos
    R34p45 = R34 @ p45
    # subproblem 4 for q1
    h = h2
    p = p06
    d = h2.T @ (p23 + p34 + R34p45)
    k = -arm.jlc.jnts[0].loc_motion_ax
    q1_cadidates, is_ls = sp4_lib.sp4_run(p, k, h, d)
    # print("q1 ", q1_cadidates, arm.jlc.jnts[0].motion_range, is_ls)
    if not is_ls:
        for q in q1_cadidates:
            if arm.jlc.jnts[0].motion_range[0] < q < arm.jlc.jnts[0].motion_range[1]:
                q1 = q
                # subproblem 3 for q3
                p1 = p34 + R34p45
                p2 = -p23
                d = rm.np.linalg.norm(p06)
                k = arm.jlc.jnts[2].loc_motion_ax
                q3_candidates, is_ls = sp3_lib.sp3_run(p1, p2, k, d)
                # print("q3 ", q3_candidates, arm.jlc.jnts[2].motion_range, is_ls)
                if not is_ls:
                    if not isinstance(q3_candidates, rm.np.ndarray):
                        q3_candidates = [q3_candidates]
                    for q in q3_candidates:
                        if arm.jlc.jnts[2].motion_range[0] < q < arm.jlc.jnts[2].motion_range[1]:
                            q3 = q
                            # subproblem 1 for q2
                            R01 = rm.rotmat_from_axangle(arm.jlc.jnts[0].loc_motion_ax, q1)
                            R23 = rm.rotmat_from_axangle(arm.jlc.jnts[2].loc_motion_ax, q3)
                            R10p06 = R01.T @ p06
                            p1 = R10p06
                            p2 = p23 + R23 @ p34 + R23 @ R34p45
                            k = -arm.jlc.jnts[1].loc_motion_ax
                            q2, is_ls = sp1_lib.sp1_run(p1, p2, k)
                            # print("q2 ", q2, arm.jlc.jnts[1].motion_range, is_ls)
                            if not is_ls:
                                if arm.jlc.jnts[1].motion_range[0] < q2 < arm.jlc.jnts[1].motion_range[1]:
                                    R12 = rm.rotmat_from_axangle(arm.jlc.jnts[1].loc_motion_ax, q2)
                                    h5 = arm.jlc.jnts[4].loc_motion_ax
                                    h6 = arm.jlc.jnts[5].loc_motion_ax
                                    p1 = h6
                                    p2 = R34.T @ R23.T @ R12.T @ R01.T @ R06 @ h6
                                    k = arm.jlc.jnts[4].loc_motion_ax
                                    q5, is_ls = sp1_lib.sp1_run(p1, p2, k)
                                    # print("q5 ", q5, arm.jlc.jnts[4].motion_range, is_ls)
                                    if arm.jlc.jnts[4].motion_range[0] < q5 < arm.jlc.jnts[4].motion_range[1]:
                                        R45 = rm.rotmat_from_axangle(arm.jlc.jnts[4].loc_motion_ax, q5)
                                        # subproblem 1 for q6
                                        p1 = h5
                                        p2 = R06.T @ R01 @ R12 @ R23 @ R34 @ h5
                                        k = -arm.jlc.jnts[5].loc_motion_ax
                                        q6, is_ls = sp1_lib.sp1_run(p1, p2, k)
                                        if arm.jlc.jnts[5].motion_range[0] < q6 < arm.jlc.jnts[5].motion_range[1]:
                                            return [q1, q2, q3, q4, q5, q6]


def objective(q4, p06, R06):
    R34 = rm.rotmat_from_axangle(arm.jlc.jnts[3].loc_motion_ax, q4)
    h2 = arm.jlc.jnts[1].loc_motion_ax
    p45 = arm.jlc.jnts[4].loc_pos + rm.np.array([0, arm.jlc.jnts[5].loc_pos[1], 0])
    p34 = arm.jlc.jnts[3].loc_pos
    p23 = arm.jlc.jnts[2].loc_pos
    R34p45 = R34 @ p45
    # subproblem 4 for q1
    h = h2
    p = p06
    d = h2.T @ (p23 + p34 + R34p45)
    k = -arm.jlc.jnts[0].loc_motion_ax
    q1_cadidates, is_ls = sp4_lib.sp4_run(p, k, h, d)
    if not is_ls:
        for q in q1_cadidates:
            if arm.jlc.jnts[0].motion_range[0] < q < arm.jlc.jnts[0].motion_range[1]:
                q1 = q
                # subproblem 3 for q3
                p1 = p34 + R34p45
                p2 = -p23
                d = rm.np.linalg.norm(p06)
                k = arm.jlc.jnts[2].loc_motion_ax
                q3_candidates, is_ls = sp3_lib.sp3_run(p1, p2, k, d)
                # print("q3 ", q3_candidates, arm.jlc.jnts[2].motion_range, is_ls)
                if not is_ls:
                    if not isinstance(q3_candidates, rm.np.ndarray):
                        q3_candidates = [q3_candidates]
                    for q in q3_candidates:
                        if arm.jlc.jnts[2].motion_range[0] < q < arm.jlc.jnts[2].motion_range[1]:
                            q3 = q
                            # subproblem 1 for q2
                            R01 = rm.rotmat_from_axangle(arm.jlc.jnts[0].loc_motion_ax, q1)
                            R23 = rm.rotmat_from_axangle(arm.jlc.jnts[2].loc_motion_ax, q3)
                            R10p06 = R01.T @ p06
                            p1 = R10p06
                            p2 = p23 + R23 @ p34 + R23 @ R34p45
                            k = -arm.jlc.jnts[1].loc_motion_ax
                            q2, is_ls = sp1_lib.sp1_run(p1, p2, k)
                            # print("q2 ", q2, arm.jlc.jnts[1].motion_range, is_ls)
                            if not is_ls:
                                if arm.jlc.jnts[1].motion_range[0] < q2 < arm.jlc.jnts[1].motion_range[1]:
                                    R12 = rm.rotmat_from_axangle(arm.jlc.jnts[1].loc_motion_ax, q2)
                                    h5 = arm.jlc.jnts[4].loc_motion_ax
                                    h6 = arm.jlc.jnts[5].loc_motion_ax
                                    e_q4 = h5.T @ (R01 @ R12 @ R23 @ R34).T @ R06 @ h6 - h5.T @ h6
                                    return abs(e_q4)


tgt_pos = rm.vec(.1, -.3, .3)
tgt_rotmat = rm.rotmat_from_euler(0, rm.pi / 3, 0)
_p12 = arm.jlc.jnts[1].loc_pos
R06 = tgt_rotmat
p06 = tgt_pos - _p12 - R06 @ rm.np.array([0, 0, arm.jlc.jnts[5].loc_pos[2]])

initial_guess = (arm.jlc.jnts[3].motion_range[0] + arm.jlc.jnts[3].motion_range[1]) / 2
tic = time.time()
result = minimize(objective, initial_guess, args=(p06, R06), method='BFGS')
jnt_values = finalize(result.x, p06, R06)
toc = time.time()
print(toc - tic)

print(jnt_values)
mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
arm.goto_given_conf(jnt_values=jnt_values)
arm.gen_meshmodel(alpha=.3).attach_to(base)
base.run()