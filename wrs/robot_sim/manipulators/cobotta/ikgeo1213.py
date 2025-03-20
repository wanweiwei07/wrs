import wrs.basis.robot_math as rm
import wrs.robot_sim._kinematics.ik_num as ikn
import wrs.robot_sim._kinematics.ikgeo.sp4_lib as sp4_lib
import wrs.robot_sim._kinematics.ikgeo.sp3_lib as sp3_lib
import wrs.robot_sim._kinematics.ikgeo.sp1_lib as sp1_lib


def err_given_q4(q4, jlc, p06, R06):
    R34 = rm.rotmat_from_axangle(jlc.jnts[3].loc_motion_ax, q4)
    h2 = jlc.jnts[1].loc_motion_ax
    p45 = rm.vec(0.15, -.59, 0)
    p23 = rm.vec(-0.05,0,0.71)
    R34p45 = R34 @ p45
    # subproblem 3 for q3
    p1 = R34p45
    p2 = -p23
    d = rm.np.linalg.norm(p06)
    k = jlc.jnts[2].loc_motion_ax
    q3_candidates, is_ls = sp3_lib.sp3_run(p1, p2, k, d)
    # print("q3 ", q3_candidates, jlc.jnts[2].motion_range, is_ls)
    if is_ls:
        return None, None
        # q3_candidates = rm.np.asarray([q3_candidates])
    # q3_candidates is always an np.array when is_ls is False
    # filter valid q3 solutions
    q3_min, q3_max = jlc.jnts[2].motion_range
    q3_valid = rm.np.array([q3 for q3 in q3_candidates if q3_min <= q3 <= q3_max])
    # subproblem 4 for q1
    h = h2
    p = p06
    d = h2.T @ (p23+R34p45)
    k = -jlc.jnts[0].loc_motion_ax
    q1_candidates, is_ls = sp4_lib.sp4_run(p, k, h, d)
    # print("q1 ", q1_candidates, jlc.jnts[0].motion_range, is_ls)
    if is_ls:
        return None, None
        # q1_candidates = rm.np.asarray([q1_candidates])
    # q1_candidates is always an np.array when is_ls is False
    # filter valid q1 solutions
    q1_min, q1_max = jlc.jnts[0].motion_range
    q1_valid = rm.np.array([q1 for q1 in q1_candidates if q1_min <= q1 <= q1_max])
    # Compute errors for all combinations of q1 and q3
    # Preallocate memory for errors
    max_combinations = len(q1_valid) * len(q3_valid)
    current_errs = rm.np.empty(max_combinations, dtype=rm.np.float64)
    q_values = rm.np.empty((max_combinations, 3), dtype=rm.np.float64)
    error_count = 0
    for q1 in q1_valid:
        for q3 in q3_valid:
            # Subproblem 1: Solve for q2
            R01 = rm.rotmat_from_axangle(jlc.jnts[0].loc_motion_ax, q1)
            R23 = rm.rotmat_from_axangle(jlc.jnts[2].loc_motion_ax, q3)
            R10p06 = R01.T @ p06
            p1 = R10p06
            p2 = p23 + R23 @ R34p45
            k = -jlc.jnts[1].loc_motion_ax
            q2, is_ls = sp1_lib.sp1_run(p1, p2, k)
            if not is_ls:
                q2_min, q2_max = jlc.jnts[1].motion_range
                if q2_min <= q2 <= q2_max:
                    R12 = rm.rotmat_from_axangle(jlc.jnts[1].loc_motion_ax, q2)
                    h5 = jlc.jnts[4].loc_motion_ax
                    h6 = jlc.jnts[5].loc_motion_ax
                    e_q4 = h5.T @ (R01 @ R12 @ R23 @ R34).T @ R06 @ h6 - h5.T @ h6
                    current_errs[error_count] = e_q4
                    q_values[error_count] = [q1, q2, q3]
                    error_count += 1
    if error_count > 0:
        return current_errs[:error_count], q_values[:error_count]
    else:
        return None, None

def search1d(jlc, start, end, n_div, p06, R06):
    q4_candidates = rm.np.linspace(start, end, n_div)
    previous_errs = None
    previous_q4 = None
    zero_crossings = []
    for q4 in q4_candidates:
        current_errs, current_q123s = err_given_q4(q4, jlc, p06, R06)
        if current_errs is None:
            previous_errs = None
            previous_q4 = None
            continue
        if previous_errs is not None:
            # Compare current errors with previous errors to detect zero crossings
            for prev_err, curr_err, curr_q123 in zip(previous_errs, current_errs, current_q123s):
                if prev_err * curr_err < 0:  # Sign change detected
                    # Interpolate to find the zero-crossing q4
                    q4_cross = previous_q4 + (-prev_err / (curr_err - prev_err)) * (q4 - previous_q4)
                    zero_crossings.append([curr_q123[0], curr_q123[1], curr_q123[2], q4_cross])
        previous_errs = current_errs
        previous_q4 = q4
    return zero_crossings


def solve_q56(jlc, R06, q1, q2, q3, q4):
    R01 = rm.rotmat_from_axangle(jlc.jnts[0].loc_motion_ax, q1)
    R12 = rm.rotmat_from_axangle(jlc.jnts[1].loc_motion_ax, q2)
    R23 = rm.rotmat_from_axangle(jlc.jnts[2].loc_motion_ax, q3)
    R34 = rm.rotmat_from_axangle(jlc.jnts[3].loc_motion_ax, q4)
    h5 = jlc.jnts[4].loc_motion_ax
    h6 = jlc.jnts[5].loc_motion_ax
    p1 = h6
    p2 = R34.T @ R23.T @ R12.T @ R01.T @ R06 @ h6
    k = jlc.jnts[4].loc_motion_ax
    q5, is_ls = sp1_lib.sp1_run(p1, p2, k)
    if jlc.jnts[4].motion_range[0] < q5 < jlc.jnts[4].motion_range[1]:
        p1 = h5
        p2 = R06.T @ R01 @ R12 @ R23 @ R34 @ h5
        k = -jlc.jnts[5].loc_motion_ax
        q6, is_ls = sp1_lib.sp1_run(p1, p2, k)
        if jlc.jnts[5].motion_range[0] < q6 < jlc.jnts[5].motion_range[1]:
            return q5, q6
    return None, None


def ik(jlc, tgt_pos, tgt_rotmat, n_div = 36, seed_jnt_values=None, option='single'):
    _backbone_solver = ikn.NumIKSolver(jlc)
    # if seed_jnt_values is not None:
    #     result = _backbone_solver(tgt_pos, tgt_rotmat, seed_jnt_values)
    #     return result
    _p01 = jlc.jnts[0].loc_pos
    # relative to base (ikgeo assumes jlc.pos = 0 and jlc.rotmat = I), thus we need to convert tgt_pos and tgt_rotmat
    rel_pos, rel_rotmat = rm.rel_pose(jlc.pos, jlc.rotmat, tgt_pos, tgt_rotmat)
    R06 = rel_rotmat @ jlc.loc_flange_rotmat.T
    p06 = rel_pos - R06 @ rm.np.array([0, jlc.jnts[5].loc_pos[1], 0]) - _p01
    zero_crossings = search1d(jlc, jlc.jnts[3].motion_range[0], jlc.jnts[3].motion_range[1], n_div, p06, R06)
    # print(zero_crossings)
    candidate_jnt_values = []
    for q1, q2, q3, q4 in zero_crossings:
        q5, q6 = solve_q56(jlc, R06, q1, q2, q3, q4)
        if q5 is not None:
            # backbone_solver uses jlc properties and takes into account jlc.pos and jlc.rotmat
            # there is not need to convert tgt_pos and tgt_rotmat
            result = _backbone_solver(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=[q1, q2, q3, q4, q5, q6],
                                      max_n_iter=7)
            if result is not None:
                candidate_jnt_values.append(result)
    if len(candidate_jnt_values) > 0:
        filtered_result = rm.np.array(candidate_jnt_values)
        if seed_jnt_values is None:
            seed_jnt_values = jlc.home
            if option == "single":
                return filtered_result[rm.np.argmin(rm.np.linalg.norm(filtered_result - seed_jnt_values, axis=1))]
            elif option == "multiple":
                return filtered_result[rm.np.argsort(rm.np.linalg.norm(filtered_result - seed_jnt_values, axis=1))]
    else:
        return None


if __name__ == '__main__':
    from tqdm import tqdm
    from wrs import wd, mcm, rm
    import wrs.robot_sim.manipulators.cobotta.cvrb1213 as cbtm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, .3])
    mcm.mgm.gen_frame().attach_to(base)
    arm = cbtm.CVRB1213()

    # jnt_vals = arm.rand_conf()
    # tgt_pos, tgt_rotmat = arm.goto_given_conf(jnt_values=jnt_vals)
    # arm.gen_meshmodel(rgb=rm.const.green, alpha=.3).attach_to(base)
    # arm.rand_conf()
    # # tgt_pos = rm.np.array([0.73, 0.518, 0.58])
    # # tgt_rotmat = rm.np.array([[0.30810811, 0.95135135, 0.],
    # #                           [0.95135135, -0.30810811, 0.],
    # #                           [0., 0., -1.]])
    # # tgt_pos = rm.vec(.1, -.3, .3)
    # # tgt_rotmat = rm.rotmat_from_euler(0, rm.pi / 3, 0)
    # mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    # candidate_jnt_values = ik(arm.jlc, tgt_pos, tgt_rotmat, option="multiple")
    # if candidate_jnt_values is not None:
    #     print(candidate_jnt_values)
    #     for jnt_values in candidate_jnt_values:
    #         arm.goto_given_conf(jnt_values=jnt_values)
    #         arm_mesh = arm.gen_meshmodel(alpha=.3)
    #         arm_mesh.attach_to(base)
    #         tmp_arm_stick = arm.gen_stickmodel(toggle_flange_frame=True)
    #         tmp_arm_stick.attach_to(base)
    # base.run()

    count = 0
    for i in tqdm(range(100)):
        jnt_vals = arm.rand_conf()
        tgt_pos, tgt_rotmat = arm.fk(jnt_values=jnt_vals)
        candidate_jnt_values = ik(arm.jlc, tgt_pos, tgt_rotmat, 36)
        if candidate_jnt_values is not None:
            count += 1
    print(count/100)
