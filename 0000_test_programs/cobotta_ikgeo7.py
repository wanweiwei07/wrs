import time
from wrs import wd, mcm, rm
import wrs.robot_sim.manipulators.cobotta.cvr038 as cbta
import wrs.robot_sim._kinematics.ikgeo.sp4_lib as sp4_lib
import wrs.robot_sim._kinematics.ikgeo.sp3_lib as sp3_lib
import wrs.robot_sim._kinematics.ikgeo.sp1_lib as sp1_lib
import matplotlib.pyplot as plt


def err_given_q4(q4, jlc, p06, R06):
    R34 = rm.rotmat_from_axangle(jlc.jnts[3].loc_motion_ax, q4)
    h2 = jlc.jnts[1].loc_motion_ax
    p45 = jlc.jnts[4].loc_pos + rm.np.array([0, jlc.jnts[5].loc_pos[1], 0])
    p34 = jlc.jnts[3].loc_pos
    p23 = jlc.jnts[2].loc_pos
    R34p45 = R34 @ p45
    # subproblem 4 for q1
    h = h2
    p = p06
    d = h2.T @ (p23 + p34 + R34p45)
    k = -jlc.jnts[0].loc_motion_ax
    q1_candidates, is_ls = sp4_lib.sp4_run(p, k, h, d)
    if is_ls:
        return None, None
    # q1_candidates is always an np.array when is_ls is False
    # filter valid q1 solutions
    q1_min, q1_max = jlc.jnts[0].motion_range
    q1_valid = rm.np.array([q1 for q1 in q1_candidates if q1_min <= q1 <= q1_max])
    # subproblem 3 for q3
    p1 = p34 + R34p45
    p2 = -p23
    d = rm.np.linalg.norm(p06)
    k = jlc.jnts[2].loc_motion_ax
    q3_candidates, is_ls = sp3_lib.sp3_run(p1, p2, k, d)
    # print("q3 ", q3_candidates, jlc.jnts[2].motion_range, is_ls)
    if is_ls:
        return None, None
    # q3_candidates is always an np.array when is_ls is False
    # filter valid q3 solutions
    q3_min, q3_max = jlc.jnts[2].motion_range
    q3_valid = rm.np.array([q3 for q3 in q3_candidates if q3_min <= q3 <= q3_max])
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
            p2 = p23 + R23 @ p34 + R23 @ R34p45
            k = -jlc.jnts[1].loc_motion_ax
            q2, is_ls = sp1_lib.sp1_run(p1, p2, k)
            # print("q2 ", q2, jlc.jnts[1].motion_range, is_ls)
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


def search1d(jlc, start, end, n_div, p06, R06, toggle_plt=False):
    q4_candidates = rm.np.linspace(start, end, n_div)
    previous_errs = None
    previous_q4 = None
    zero_crossings = []
    for q4 in q4_candidates:
        current_errs, current_q123s = err_given_q4(q4, jlc, p06, R06)
        if toggle_plt:
            plt.plot([q4] * len(current_errs), current_errs, 'ro', linestyle=None)
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
        if jlc.jnts[5].motion_range[0] < q6 < arm.jlc.jnts[5].motion_range[1]:
            return q5, q6
    return None, None


if __name__ == '__main__':

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, .3])
    mcm.mgm.gen_frame().attach_to(base)
    arm = cbta.CVR038()

    # tgt_pos = rm.vec(.1, -.3, .3)
    # tgt_rotmat = rm.rotmat_from_euler(0, rm.pi / 3, 0)
    # tgt_pos = rm.vec(0.04448112, 0.16670252, 0.46935543)
    # tgt_rotmat = rm.np.array([[-0.54495302, -0.11483439, 0.83056563],
    #                           [-0.49965492, -0.75100382, -0.43166911],
    #                           [0.67332842, -0.65023559, 0.35188423]])

    ddik_time = []
    geoik_time = []
    geo_all = []
    geo_seed = []

    success_rate = 0
    n_value = 100
    for i in range(n_value):
        rand_conf = arm.rand_conf()
        tgt_pos, tgt_rotmat = arm.fk(jnt_values=rand_conf)
        mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        _p12 = arm.jlc.jnts[1].loc_pos
        R06 = tgt_rotmat
        p06 = tgt_pos - _p12 - R06 @ rm.np.array([0, 0, arm.jlc.jnts[5].loc_pos[2]])
        tic = time.time()
        zero_crossings = search1d(arm.jlc, arm.jlc.jnts[3].motion_range[0], arm.jlc.jnts[3].motion_range[1], 360, p06,
                                  R06)
        # for _, _, _, q4_cross in zero_crossings:
        #     current_errs, _ = err_given_q4(q4_cross, arm.jlc, p06, R06)
        #     plt.plot(q4_cross, min(current_errs), 'go', linestyle=None)
        # plt.show()

        # geo_all.append(0)
        # geo_seed.append(0)
        candidate_jnt_values = []
        for q1, q2, q3, q4 in zero_crossings:
            q5, q6 = solve_q56(arm.jlc, R06, q1, q2, q3, q4)
            if q5 is not None:
                # geo_all[-1] += 1
                result = arm.jlc.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=[q1, q2, q3, q4, q5, q6])
                if result is not None:
                    # geo_seed[-1] += 1
                    candidate_jnt_values.append(result)
        toc = time.time()
        geoik_time.append(toc - tic)
        tic = time.time()
        result = arm.jlc.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
        toc = time.time()
        ddik_time.append(toc - tic)
        if len(candidate_jnt_values) != 0:
            success_rate += 1

    import matplotlib.pyplot as plt

    plt.plot(range(n_value), ddik_time, 'r')
    plt.plot(range(n_value), geoik_time, 'g')
    # plt.plot(range(n_value), geo_all, 'm--')
    # plt.plot(range(n_value), geo_seed, 'b-.')
    print(success_rate)
    plt.show()


    class Data(object):
        def __init__(self, rbt, candidate_jnt_values):
            self.rbt = rbt
            self.counter = 0
            self.candidate_jnt_values = candidate_jnt_values
            self.mesh_onscreen = []


    anime_data = Data(rbt=arm, candidate_jnt_values=candidate_jnt_values)


    def update(anime_data, task):
        for item in anime_data.mesh_onscreen:
            item.detach()
        if anime_data.counter >= len(anime_data.candidate_jnt_values):
            # for mesh_model in anime_data.mot_data.mesh_list:
            #     mesh_model.detach()
            anime_data.counter = 0
        # print(anime_data.counter)
        # print(rm.np.degrees(anime_data.candidate_jnt_values[anime_data.counter]))
        anime_data.rbt.goto_given_conf(jnt_values=anime_data.candidate_jnt_values[anime_data.counter])
        anime_data.mesh_onscreen.append(anime_data.rbt.gen_meshmodel(alpha=.3))
        anime_data.mesh_onscreen[-1].attach_to(base)
        anime_data.mesh_onscreen.append(anime_data.rbt.gen_stickmodel())
        anime_data.mesh_onscreen[-1].attach_to(base)
        if base.inputmgr.keymap['space']:
            anime_data.counter += 1
        # time.sleep(.5)
        return task.again


    taskMgr.doMethodLater(0.1, update, "update",
                          extraArgs=[anime_data],
                          appendTask=True)
    base.run()
