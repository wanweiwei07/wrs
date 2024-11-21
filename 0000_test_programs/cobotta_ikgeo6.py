import time
from wrs import wd, mcm, rm
import wrs.robot_sim.manipulators.cobotta_arm.cobotta_arm as cbta
import wrs.robot_sim._kinematics.ikgeo.sp4_lib as sp4_lib
import wrs.robot_sim._kinematics.ikgeo.sp3_lib as sp3_lib
import wrs.robot_sim._kinematics.ikgeo.sp1_lib as sp1_lib

base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)
arm = cbta.CVR038()

ddik_time = []
geoik_time = []
geo_all = []
geo_seed = []

n_value = 100
for i in range(n_value):
    rand_conf = arm.rand_conf()
    tgt_pos, tgt_rotmat = arm.fk(jnt_values=rand_conf)
    # tgt_pos = rm.vec(.2, .2, .1)
    # tgt_rotmat = rm.rotmat_from_euler(0, rm.pi/4, 0)
    # tgt_pos = rm.vec(.1, -.3, .3)
    # tgt_rotmat = rm.rotmat_from_euler(0, rm.pi / 3, 0)
    mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

    _p12 = arm.jlc.jnts[1].loc_pos
    R06 = tgt_rotmat
    p06 = tgt_pos - _p12 - R06 @ rm.np.array([0, 0, arm.jlc.jnts[5].loc_pos[2]])
    mcm.mgm.gen_myc_frame(pos=p06, rotmat=R06).attach_to(base)

    candidate_jnt_values = []
    prev_error = rm.nan
    tic =time.time()
    for q4 in rm.np.linspace(arm.jlc.jnts[3].motion_range[0], arm.jlc.jnts[3].motion_range[1], 6):
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
                                        if prev_error is not rm.nan and e_q4 * prev_error < 0:
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
                                                    candidate_jnt_values.append([q1, q2, q3, q4, q5, q6])
                                        prev_error = e_q4
    new_result = []
    for jnt_values in candidate_jnt_values:
        result = arm.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=jnt_values)
        if result is not None:
            # print(result)
            new_result.append(result)
    toc=time.time()
    # print("1D search time: ", toc-tic)
    geoik_time.append(toc-tic)
    tic = time.time()
    arm.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    toc=time.time()
    # print("DDIK time: ", toc-tic)
    ddik_time.append(toc-tic)
    # print(len(candidate_jnt_values))
    # print(len(new_result))
    if len(candidate_jnt_values) > 16:
        print(tgt_pos, tgt_rotmat)
    geo_all.append(len(candidate_jnt_values))
    geo_seed.append(len(new_result))

print("avg DDIK time: ", sum(ddik_time)/len(ddik_time))
print("avg 1D search time: ", sum(geoik_time)/len(geoik_time))

import matplotlib.pyplot as plt
# plt.plot(range(n_value), ddik_time, 'r')
# plt.plot(range(n_value), geoik_time, 'g')
plt.plot(range(n_value), geo_all, 'm--')
plt.plot(range(n_value), geo_seed, 'b-.')
plt.show()

class Data(object):
    def __init__(self, rbt, candidate_jnt_values):
        self.rbt = rbt
        self.counter = 0
        self.candidate_jnt_values = candidate_jnt_values
        self.mesh_onscreen = []


anime_data = Data(rbt=arm, candidate_jnt_values=new_result)


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

print(len(candidate_jnt_values))
for jnt_values in candidate_jnt_values:
    arm.goto_given_conf(jnt_values=rm.vec(*jnt_values))
    arm.gen_meshmodel(alpha=.3).attach_to(base)
    center = arm.gl_tcp_pos - arm.gl_tcp_rotmat @ rm.np.array([0, 0, arm.jlc.jnts[5].loc_pos[2]])
    mcm.mgm.gen_sphere(pos=center, radius=.005).attach_to(base)

# ddik
jnt_values = arm.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
if jnt_values is not None:
    arm.goto_given_conf(jnt_values=jnt_values)
    # arm.jlc._ik_solver.test_success_rate()
    arm_mesh = arm.gen_meshmodel(alpha=.3)
    arm_mesh.attach_to(base)
    tmp_arm_stick = arm.gen_stickmodel(toggle_flange_frame=True)
    tmp_arm_stick.attach_to(base)
else:
    print("DDIK failed.")
base.run()
