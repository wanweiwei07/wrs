import time
from wrs import wd, mcm, rm
from wrs.robot_sim.manipulators.ur3 import ur3 as ur3a
import wrs.robot_sim._kinematics.ikgeo.sp4_lib as sp4_lib
import wrs.robot_sim._kinematics.ikgeo.sp3_lib as sp3_lib
import wrs.robot_sim._kinematics.ikgeo.sp1_lib as sp1_lib

base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)
arm = ur3a.UR3()

tgt_pos = rm.vec(.25, .1, .1)
tgt_rotmat = rm.rotmat_from_euler(0, rm.pi, 0)
mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

candidate_jnt_values = []
tic = time.time()
R06 = tgt_rotmat
p01 = arm.jlc.jnts[0].loc_pos
h2 = arm.jlc.jnts[1].loc_motion_ax
p06 = tgt_pos - p01 - R06 @ rm.vec(0, 0, arm.jnts[5].loc_pos[1])
p12 = arm.jlc.jnts[1].loc_pos
p23 = arm.jlc.jnts[2].loc_pos
p34 = arm.jlc.jnts[3].loc_pos
p45 = arm.jlc.jnts[4].loc_pos
# subproblem 4 for q1
h = arm.jlc.jnts[1].loc_rotmat @ h2
p = p06
d = h2.T @ (p23 + p34 + p45)
k = -arm.jlc.jnts[0].loc_motion_ax
q1_cadidates, is_ls = sp4_lib.sp4_run(p, k, h, d)
if not is_ls:
    for q in q1_cadidates:
        if arm.jlc.jnts[0].motion_range[0] < q < arm.jlc.jnts[0].motion_range[1]:
            q1 = q
            # print("q1 ", q1, arm.jlc.jnts[0].motion_range)
            # subproblem 4 for q5
            h6 = arm.jlc.jnts[5].loc_motion_ax
            R01 = rm.rotmat_from_axangle(arm.jlc.jnts[0].loc_motion_ax, q1)
            h = (h2.T @ arm.jlc.jnts[4].loc_rotmat).T
            p = arm.jlc.jnts[5].loc_rotmat @ h6
            R10R06 = R01.T @ R06
            d = (arm.jlc.jnts[1].loc_rotmat @ h2).T @ R10R06 @ h6
            k = arm.jlc.jnts[4].loc_motion_ax
            q5_candidates, is_ls = sp4_lib.sp4_run(p, k, h, d)
            if not is_ls:
                for q in q5_candidates:
                    if arm.jlc.jnts[4].motion_range[0] < q < arm.jlc.jnts[4].motion_range[1]:
                        q5 = q
                        # print("q5 ", q5, arm.jlc.jnts[4].motion_range)
                        # subproblem 1 for q6
                        R45 = rm.rotmat_from_axangle(arm.jlc.jnts[4].loc_motion_ax, q5)
                        p1 = arm.jnts[5].loc_rotmat.T @ R45.T @ arm.jnts[4].loc_rotmat.T @ h2
                        p2 = R10R06.T @ arm.jnts[1].loc_rotmat @ h2
                        k = -arm.jlc.jnts[5].loc_motion_ax
                        q6, is_ls = sp1_lib.sp1_run(p1, p2, k)
                        if not is_ls:
                            if arm.jlc.jnts[5].motion_range[0] < q6 < arm.jlc.jnts[5].motion_range[1]:
                                # print("q6 ", q6, arm.jlc.jnts[5].motion_range)
                                R56 = rm.rotmat_from_axangle(arm.jlc.jnts[5].loc_motion_ax, q6)
                                # subprolbem 3 for q3
                                p1 = p34
                                p2 = -p23
                                d_ = R01.T @ p06 - R10R06 @ R56.T @ arm.jnts[5].loc_rotmat.T @ R45.T @ \
                                     arm.jnts[4].loc_rotmat.T @ p45
                                d = rm.np.linalg.norm(d_)
                                k = arm.jlc.jnts[2].loc_motion_ax
                                q3_candidates, is_ls = sp3_lib.sp3_run(p1, p2, k, d)
                                if not is_ls:
                                    if not isinstance(q3_candidates, rm.np.ndarray):
                                        q3_candidates = [q3_candidates]
                                    # if not is_ls:
                                    for q in q3_candidates:
                                        if arm.jlc.jnts[2].motion_range[0] < q < arm.jlc.jnts[2].motion_range[1]:
                                            q3 = q
                                            # print("q3 ", q3, arm.jlc.jnts[2].motion_range)
                                            R23 = rm.rotmat_from_axangle(arm.jlc.jnts[2].loc_motion_ax, q3)
                                            # subproblem 1 for q2
                                            p1 = p23 + R23 @ p34
                                            p2 = arm.jnts[1].loc_rotmat.T @ d_
                                            k = arm.jlc.jnts[1].loc_motion_ax
                                            q2, is_ls = sp1_lib.sp1_run(p1, p2, k)
                                            if not is_ls:
                                                if arm.jlc.jnts[1].motion_range[0] < q2 < arm.jlc.jnts[1].motion_range[1]:
                                                    # print("q2 ", q2, arm.jlc.jnts[1].motion_range)
                                                    # subproblem 1 for q4
                                                    R12 = rm.rotmat_from_axangle(arm.jlc.jnts[1].loc_motion_ax, q2)
                                                    p1 = p45
                                                    p2 = R23.T @ R12.T @ arm.jnts[1].loc_rotmat.T @ (
                                                            R01.T @ p06 - arm.jnts[1].loc_rotmat @ R12 @ (
                                                            p23 + R23 @ p34))
                                                    k = arm.jlc.jnts[3].loc_motion_ax
                                                    q4, is_ls = sp1_lib.sp1_run(p1, p2, k)
                                                    if not is_ls:
                                                        if arm.jlc.jnts[3].motion_range[0] < q4 < arm.jlc.jnts[3].motion_range[
                                                            1]:
                                                            # print("q4 ", q4, arm.jlc.jnts[3].motion_range)
                                                            candidate_jnt_values.append([q1, q2, q3, q4, q5, q6])
toc = time.time()
print(toc-tic)
print(len(candidate_jnt_values))


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
    print(anime_data.counter)
    anime_data.rbt.goto_given_conf(jnt_values=anime_data.candidate_jnt_values[anime_data.counter])
    anime_data.mesh_onscreen.append(anime_data.rbt.gen_meshmodel())
    anime_data.mesh_onscreen[-1].attach_to(base)
    if base.inputmgr.keymap['space']:
        anime_data.counter += 1
    # time.sleep(.5)
    return task.again


taskMgr.doMethodLater(0.1, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()
