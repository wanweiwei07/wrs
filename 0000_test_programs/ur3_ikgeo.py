import time
from wrs import wd, mcm, rm
from wrs.robot_sim.manipulators.ur3 import ur3 as ur3a
import wrs.robot_sim._kinematics.ikgeo.sp5_lib as sp5_lib
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
R06 = tgt_rotmat
p01 = arm.jlc.jnts[0].loc_pos
h2 = arm.jlc.jnts[1].loc_motion_ax
p06 = tgt_pos - p01 - R06 @ arm.jnts[5].loc_rotmat @ arm.jnts[5].loc_pos
p12 = arm.jlc.jnts[1].loc_pos
p23 = arm.jlc.jnts[2].loc_pos
p34 = arm.jlc.jnts[3].loc_pos
p45 = arm.jlc.jnts[4].loc_pos
# subproblem for q1
h = arm.jlc.jnts[1].loc_rotmat @ h2
p = p06
d = h.T @ p12 + h2.T @ (p23 + p34 + p45)
k = -arm.jlc.jnts[0].loc_motion_ax
q1_cadidates, is_ls = sp4_lib.sp4_run(p, k, h, d)
if not is_ls:
    for q in q1_cadidates:
        if arm.jlc.jnts[0].motion_range[0] < q < arm.jlc.jnts[0].motion_range[1]:
            q1 = q
            print("q1 ", q1)
            # subproblem for q5
            h6 = arm.jlc.jnts[5].loc_motion_ax
            R01 = rm.rotmat_from_axangle(arm.jlc.jnts[0].loc_motion_ax, q1)
            h = (h2.T @ arm.jlc.jnts[4].loc_rotmat).T
            p = arm.jlc.jnts[5].loc_rotmat @ h6
            d = (arm.jlc.jnts[1].loc_rotmat @ h2).T @ R01.T @ R06 @ h6
            k = arm.jlc.jnts[4].loc_motion_ax
            q5_candidates, is_ls = sp4_lib.sp4_run(p, k, h, d)
            if not is_ls:
                for q in q5_candidates:
                    if arm.jlc.jnts[4].motion_range[0] < q < arm.jlc.jnts[4].motion_range[1]:
                        q5 = q
                        print("q5 ", q5)
                        # subproblem 1 for q6
                        R45 = rm.rotmat_from_axangle(arm.jlc.jnts[4].loc_motion_ax, q5)
                        p1 = arm.jnts[5].loc_rotmat.T @ R45.T @ arm.jnts[4].loc_rotmat.T @ h2
                        p2 = R06.T @ R01 @ arm.jnts[1].loc_rotmat @ h2
                        k = -arm.jlc.jnts[5].loc_motion_ax
                        q6, is_ls = sp1_lib.sp1_run(p1, p2, k)
                        # arm.gen_meshmodel(alpha=.3).attach_to(base)
                        # jnt_values = arm.get_jnt_values()
                        # jnt_values[0]=q1
                        # jnt_values[4]=q5
                        # jnt_values[5]=q6
                        # arm.goto_given_conf(jnt_values)
                        # arm.gen_meshmodel().attach_to(base)
                        # base.run()
                        if not is_ls:
                            if arm.jlc.jnts[5].motion_range[0] < q6 < arm.jlc.jnts[5].motion_range[1]:
                                print("q6 ", q6)
                                R56 = rm.rotmat_from_axangle(arm.jlc.jnts[5].loc_motion_ax, q6)
                                # subprolbem 3 for q3
                                p1 = p34
                                p2 = -p23
                                d = R01.T @ p06 - p12 - R01.T @ R06 @ R56.T @ arm.jnts[5].loc_rotmat.T @ R45.T @ \
                                    arm.jnts[4].loc_rotmat.T @ p45
                                k = arm.jlc.jnts[2].loc_motion_ax
                                q3_candidates, is_ls = sp3_lib.sp3_run(p1, p2, d, k)
                                print(q3_candidates)
                                if type(q3_candidates) is not list:
                                    q3_candidates = [q3_candidates]
                                # if not is_ls:
                                for q in q3_candidates:
                                    if arm.jlc.jnts[2].motion_range[0] < q < arm.jlc.jnts[2].motion_range[1]:
                                        q3 = q
                                        arm.gen_meshmodel(alpha=.3).attach_to(base)
                                        jnt_values = arm.get_jnt_values()
                                        jnt_values[0]=q1
                                        jnt_values[2]=q3
                                        jnt_values[4]=q5
                                        jnt_values[5]=q6
                                        arm.goto_given_conf(jnt_values)
                                        arm.gen_meshmodel().attach_to(base)
base.run()


print(len(candidate_jnt_values))


class Data(object):
    def __init__(self, rbt, candidate_jnt_values):
        self.rbt = rbt
        self.counter = 0
        self.candidate_jnt_values = candidate_jnt_values
        self.mesh_onscreen = None


anime_data = Data(rbt=arm, candidate_jnt_values=candidate_jnt_values)


def update(anime_data, task):
    if anime_data.counter > 0:
        anime_data.mesh_onscreen.detach()
    if anime_data.counter >= len(anime_data.candidate_jnt_values):
        # for mesh_model in anime_data.mot_data.mesh_list:
        #     mesh_model.detach()
        anime_data.counter = 0
    print(anime_data.counter)
    anime_data.rbt.goto_given_conf(jnt_values=anime_data.candidate_jnt_values[anime_data.counter])
    anime_data.mesh_onscreen = anime_data.rbt.gen_meshmodel()
    anime_data.mesh_onscreen.attach_to(base)
    if base.inputmgr.keymap['space']:
        anime_data.counter += 1
    # time.sleep(.5)
    return task.again


taskMgr.doMethodLater(0.1, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()
