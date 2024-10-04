import time
from wrs import basis, robot_sim as rkmg, robot_sim as rskj, modeling as gm
import numpy as np
import wrs.visualization.panda.world as wd

if __name__ == '__main__':
    base = wd.World(cam_pos=[1.25, .75, .75], lookat_pos=[0, 0, .3])
    gm.gen_frame().attach_to(base)

    jlc = rskj.JLChain(n_dof=6)
    jlc.jnts[0].loc_pos = np.array([0, 0, 0])
    jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[0].motion_range = np.array([-np.pi / 2, np.pi / 2])
    # jlc.joints[1].change_type(rkc.JntType.PRISMATIC)
    jlc.jnts[1].loc_pos = np.array([0, 0, .05])
    jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
    jlc.jnts[1].motion_range = np.array([-np.pi / 2, np.pi / 2])
    jlc.jnts[2].loc_pos = np.array([0, 0, .2])
    jlc.jnts[2].loc_motion_ax = np.array([0, 1, 0])
    jlc.jnts[2].motion_range = np.array([-np.pi, np.pi])
    jlc.jnts[3].loc_pos = np.array([0, 0, .2])
    jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[3].motion_range = np.array([-np.pi / 2, np.pi / 2])
    jlc.jnts[4].loc_pos = np.array([0, 0, .1])
    jlc.jnts[4].loc_motion_ax = np.array([0, 1, 0])
    jlc.jnts[4].motion_range = np.array([-np.pi / 2, np.pi / 2])
    jlc.jnts[5].loc_pos = np.array([0, 0, .05])
    jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[5].motion_range = np.array([-np.pi / 2, np.pi / 2])
    jlc._loc_flange_pos = np.array([0, 0, .01])
    jlc.finalize()
    rkmg.gen_jlc_stick(jlc, stick_rgba=basis.constant.navy_blue, toggle_tcp_frame=True).attach_to(base)
    seed_jnt_vals = jlc.get_jnt_values()
    # seed_jnt_values = np.array([0.69103164, -1.42838988, 1.1103724, 0.94371771, -0.64419981,
    #                           1.23253726])
    # tgt_pos = np.array([-0.04016656, -0.16026002, 0.05019466])
    # tgt_rotmat = np.array([[0.49100466, 0.6792442, 0.54547386],
    #                        [0.33862061, 0.4281006, -0.83789376],
    #                        [-0.80265217, 0.59611844, -0.01980669]])
    # tgt_pos = np.array([-0.026943, 0.15541492, 0.15165676])
    # tgt_rotmat = np.array([[0.1139233, 0.94946477, -0.29246902],
    #                        [0.9883866, -0.07851674, 0.130104],
    #                        [0.10056545, -0.30389433, -0.94738315]])
    tgt_pos = np.array([0.03306253, 0.04412065, -0.14756892])
    tgt_rotmat = np.array([[-0.29832769, 0.91526211, -0.27073208],
                           [0.50770504, 0.39236168, 0.76699929],
                           [0.80823028, 0.09136508, -0.58173554]])

    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    tic = time.time()
    joint_values_with_dbg_info = jlc.ik(tgt_pos=tgt_pos,
                                        tgt_rotmat=tgt_rotmat,
                                        seed_jnt_values=seed_jnt_vals,
                                        max_n_iter=100,
                                        toggle_dbg=True)
    toc = time.time()
    print(f"time cost is {toc - tic}")
    print(joint_values_with_dbg_info)
    jlc.fk(jnt_values=joint_values_with_dbg_info[1], update=True)
    rkmg.gen_jlc_stick(jlc, stick_rgba=basis.constant.navy_blue, toggle_tcp_frame=True).attach_to(base)
    base.run()
    # success = 0
    # num_win = 0
    # opt_win = 0
    # time_list = []
    # tgt_list = []
    # for i in tqdm(range(100), desc='ik'):
    #     jnts = jlc.rand_conf()
    #     tgt_pos, tgt_rotmat = jlc.forward_kinematics(jnt_values=jnts, update=False, toggle_jacobian=False)
    #     a = time.time()
    #     joint_values_with_dbg_info = jlc.ik(tgt_pos=tgt_pos,
    #                                         tgt_rotmat=tgt_rotmat,
    #                                         seed_jnt_values=seed_jnt_values,
    #                                         max_n_iter=100,
    #                                         toggle_dbg_info=True)
    #     b = time.time()
    #     time_list.append(b - a)
    #     if joint_values_with_dbg_info is not None:
    #         success += 1
    #         print(joint_values_with_dbg_info)
    #         if joint_values_with_dbg_info[0] == 'o':
    #             opt_win += 1
    #         elif joint_values_with_dbg_info[0] == 'n':
    #             num_win += 1
    #     else:
    #         print(tgt_pos, tgt_rotmat)
    #         tgt_list.append([tgt_pos, tgt_rotmat])
    #         base.run()
    # with open("unsolved.pkl", "wb") as f_:
    #     pickle.dump(tgt_list, f_)
    # print(success)
    # print(f'num_win: {num_win}, opt_win: {opt_win}')
    # print('average', np.mean(time_list))
    # print('max', np.max(time_list))
    # print('min', np.min(time_list))
    #
    # f_name = "unsolved.pkl"
    # for i in range(5):
    #     with open(f_name, "rb") as a:
    #         tgt_list = pickle.load(a)
    #     success = 0
    #     num_win = 0
    #     opt_win = 0
    #     new_tgt_list = []
    #     for id in tqdm(range(len(tgt_list)), desc="failed iks"):
    #         joint_values_with_dbg_info = jlc.ik(tgt_pos=tgt_list[id][0],
    #                                             tgt_rotmat=tgt_list[id][1],
    #                                             seed_jnt_values=seed_jnt_values,
    #                                             toggle_dbg_info=True)
    #         print("ik is done!")
    #         print(joint_values_with_dbg_info)
    #         if joint_values_with_dbg_info is not None:
    #             success += 1
    #             if joint_values_with_dbg_info[0] == 'o':
    #                 opt_win += 1
    #             elif joint_values_with_dbg_info[0] == 'n':
    #                 num_win += 1
    #         else:
    #             new_tgt_list.append([tgt_list[id][0], tgt_list[id][1]])
    #     print(f"success rate: {success}/{len(tgt_list)}")
    #     print(f'num_win: {num_win}, opt_win: {opt_win}')
    #     with open("new_unsolved.pkl", "wb") as f_:
    #         pickle.dump(new_tgt_list, f_)
    #     f_name = "new_unsolved.pkl"
    # base.run()
