"""
Data driven ik solver
"""

import numpy as np
import pickle
import basis.robot_math as rm
import scipy.spatial
from tqdm import tqdm
import robot_sim.kinematics.ik_num as rkn
import robot_sim.kinematics.ik_opt as rko
import robot_sim.kinematics.ik_trac as rkt


def data_builder(jlc, path='./'):
    # gen sampled qs
    sampled_jnts = []
    n_intervals = np.linspace(8, 8, jlc.n_dof, endpoint=True)
    for i in range(jlc.n_dof):
        print(int(n_intervals[i]))
        sampled_jnts.append(np.linspace(jlc.jnt_rngs[i][0], jlc.jnt_rngs[i][1], int(n_intervals[i]), endpoint=False))
    grid = np.meshgrid(*sampled_jnts)
    sampled_qs = np.vstack([x.ravel() for x in grid]).T
    # gen sampled qs and their correspondent tcps
    tcp_data = []
    tcp_vecmat_data = []
    jnt_data = []
    for id in tqdm(range(len(sampled_qs))):
        jnt_vals = sampled_qs[id]
        tmp_tcp_pos, tmp_tcp_rotmat = jlc.forward_kinematics(jnt_vals=jnt_vals, toggle_jac=False)
        tmp_tcp_w = rm.delta_w_between_rotmat(tmp_tcp_rotmat, np.eye(3))
        tcp_data.append(np.concatenate((tmp_tcp_pos, tmp_tcp_w)))
        tcp_vecmat_data.append((tmp_tcp_pos, tmp_tcp_rotmat))
        jnt_data.append(jnt_vals)
    querry_tree = scipy.spatial.cKDTree(tcp_data)
    pickle.dump(querry_tree, open(path + 'ikdd_tree.pkl', 'wb'))
    pickle.dump(jnt_data, open(path + 'jnt_data.pkl', 'wb'))
    return querry_tree, tcp_data, jnt_data


class DDIKSolver(object):
    def __init__(self, jlc, path='./', rebuild=False):
        self.jlc = jlc
        if rebuild:
            data_builder(jlc)
        else:
            try:
                self.querry_tree = pickle.load(open(path + 'ikdd_tree.pkl', 'rb'))
                self.jnt_data = pickle.load(open(path + 'jnt_data.pkl', 'rb'))
            except FileNotFoundError:
                self.querry_tree, self.tcp_vecmat_data, self.jnt_data = data_builder(jlc)
        self._nik_solver = rkn.NumIKSolver(self.jlc)
        # self._oik_solver = rko.OptIKSolver(self.jlc)
        # self.tik_solver = rkt.TracIKSolver(self.jlc)

    def _custom_distance(self, tcp_vecmat1, tcp_vecmat2):
        pos_err, rot_err, delta = rm.diff_between_posrot(tcp_vecmat1[0], tcp_vecmat1[1], tcp_vecmat2[0], tcp_vecmat2[1])
        return pos_err + rot_err

    def ik(self,
           tgt_pos,
           tgt_rotmat,
           seed_jnt_vals=None,
           max_n_iter=100,
           toggle_dbg_info=False):
        tmp_tcp_w = rm.delta_w_between_rotmat(tgt_rotmat, np.eye(3))
        tgt_tcp = np.concatenate((tgt_pos, tmp_tcp_w))
        dist_val, nn_indx = self.querry_tree.query(tgt_tcp, k=1)
        seed_jnt_vals = self.jnt_data[nn_indx]
        return self._nik_solver.pinv_rr(tgt_pos=tgt_pos,
                                        tgt_rotmat=tgt_rotmat,
                                        seed_jnt_vals=seed_jnt_vals,
                                        max_n_iter=max_n_iter)
        # return self._oik_solver.sqpss(tgt_pos=tgt_pos,
        #                               tgt_rotmat=tgt_rotmat,
        #                               seed_jnt_vals=seed_jnt_vals,
        #                               max_n_iter=max_n_iter)
        # return self.tik_solver.ik(tgt_pos=tgt_pos,
        #                           tgt_rotmat=tgt_rotmat,
        #                           seed_jnt_vals=seed_jnt_vals,
        #                           max_n_iter=max_n_iter)


if __name__ == '__main__':
    import modeling.geometric_model as gm
    import robot_sim.kinematics.jlchain as rskj
    import time
    import basis.constant as bc
    import robot_sim.kinematics.model_generator as rkmg
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[1.25, .75, .75], lookat_pos=[0, 0, .3])
    gm.gen_frame().attach_to(base)

    jlc = rskj.JLChain(n_dof=6)
    jlc.jnts[0].loc_pos = np.array([0, 0, 0])
    jlc.jnts[0].loc_motion_axis = np.array([0, 0, 1])
    jlc.jnts[0].motion_rng = np.array([-np.pi / 2, np.pi / 2])
    # jlc.joints[1].change_type(rkc.JointType.PRISMATIC)
    jlc.jnts[1].loc_pos = np.array([0, 0, .05])
    jlc.jnts[1].loc_motion_axis = np.array([0, 1, 0])
    jlc.jnts[1].motion_rng = np.array([-np.pi / 2, np.pi / 2])
    jlc.jnts[2].loc_pos = np.array([0, 0, .2])
    jlc.jnts[2].loc_motion_axis = np.array([0, 1, 0])
    jlc.jnts[2].motion_rng = np.array([-np.pi, np.pi])
    jlc.jnts[3].loc_pos = np.array([0, 0, .2])
    jlc.jnts[3].loc_motion_axis = np.array([0, 0, 1])
    jlc.jnts[3].motion_rng = np.array([-np.pi / 2, np.pi / 2])
    jlc.jnts[4].loc_pos = np.array([0, 0, .1])
    jlc.jnts[4].loc_motion_axis = np.array([0, 1, 0])
    jlc.jnts[4].motion_rng = np.array([-np.pi / 2, np.pi / 2])
    jlc.jnts[5].loc_pos = np.array([0, 0, .05])
    jlc.jnts[5].loc_motion_axis = np.array([0, 0, 1])
    jlc.jnts[5].motion_rng = np.array([-np.pi / 2, np.pi / 2])
    jlc.tcp_loc_pos = np.array([0, 0, .01])
    jlc.finalize(ik_solver_class=DDIKSolver)
    seed_jnt_vals = jlc.get_joint_values()

    # random_jnts = jlc.rand_conf()
    # tgt_pos, tgt_rotmat = jlc.forward_kinematics(jnt_vals=random_jnts, update=False, toggle_jac=False)
    # tic = time.time()
    # jnt_vals = jlc.ik(tgt_pos=tgt_pos,
    #                   tgt_rotmat=tgt_rotmat,
    #                   seed_jnt_vals=seed_jnt_vals,
    #                   max_n_iter=100)
    # gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    # jlc.forward_kinematics(jnt_vals=jnt_vals, update=True, toggle_jac=False)
    # rkmg.gen_jlc_stick(jlc, stick_rgba=bc.navy_blue, toggle_tcp_frame=True,
    #                    toggle_joint_frame=True).attach_to(base)
    # base.run()

    success = 0
    num_win = 0
    opt_win = 0
    time_list = []
    tgt_list = []
    for i in tqdm(range(1000), desc="ik"):
        random_jnts = jlc.rand_conf()
        tgt_pos, tgt_rotmat = jlc.forward_kinematics(jnt_vals=random_jnts, update=False, toggle_jac=False)
        tic = time.time()
        joint_values_with_dbg_info = jlc.ik(tgt_pos=tgt_pos,
                                            tgt_rotmat=tgt_rotmat,
                                            seed_jnt_vals=seed_jnt_vals,
                                            max_n_iter=10,
                                            toggle_dbg_info=True)
        toc = time.time()
        time_list.append(toc - tic)
        if joint_values_with_dbg_info is not None:
            success += 1
        else:
            tgt_list.append((tgt_pos, tgt_rotmat))
    print(success)
    print('average', np.mean(time_list))
    print('max', np.max(time_list))
    print('min', np.min(time_list))
    print('std', np.std(time_list))
    base.run()
