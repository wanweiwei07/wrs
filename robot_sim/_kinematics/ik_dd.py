"""
Data driven ik solver
author: weiwei
date: 20231107
"""

import numpy as np
import pickle
import basis.robot_math as rm
import scipy.spatial
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import robot_sim._kinematics.ik_num as rkn
import robot_sim._kinematics.ik_opt as rko
import robot_sim._kinematics.ik_trac as rkt


class DDIKSolver(object):
    def __init__(self, jlc, path='./', solver='n', rebuild=False):
        """
        :param jlc:
        :param path:
        :param solver: 'n': num ik; 'o': opt ik; 't': trac ik
        :param rebuild:
        """
        self.jlc = jlc
        if rebuild:
            self._data_builder()
        else:
            try:
                self.querry_tree = pickle.load(open(path + 'ikdd_tree.pkl', 'rb'))
                self.jnt_data = pickle.load(open(path + 'jnt_data.pkl', 'rb'))
            except FileNotFoundError:
                self.querry_tree, self.jnt_data = self._data_builder()
        if solver == 'n':
            self._ik_solver = rkn.NumIKSolver(self.jlc)
            self._ik_solver_fun = self._ik_solver.pinv_wc
        elif solver == 'o':
            self._ik_solver = rko.OptIKSolver(self.jlc)
            self._ik_solver_fun = self._ik_solver.sqpss
        elif solver == 't':
            self._ik_solver = rkt.TracIKSolver(self.jlc)
            self._ik_solver_fun = self._ik_solver.ik

    def _rotmat_to_vec(self, rotmat, method='q'):
        """
        convert a rotmat to vectors
        this will facilitate the Minkowski p-norm computation required by KDTree query
        :param method: 'f': Frobenius; 'q': Quaternion; 'r': rpy; '-': same value
        :return:
        author: weiwei
        date: 20231107
        """
        if method == 'f':
            return rotmat.ravel()
        if method == 'q':
            return Rotation.from_matrix(rotmat).as_quat()
        if method == 'r':
            return rm.rotmat_to_euler(rotmat)
        if method == '-':
            return np.array([0])

    def _data_builder(self, path='./'):
        # gen sampled qs
        sampled_jnts = []
        n_intervals = np.linspace(12, 8, self.jlc.n_dof, endpoint=True)
        for i in range(self.jlc.n_dof):
            print(int(n_intervals[i]))
            sampled_jnts.append(
                np.linspace(jlc.jnt_rngs[i][0], self.jlc.jnt_rngs[i][1], int(n_intervals[i]), endpoint=False))
        grid = np.meshgrid(*sampled_jnts)
        sampled_qs = np.vstack([x.ravel() for x in grid]).T
        # gen sampled qs and their correspondent tcps
        tcp_data = []
        jnt_data = []
        for id in tqdm(range(len(sampled_qs))):
            jnt_vals = sampled_qs[id]
            tcp_pos, tcp_rotmat = self.jlc.forward_kinematics(jnt_vals=jnt_vals, toggle_jac=False)
            tcp_rotvec = self._rotmat_to_vec(tcp_rotmat)
            tcp_data.append(np.concatenate((tcp_pos, tcp_rotvec)))
            jnt_data.append(jnt_vals)
        querry_tree = scipy.spatial.cKDTree(tcp_data)
        pickle.dump(querry_tree, open(path + 'ikdd_tree.pkl', 'wb'))
        pickle.dump(jnt_data, open(path + 'jnt_data.pkl', 'wb'))
        return querry_tree, jnt_data

    def ik(self,
           tgt_pos,
           tgt_rotmat,
           seed_jnt_vals=None,
           max_n_iter=10,
           toggle_dbg=False):
        """
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_vals: ignored
        :param max_n_iter:
        :param toggle_dbg: ignored
        :return:
        author: weiwei
        date: 20231107
        """
        tcp_rotvec = self._rotmat_to_vec(tgt_rotmat)
        tgt_tcp = np.concatenate((tgt_pos, tcp_rotvec))
        # dist_val, nn_indx = self.querry_tree.query(tgt_tcp, k=1, workers=-1)
        # seed_jnt_vals = self.jnt_data[nn_indx]
        # return self._ik_solver_fun(tgt_pos=tgt_pos,
        #                            tgt_rotmat=tgt_rotmat,
        #                            seed_jnt_vals=seed_jnt_vals,
        #                            max_n_iter=max_n_iter)
        dist_val_array, nn_indx_array = self.querry_tree.query(tgt_tcp, k=10, workers=-1)
        for nn_indx in nn_indx_array:
            seed_jnt_vals = self.jnt_data[nn_indx]
            result = self._ik_solver_fun(tgt_pos=tgt_pos,
                                         tgt_rotmat=tgt_rotmat,
                                         seed_jnt_vals=seed_jnt_vals,
                                         max_n_iter=max_n_iter)
            if result is None:
                continue
            else:
                return result
        return None


if __name__ == '__main__':
    import modeling.geometric_model as gm
    import robot_sim._kinematics.jlchain as rskj
    import time
    import basis.constant as bc
    import robot_sim._kinematics.model_generator as rkmg
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
    jlc.reinitialize(ik_solver_class=DDIKSolver)
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
                                            toggle_dbg=False)
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
