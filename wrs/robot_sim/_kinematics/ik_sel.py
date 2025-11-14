import os
import pickle
import numpy as np
import scipy.spatial
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import wrs.basis.robot_math as rm
import wrs.robot_sim._kinematics.ik_num as ikn
import wrs.robot_sim._kinematics.ik_opt as iko
import wrs.robot_sim._kinematics.ik_trac as ikt
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim._kinematics.model_generator as rkmg
import wrs.modeling.geometric_model as mgm
import wrs.basis.utils as bu
import samply

class SELIKSolver(object):
    def __init__(self, jlc, path=None, identifier_str='test', backbone_solver='n', rebuild=False):
        """
        :param jlc:
        :param path:
        :param backbone_solver: 'n': num ik; 'o': opt ik; 't': trac ik
        :param rebuild:
        author: weiwei
        date: 20231111
        """
        self.jlc = jlc
        current_file_dir = os.path.dirname(__file__)
        if path is None:
            path = os.path.join(os.path.dirname(current_file_dir), "_data_files")
        self._fname_tree = os.path.join(path, f"{identifier_str}_iksel_tree.pkl")
        self._fname_jnt = os.path.join(path, f"{identifier_str}_iksel_jnt_data.pkl")
        self._k_max = 200  # maximum nearest neighbours examined by the backbone solver
        self._max_n_iter = 7  # max_n_iter of the backbone solver
        if backbone_solver == 'n':
            self._backbone_solver = ikn.NumIKSolver(self.jlc)
        elif backbone_solver == 'o':
            self._backbone_solver = iko.OptIKSolver(self.jlc)
        elif backbone_solver == 't':
            self._backbone_solver = ikt.TracIKSolver(self.jlc)
        if rebuild:
            print("Rebuilding the database. It starts a new evolution and is costly.")
            y_or_n = bu.get_yesno()
            if y_or_n == 'y':
                self.query_tree, self.jnt_data, self.tcp_data, self.jinv_data = self._build_data()
                self.persist_data()
        else:
            try:
                with open(self._fname_tree, 'rb') as f_tree:
                    self.query_tree = pickle.load(f_tree)
                with open(self._fname_jnt, 'rb') as f_jnt:
                    self.jnt_data, self.tcp_data, self.jinv_data = pickle.load(f_jnt)
            except FileNotFoundError:
                self.query_tree, self.jnt_data, self.tcp_data, self.jinv_data = self._build_data()
                self.persist_data()

    def __call__(self,
                 tgt_pos,
                 tgt_rotmat,
                 seed_jnt_values=None,
                 max_n_iter=None,
                 toggle_dbg=False):
        """
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param max_n_iter: use self._max_n_iter if None
        :param toggle_evolve: do we update the database file
        :param toggle_dbg:
        :return:
        """
        return self.ik(tgt_pos=tgt_pos,
                       tgt_rotmat=tgt_rotmat,
                       seed_jnt_values=seed_jnt_values,
                       max_n_iter=max_n_iter,
                       toggle_dbg=toggle_dbg)

    def _rotmat_to_vec(self, rotmat, method='v'):
        """
        convert a rotmat to vectors
        this will be used for computing the Minkowski p-norm required by KDTree query
        'f' or 'q' are recommended, they both have satisfying performance
        :param method: 'f': Frobenius; 'q': Quaternion; 'r': rpy; 'v': rotvec; '-': same value
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
        if method == 'v':
            return Rotation.from_matrix(rotmat).as_rotvec()
        if method == '-':
            return np.array([0])

    def _build_data(self):
        # gen normalized qs first and then linearly scale to sampled qs
        sample_number = 40320 if self.jlc.n_dof == 6 else 201600
        print("Generating normalized uniform samples using CVT...")
        normalized_qs = samply.hypercube.cvt(sample_number, self.jlc.n_dof)
        print(f"Building Data for SELIK using CVT for {sample_number} uniform samples...")
        sampled_qs = self.jlc.jnt_ranges[:, 0] + normalized_qs * (self.jlc.jnt_ranges[:, 1] - self.jlc.jnt_ranges[:, 0])

        query_data = []
        jnt_data = []
        jinv_data = []
        for id in tqdm(range(len(sampled_qs))):
            jnt_values = sampled_qs[id]
            # pinv of jacobian
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=jnt_values, toggle_jacobian=True)
            jinv = np.linalg.pinv(j_mat, rcond=1e-4)
            # jinv = np.linalg.inv(j_mat.T @ j_mat + 1e-4 * np.eye(j_mat.shape[1])) @ j_mat.T
            # relative to base
            rel_pos, rel_rotmat = rm.rel_pose(self.jlc.pos, self.jlc.rotmat, flange_pos, flange_rotmat)
            rel_rotvec = self._rotmat_to_vec(rel_rotmat)
            query_data.append(rel_pos.tolist() + rel_rotvec.tolist())
            jnt_data.append(jnt_values)
            jinv_data.append(jinv)
        query_tree = scipy.spatial.cKDTree(query_data)
        return query_tree, np.asarray(jnt_data), np.asarray(query_data), np.asarray(jinv_data)

    def persist_data(self):
        os.makedirs(os.path.dirname(self._fname_tree), exist_ok=True)
        with open(self._fname_tree, 'wb') as f_tree:
            pickle.dump(self.query_tree, f_tree)
        with open(self._fname_jnt, 'wb') as f_jnt:
            pickle.dump([self.jnt_data, self.tcp_data, self.jinv_data], f_jnt)
        print("selik data file saved.")

    def ik(self,
           tgt_pos,
           tgt_rotmat,
           seed_jnt_values=None,
           max_n_iter=None,
           toggle_dbg=False):
        """
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values: ignored
        :param toggle_dbg: ignored
        :return:
        author: weiwei
        date: 20231107
        """
        max_n_iter = self._max_n_iter if max_n_iter is None else max_n_iter
        if seed_jnt_values is not None:
            return self._backbone_solver(tgt_pos=tgt_pos,
                                         tgt_rotmat=tgt_rotmat,
                                         seed_jnt_values=seed_jnt_values,
                                         max_n_iter=max_n_iter,
                                         toggle_dbg=toggle_dbg)
        else:
            # relative to base
            rel_pos, rel_rotmat = rm.rel_pose(self.jlc.pos, self.jlc.rotmat, tgt_pos, tgt_rotmat)
            rel_rotvec = self._rotmat_to_vec(rel_rotmat)
            query_point = np.concatenate((rel_pos, rel_rotvec))
            dist_value_list, nn_indx_list = self.query_tree.query(query_point, k=self._k_max, workers=-1)
            if type(nn_indx_list) is int:
                nn_indx_list = [nn_indx_list]
            seed_jnt_array = self.jnt_data[nn_indx_list]
            seed_tcp_array = self.tcp_data[nn_indx_list]
            seed_jinv_array = self.jinv_data[nn_indx_list]
            seed_posrot_diff_array = query_point - seed_tcp_array
            adjust_array = np.einsum('ijk,ik->ij', seed_jinv_array, seed_posrot_diff_array)
            square_sums = np.sum((adjust_array) ** 2, axis=1)
            sorted_indices = np.argsort(square_sums)
            seed_jnt_array_cad = seed_jnt_array[sorted_indices[:20]]
            for id, seed_jnt_values in enumerate(seed_jnt_array_cad):
                if id > 3:
                    return None
                if toggle_dbg:
                    rkmg.gen_jlc_stick_by_jnt_values(self.jlc,
                                                     jnt_values=seed_jnt_values,
                                                     stick_rgba=rm.const.red).attach_to(base)
                result = self._backbone_solver(tgt_pos=tgt_pos,
                                               tgt_rotmat=tgt_rotmat,
                                               seed_jnt_values=seed_jnt_values,
                                               max_n_iter=max_n_iter,
                                               toggle_dbg=toggle_dbg)
                if result is None:
                    nid = id+1
                    distances = np.linalg.norm(nid*seed_jnt_array_cad[nid:] - np.sum(seed_jnt_array_cad[:nid], axis=0), axis=1)
                    sorted_cad_indices = np.argsort(-distances)
                    seed_jnt_array_cad[nid:] = seed_jnt_array_cad[nid:][sorted_cad_indices]
                    continue
                else:
                    return result
            return None


if __name__ == '__main__':
    import math
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1.25, .75, .75], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)

    _jnt_safemargin = math.pi / 18.0
    jlc = rkjlc.JLChain(n_dof=7)
    jlc.anchor.pos = np.array([.0, .0, .3])
    jlc.anchor.rotmat = rm.rotmat_from_euler(np.pi / 3, 0, 0)
    jlc.jnts[0].loc_pos = np.array([.0, .0, .0])
    jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, np.pi)
    jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[0].motion_range = np.array([-2.94087978961 + _jnt_safemargin, 2.94087978961 - _jnt_safemargin])
    jlc.jnts[1].loc_pos = np.array([0.03, .0, .1])
    jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(np.pi / 2, 0.0, 0.0)
    jlc.jnts[1].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[1].motion_range = np.array([-2.50454747661 + _jnt_safemargin, 0.759218224618 - _jnt_safemargin])
    jlc.jnts[2].loc_pos = np.array([-0.03, 0.17283, 0.0])
    jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(-np.pi / 2, 0.0, 0.0)
    jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[2].motion_range = np.array([-2.94087978961 + _jnt_safemargin, 2.94087978961 - _jnt_safemargin])
    jlc.jnts[3].loc_pos = np.array([-0.04188, 0.0, 0.07873])
    jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(np.pi / 2, -np.pi / 2, 0.0)
    jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[3].motion_range = np.array([-2.15548162621 + _jnt_safemargin, 1.3962634016 - _jnt_safemargin])
    jlc.jnts[4].loc_pos = np.array([0.0405, 0.16461, 0.0])
    jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(-np.pi / 2, 0.0, 0.0)
    jlc.jnts[4].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[4].motion_range = np.array([-5.06145483078 + _jnt_safemargin, 5.06145483078 - _jnt_safemargin])
    jlc.jnts[5].loc_pos = np.array([-0.027, 0, 0.10039])
    jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(np.pi / 2, 0.0, 0.0)
    jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[5].motion_range = np.array([-1.53588974176 + _jnt_safemargin, 2.40855436775 - _jnt_safemargin])
    jlc.jnts[6].loc_pos = np.array([0.027, 0.029, 0.0])
    jlc.jnts[6].loc_rotmat = rm.rotmat_from_euler(-np.pi / 2, 0.0, 0.0)
    jlc.jnts[6].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[6].motion_range = np.array([-3.99680398707 + _jnt_safemargin, 3.99680398707 - _jnt_safemargin])
    jlc._loc_flange_pos = np.array([0, 0, .007])
    jlc.finalize(ik_solver='s', identifier_str="test_new")
    jlc.test_ik_success_rate()

    # goal_jnt_values = jlc.rand_conf()
    # rkmg.gen_jlc_stick_by_jnt_values(jlc, jnt_values=goal_jnt_values, stick_rgba=rm.bc.blue).attach_to(base)
    #
    # tgt_pos, tgt_rotmat = jlc.fk(jnt_values=goal_jnt_values)
    # tic = time.time()
    #
    # jnt_values = jlc.ik(tgt_pos=tgt_pos,
    #                     tgt_rotmat=tgt_rotmat,
    #                     toggle_dbg=False)
    # toc = time.time()
    # mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    # print(toc - tic, jnt_values)
    # base.run()
    # if jnt_values is not None:
    #     jlc.goto_given_conf(jnt_values=jnt_values)
    #     rkmg.gen_jlc_stick(jlc, stick_rgba=rm.bc.navy_blue, toggle_flange_frame=True,
    #                        toggle_jnt_frames=False).attach_to(base)
    #     base.run()

    # jlc._ik_solver._test_success_rate()
    # jlc._ik_solver.multiepoch_evolve(n_times_per_epoch=10000)
    # jlc._ik_solver.test_success_rate()
    base.run()