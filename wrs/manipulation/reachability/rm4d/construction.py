import numpy as np
from tqdm import tqdm
from .rmap import MapBase

import wrs.basis.robot_math as rm
from wrs.robot_sim.manipulators.manipulator_interface import ManipulatorInterface


class JointSpaceConstructor:
    def __init__(self, rmap: MapBase, robot: ManipulatorInterface, seed=0):
        self._map = rmap
        self.rbt = robot
        self.rng = np.random.default_rng(seed=seed)

    def sample(self, n_samples=10_000, prevent_collisions=True, hit_stats=None):
        for _ in tqdm(range(n_samples)):
            # sample random collision-free config from robot
            q_rand = self.rbt.rand_conf()
            # get EE pose as TF_EE
            ee_pos, ee_rot = self.rbt.fk(q_rand, update=True)
            if self.rbt.is_collided():
                continue
            homomat = rm.homomat_from_posrot(ee_pos, ee_rot)
            idcs = self._map.get_indices_for_ee_pose(homomat)
            self._map.mark_reachable(idcs)

            if hit_stats is not None:
                hit_stats.record_access(idcs)
