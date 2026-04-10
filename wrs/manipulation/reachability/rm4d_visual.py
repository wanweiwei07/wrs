import itertools
import os
import time

import numpy as np

import wrs.basis.robot_math as rm
import wrs.manipulation.reachability.rm4d as rm4d
import wrs.modeling.geometric_model as gm
import wrs.robot_sim.manipulators.ur3e.ur3e as ur3e
import wrs.visualization.panda.world as wd


def create_robot_with_reachability_map() -> tuple[ur3e.UR3e, rm4d.ReachabilityMap4D]:
    """Instantiate the UR3e robot and load its 4D reachability map from disk."""
    robot = ur3e.UR3e(enable_cc=True)
    rmap_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "rm4d_data",
        f"{robot.name}_rm4d",
        "rmap.npy",
    )
    rmap = rm4d.ReachabilityMap4D.from_file(rmap_path)
    return robot, rmap


def benchmark_random_queries(robot: ur3e.UR3e, rmap: rm4d.ReachabilityMap4D, n_samples: int = 10_000) -> float:
    """Benchmark average query time of the reachability map using random robot configurations.

    Returns
    -------
    float
        Average query time in microseconds.
    """
    total_elapsed = 0.0
    for _ in range(n_samples):
        q_rand = robot.rand_conf()
        ee_pos, ee_rot = robot.fk(q_rand, update=True)
        ee_homomat = rm.homomat_from_posrot(ee_pos, ee_rot)

        start = time.time()
        indices = rmap.get_indices_for_ee_pose(ee_homomat)
        _ = rmap.is_reachable(indices)
        end = time.time()

        total_elapsed += end - start

    return total_elapsed / n_samples * 1e6


def main() -> None:
    """Entry-point for benchmarking and visualization of the UR3e 4D reachability map."""
    base = wd.World(cam_pos=rm.vec(1, 1.5, 1.5), lookat_pos=rm.vec(0, 0, 0))
    robot, rmap = create_robot_with_reachability_map()
    robot.gen_meshmodel().attach_to(base)
    gm.gen_frame().attach_to(base)
    rmap.gen_volume_visualization(
        alpha=.1,
        step_z=2,
        free_rgb=(.8, .8, .8),
        step_xy=2
    ).attach_to(base)
    base.run()


if __name__ == "__main__":
    main()
