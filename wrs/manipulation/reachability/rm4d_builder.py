"""Example script for constructing and persisting a 4D reachability map for a UR3e robot.

This module can be executed as a script to:

- Sample the UR3e joint space (Use rand_conf)
- Build a 4D reachability map over (x, y, z, theta)
- Persist the reachability map and hit statistics to disk

The main entry point is :func:`build_ur3e_reachability_map`.
Author: Hao Chen (chen960216@gmail.com)
Date: 20251216fuzhou
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import os

import wrs.manipulation.reachability.rm4d as rm4d
import wrs.robot_sim.manipulators.ur3e.ur3e as ur3e

XYLimits = Tuple[float, float]
ZLimits = Tuple[float, float]


@dataclass(frozen=True)
class UR3eReachabilityConfig:
    """Configuration parameters for UR3e reachability map construction."""
    n_samples: int = 1_000_000
    xy_limits: XYLimits = (-0.6, 0.6)
    z_limits: ZLimits = (-0.8, 0.8)
    voxel_resolution: float = 0.05
    n_theta_bins: int = 36

    def output_dir(self, base_dir: os.PathLike | str) -> Path:
        """Return the directory where reachability artifacts will be saved."""
        robot_name = ur3e.UR3e.__name__.lower()
        base_dir_p = Path(base_dir).resolve()
        return base_dir_p / "rm4d_data" / f"{robot_name}_rm4d"


def build_ur3e_reachability_map(
        *,
        config: UR3eReachabilityConfig | None = None,
        base_dir: os.PathLike | str | None = None,
) -> None:
    """Construct and store a 4D reachability map for a UR3e robot.

    Parameters
    ----------
    config:
        Optional configuration object. If omitted, :class:`UR3eReachabilityConfig`
        defaults are used.
    base_dir:
        Base directory used to resolve the output directory. Defaults to the
        directory containing this file.
    """
    cfg = config or UR3eReachabilityConfig()
    if base_dir is None:
        base_dir = Path(__file__).parent
    output_dir = cfg.output_dir(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    robot = ur3e.UR3e(enable_cc=True)
    print("Setting up reachability map construction...")
    rmap = rm4d.reachability_map.ReachabilityMap4D(
        xy_limits=list(cfg.xy_limits),
        z_limits=list(cfg.z_limits),
        voxel_res=cfg.voxel_resolution,
        n_bins_theta=cfg.n_theta_bins,
    )
    constructor = rm4d.construction.JointSpaceConstructor(
        rmap=rmap,
        robot=robot,
    )
    constructor.sample(n_samples=cfg.n_samples, )
    rmap.to_file(str(output_dir / "rmap"))
    print("Finished saving reachability map. The map is stored at:", output_dir / "rmap")


if __name__ == "__main__":
    build_ur3e_reachability_map()
