from typing import Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt

from wrs.modeling import geometric_model as gm
from .rmap import MapBase


class ReachabilityMap4D(MapBase):
    """4D reachability map indexed by (z, theta, x*, y*).

    The map internally stores reachability over a *reduced* parameter space
    (z, theta, x*, y*). It is **not** a Cartesian voxel grid over (x, y, z).

    To query reachability for a concrete EE pose, always use
    ``get_indices_for_ee_pose`` + ``is_reachable``. To obtain a Cartesian
    occupancy grid for visualization or volumetric planning.
    """

    def __init__(
            self,
            xy_limits: Sequence[float] | None = None,
            z_limits: Sequence[float] | None = None,
            voxel_res: float = 0.05,
            n_bins_theta: int = 36,
            no_map: bool = False,
    ) -> None:
        if xy_limits is None:
            xy_limits = (-1.05, 1.05)
        if z_limits is None:
            z_limits = (0.0, 1.35)

        # [min, max]
        self.xy_limits = np.asarray(xy_limits, dtype=float)
        self.z_limits = np.asarray(z_limits, dtype=float)
        self.theta_limits = np.array((0.0, np.pi), dtype=float)

        # grid resolution and dimensions
        self.voxel_res: float = float(voxel_res)
        self.n_bins_xy: int = int(np.ceil((self.xy_limits[1] - self.xy_limits[0]) / self.voxel_res))
        self.n_bins_z: int = int(np.ceil((self.z_limits[1] - self.z_limits[0]) / self.voxel_res))
        self.n_bins_theta: int = int(n_bins_theta)

        # check achieved resolution
        assert np.isclose((self.xy_limits[1] - self.xy_limits[0]) / self.n_bins_xy, self.voxel_res)
        assert np.isclose((self.z_limits[1] - self.z_limits[0]) / self.n_bins_z, self.voxel_res)
        self.theta_res: float = (self.theta_limits[1] - self.theta_limits[0]) / self.n_bins_theta

        # create map
        if no_map:
            self.map: np.ndarray | None = None
        else:
            self.map = np.zeros(
                shape=(self.n_bins_z, self.n_bins_theta, self.n_bins_xy, self.n_bins_xy),
                dtype=bool,
            )

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def to_file(self, filename: str) -> None:
        """Save map and configuration to a numpy ``.npy`` file."""
        if self.map is None:
            raise ValueError("Reachability map is empty – nothing to save.")

        save_dict = {
            "map": self.map,
            "xy_limits": self.xy_limits,
            "z_limits": self.z_limits,
            "voxel_res": self.voxel_res,
            "n_bins_theta": self.n_bins_theta,
        }
        np.save(filename, save_dict)

    @classmethod
    def from_file(cls, filename: str) -> "ReachabilityMap4D":
        """Load map and configuration from ``to_file`` output."""
        data = np.load(filename, allow_pickle=True).item()
        rm = cls(
            xy_limits=data["xy_limits"],
            z_limits=data["z_limits"],
            voxel_res=data["voxel_res"],
            n_bins_theta=data["n_bins_theta"],
            no_map=True,
        )
        rm.map = data["map"]

        print(f"{cls.__name__} loaded from {filename}")
        rm.print_structure()
        return rm

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------
    def print_structure(self) -> None:
        """Pretty-print the structure and memory usage of the map."""
        if self.map is None:
            print("Reachability map is empty (no_map=True).")
            return

        print("Structure of reachability map:")
        print(f"\txy: {self.xy_limits.tolist()}, {self.n_bins_xy} bins, {self.voxel_res} m resolution")
        print(f"\tz: {self.z_limits.tolist()}, {self.n_bins_z} bins, {self.voxel_res} m resolution")
        print(f"\ttheta: {self.theta_limits.tolist()}, {self.n_bins_theta} bins, {self.theta_res:.4f} rad resolution")
        print(f"total elements: {self.map.size}")
        print(f"memory required: {self.map.nbytes / 1024 / 1024:.2f} MB")

    def get_p_z(self, tf_ee):
        """
        Gets the z-coordinate of the EE position.
        """
        return tf_ee[2, 3]

    def get_rotation_2d(self, tf_ee):
        """
        Gives the rotation that aligns tf_ee such that its z-axis is in the x+z plane as 2d rotation matrix.
        """
        rz_x, rz_y = tf_ee[:2, 2]  # first two components of the z-axis
        # get angle of rotation
        psi = np.arctan2(rz_y, rz_x)
        # build inverse rotation matrix to rotate back by psi
        rot_mat_2d = np.array([
            [np.cos(psi), np.sin(psi)],
            [-np.sin(psi), np.cos(psi)]
        ])

        return rot_mat_2d

    def get_theta(self, tf_ee):
        """
        Gets the angle between EE's r_z and the world z axis in rad.
        """
        # dot product: [0, 0, 1] dot [rz_x, rz_y, rz_z] -- simplifies to rz_z
        rz_z = tf_ee[2, 2]
        theta = np.arccos(rz_z)
        return theta

    def get_canonical_base_position(self, tf_ee):
        """
        Calculates (x*, y*) for a given EE pose.
        """
        p_x, p_y = tf_ee[:2, 3]
        rot2d = self.get_rotation_2d(tf_ee)
        x_star, y_star = rot2d @ np.array([-p_x, -p_y])
        return x_star, y_star

    def get_z_index(self, p_z):
        """
        Given a p_z, gives the corresponding index in the map.
        """
        z_idx = int((p_z - self.z_limits[0]) / self.voxel_res)
        if z_idx < 0:
            raise IndexError(f'z idx < 0 -- {p_z}')
        if z_idx >= self.n_bins_z:
            raise IndexError(f'z idx too large -- {p_z}')
        return z_idx

    def get_theta_index(self, theta):
        """
        Given the value of theta, gives the corresponding index.
        """
        # if theta is pi, we want it to be included in the last bin
        if np.isclose(theta, np.pi):
            return self.n_bins_theta - 1

        theta_idx = int((theta - self.theta_limits[0]) / self.theta_res)
        if theta_idx < 0:
            raise IndexError(f"theta index < 0 for theta={theta}")
        if theta_idx >= self.n_bins_theta:
            raise IndexError(f"theta index too large for theta={theta}")
        return theta_idx

    def get_xy_index(self, xy):
        """
        Given x* or y* from the canonical base position, gives the corresponding index.
        """
        x_idx = int((xy - self.xy_limits[0]) / self.voxel_res)
        if x_idx < 0:
            raise IndexError(f'xy_idx < 0 -- {xy}')
        if x_idx >= self.n_bins_xy:
            raise IndexError(f'xy idx too large -- {xy}')
        return x_idx

    def get_indices_for_ee_pose(self, tf_ee):
        """
        Gives the indices to the element of the reachability map that corresponds to the given end-effector pose.
        May throw an IndexError if the pose is not in the map.

        :param tf_ee: ndarray (4, 4), end-effector pose
        :returns: tuple, indices for the map
        """
        # perform the dimensionality reduction
        p_z = self.get_p_z(tf_ee)
        theta = self.get_theta(tf_ee)
        x_star, y_star = self.get_canonical_base_position(tf_ee)

        # determine indices
        z_idx = self.get_z_index(p_z)
        theta_idx = self.get_theta_index(theta)
        x_idx = self.get_xy_index(x_star)
        y_idx = self.get_xy_index(y_star)

        return z_idx, theta_idx, x_idx, y_idx

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        if self.map is None:
            raise ValueError("Reachability map has not been allocated (no_map=True).")
        return self.map.shape

    def mark_reachable(self, map_indices: Tuple[int, int, int, int]) -> None:
        if self.map is None:
            raise ValueError("Reachability map has not been allocated (no_map=True).")
        self.map[map_indices] = True

    def is_reachable(self, map_indices: Tuple[int, int, int, int]) -> bool:
        if self.map is None:
            raise ValueError("Reachability map has not been allocated (no_map=True).")
        return bool(self.map[map_indices])

    def is_reachable_world_coords(self, tf_ee: np.ndarray) -> bool:
        """
        Convenience method to check reachability directly from an end-effector pose.
        :param tf_ee: (4, 4) ndarray, end-effector pose
        :return: bool, whether the pose is reachable
        """
        indices = self.get_indices_for_ee_pose(tf_ee)
        return self.is_reachable(indices)

    def _get_xy_points(self):
        """
        gives a set of points, where each point is in the centre of an xy bin
        :returns: ndarray, (n_bins_xy, n_bins_xy, 2)
        """
        points = np.empty(shape=(self.n_bins_xy, self.n_bins_xy, 2))
        for i in range(self.n_bins_xy):
            coord = self.xy_limits[0] + (i + 0.5) * self.voxel_res  # use center point of bin
            points[i, :, 0] = coord
            points[:, i, 1] = coord
        return points

    def get_base_positions(self, tf_ee: np.ndarray, as_3d: bool = False) -> np.ndarray:
        """
        Inverse operation - retrieves the base position given an end-effector pose.
        :param tf_ee: (4, 4) ndarray, requested pose of end-effector.
        :param as_3d: bool, if True, will add a 0 z-coordinate to the points
        :return: (n, 3) ndarray, containing (x, y, score); (x, y, z, score) if as_3d is set to True
        """
        if self.map is None:
            raise ValueError("Reachability map has not been allocated (no_map=True).")
        # IDENTIFICATION
        # given the tf_ee, we need to determine z_ee and theta, such that we can retrieve the x/y slice
        p_z = self.get_p_z(tf_ee)
        z_idx = self.get_z_index(p_z)
        theta = self.get_theta(tf_ee)
        theta_idx = self.get_theta_index(theta)
        map_2d = self.map[z_idx, theta_idx]
        # BACK-PROJECTION
        # we want to transform the x/y slice into 3d points w/ additional field for the value
        # the 3d points get back-projected, i.e. un-rotate, and then shift xy
        pts = self._get_xy_points()  # (n_bins_xy, n_bins_xy, 2)
        pts = pts.reshape(-1, 2)
        # apply inverse rotation
        rot_2d = self.get_rotation_2d(tf_ee)
        pts = (np.linalg.inv(rot_2d) @ pts.T).T
        pts[:, 0] += tf_ee[0, 3]
        pts[:, 1] += tf_ee[1, 3]

        pts = pts.reshape(self.n_bins_xy, self.n_bins_xy, 2)

        if as_3d:
            points_3d = np.concatenate(
                [
                    pts,
                    np.zeros((self.n_bins_xy, self.n_bins_xy, 1)),
                    map_2d.reshape(self.n_bins_xy, self.n_bins_xy, 1),
                ],
                axis=-1,
            )
            return points_3d.reshape(-1, 4)

        points_2d = np.concatenate(
            [
                pts,
                map_2d.reshape(self.n_bins_xy, self.n_bins_xy, 1),
            ],
            axis=-1,
        )
        return points_2d.reshape(-1, 3)

    # -------------------------------------------------------------------------
    # Visualization utilities
    # -------------------------------------------------------------------------
    def show_occupancy_per_dim(self) -> None:
        """Plot occupancy distribution along each dimension using Matplotlib."""

        if self.map is None:
            raise ValueError("Reachability map has not been allocated (no_map=True).")

        non_zero = int(np.count_nonzero(self.map))
        print("OCCUPANCY OF REACHABILITY MAP")
        print(f"  total elements: {self.map.size}")
        print(f"  non-zero elements: {non_zero}")
        print(f"  percentage: {non_zero / self.map.size:.6f}")
        print(
            f"  statistics – mean: {np.mean(self.map):.6f}, "
            f"min: {np.min(self.map)}, max: {np.max(self.map)}"
        )

        # sum occupancy along the other dimensions
        z_data = np.apply_over_axes(np.sum, self.map, [1, 2, 3]).flatten()
        theta_data = np.apply_over_axes(np.sum, self.map, [0, 2, 3]).flatten()
        x_data = np.apply_over_axes(np.sum, self.map, [0, 1, 3]).flatten()
        y_data = np.apply_over_axes(np.sum, self.map, [0, 1, 2]).flatten()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

        axes[0, 0].bar(np.arange(self.n_bins_z), z_data)
        axes[0, 0].set_title(f"z {self.z_limits.tolist()}")
        axes[0, 0].set_xlabel("z-bin index")
        axes[0, 0].set_ylabel("occupied voxels")

        axes[0, 1].bar(np.arange(self.n_bins_theta), theta_data)
        axes[0, 1].set_title(f"theta {self.theta_limits.tolist()}")
        axes[0, 1].set_xlabel("theta-bin index")

        axes[1, 0].bar(np.arange(self.n_bins_xy), x_data)
        axes[1, 0].set_title(f"x {self.xy_limits.tolist()}")
        axes[1, 0].set_xlabel("x*-bin index")

        axes[1, 1].bar(np.arange(self.n_bins_xy), y_data)
        axes[1, 1].set_title(f"y {self.xy_limits.tolist()}")
        axes[1, 1].set_xlabel("y*-bin index")

        fig.tight_layout()
        plt.show()

    def visualize_in_sim(
            self,
            tf_ee: np.ndarray,
            skip_zero: bool = True,
            radius: float = 0.025,
    ) -> list[gm.StaticGeometricModel]:
        """Visualize reachability slice as spheres using ``geometric_model``.

        Parameters
        ----------
        base_world : ShowBase or panda3d world-like object
            Target render context. Typically the Panda3D ``World`` or
            ``ShowBase`` instance used elsewhere in WRS.
        tf_ee : (4, 4) ndarray
            End-effector pose for which to visualize base reachability.
        skip_zero : bool, default True
            If ``True``, voxels with score 0 are not rendered.
        radius : float, default 0.025
            Sphere radius in meters.

        Returns
        -------
        list[StaticGeometricModel]
            List of created geometric models (one per sphere) so that the
            caller can manage their lifecycle if needed.
        """

        if self.map is None:
            raise ValueError("Reachability map has not been allocated (no_map=True).")
        points_4d = self.get_base_positions(tf_ee, as_3d=True)
        scores = points_4d[:, 3]
        max_val = float(np.max(scores)) if np.any(scores) else 1.0
        models: list[gm.StaticGeometricModel] = []
        for point, val in zip(points_4d[:, :3], scores):
            if skip_zero and val == 0:
                continue
            # simple green↔red colormap depending on occupancy
            t = float(val) / max_val if max_val > 0 else 0.0
            color = np.array([1.0 - t, t, 0.0])
            sphere = gm.gen_sphere(pos=np.asarray(point), radius=radius, rgb=color)
            models.append(sphere)
        return models

    def gen_volume_visualization(
            self,
            base_origin: Sequence[float] | None = None,
            occupied_rgb: np.ndarray | None = None,
            free_rgb: np.ndarray | None = None,
            alpha: float = 0.4,
            step_z: int = 1,
            step_xy: int = 1,
    ) -> gm.mmc.ModelCollection:
        """Generate a 3D voxel visualization by sampling RM4D into Cartesian space.

        Parameters
        ----------
        base_origin : sequence of float, optional
            World pose of the robot base origin *(x, y, z)*. The reachability
            map itself is defined for a base at (0, 0, 0). This argument lets
            you visualize the same map for a robot that is located at an
            arbitrary position in the world. If ``None`` (default), the base is
            assumed to be at the world origin.
        occupied_rgb, free_rgb : np.ndarray, optional
            RGB colors used for occupied / free voxels. Each should be a
            length-3 array in [0, 1]. If ``free_rgb`` is ``None`` (default),
            free space is not visualized.
        alpha : float, default 0.4
            Alpha value for the voxel boxes.
        step_z, step_xy : int, default 1
            Sub-sampling factors along z and xy to speed up visualization
            (larger values → fewer voxels).
        """
        if self.map is None:
            raise ValueError("Reachability map has not been allocated (no_map=True).")
        # ------------------------------------------------------------------
        # 1) Sample RM4D into a world-aligned Cartesian occupancy grid.
        #    This converts the (z, theta, x*, y*) map into a 3D (x, y, z)
        #    occupancy grid for a robot base at (0, 0, 0).
        # ------------------------------------------------------------------
        # The map's xy_limits are for the canonical (x*, y*) space. The true
        # workspace is rotationally symmetric. The max reach in the XY plane is
        # related to the extent of the canonical space.
        max_r_star = np.sqrt(2 * self.xy_limits[1] ** 2)
        cart_xy_lim = (-max_r_star, max_r_star)
        n_bins_cart_xy = int(np.ceil((cart_xy_lim[1] - cart_xy_lim[0]) / self.voxel_res))
        occ_3d_full = np.zeros((self.n_bins_z, n_bins_cart_xy, n_bins_cart_xy), dtype=bool)
        origin_min = np.array([cart_xy_lim[0], cart_xy_lim[0], self.z_limits[0]])
        # Vectorized implementation to accelerate the process
        reachable_indices = np.argwhere(self.map)
        if reachable_indices.shape[0] == 0:
            return gm.mmc.ModelCollection(name="reachability_volume")

        z_indices = reachable_indices[:, 0]
        x_star_indices = reachable_indices[:, 2]
        y_star_indices = reachable_indices[:, 3]

        x_star_vals = self.xy_limits[0] + (x_star_indices + 0.5) * self.voxel_res
        y_star_vals = self.xy_limits[0] + (y_star_indices + 0.5) * self.voxel_res

        psi_samples = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        cos_psi = np.cos(psi_samples)
        sin_psi = np.sin(psi_samples)

        # Vectorized rotation
        # p_x shape: (N, 12), p_y shape: (N, 12)
        p_x = -(cos_psi[np.newaxis, :] * x_star_vals[:, np.newaxis] - sin_psi[np.newaxis, :] * y_star_vals[:, np.newaxis])
        p_y = -(sin_psi[np.newaxis, :] * x_star_vals[:, np.newaxis] + cos_psi[np.newaxis, :] * y_star_vals[:, np.newaxis])

        # Flatten and get grid indices
        p_x_flat = p_x.flatten()
        p_y_flat = p_y.flatten()
        z_indices_rep = np.repeat(z_indices, len(psi_samples))

        ix = ((p_x_flat - cart_xy_lim[0]) / self.voxel_res).astype(int)
        iy = ((p_y_flat - cart_xy_lim[0]) / self.voxel_res).astype(int)

        # Filter out-of-bounds indices
        valid_mask = (ix >= 0) & (ix < n_bins_cart_xy) & (iy >= 0) & (iy < n_bins_cart_xy)
        ix_valid = ix[valid_mask]
        iy_valid = iy[valid_mask]
        iz_valid = z_indices_rep[valid_mask]

        # Populate the 3D grid
        occ_3d_full[iz_valid, iy_valid, ix_valid] = True

        # ------------------------------------------------------------------
        # 2) Down-sample indices for faster visualization.
        # ------------------------------------------------------------------
        occ_3d = occ_3d_full[::step_z, ::step_xy, ::step_xy]
        voxel_res_vis_xy = self.voxel_res * step_xy
        voxel_res_vis_z = self.voxel_res * step_z
        # ------------------------------------------------------------------
        # 3) Decide the *unshifted* grid origin in the original RM frame
        #    (where the base used for map construction is at (0, 0, 0)).
        # ------------------------------------------------------------------
        origin_rm = origin_min
        # ------------------------------------------------------------------
        # 4) Apply the user-specified base_origin shift to place the same
        #    occupancy volume into the world frame where the actual robot base
        #    sits at ``base_origin``.  If ``base_origin`` is None, this is
        #    simply the original RM frame (i.e. equivalent to old behavior).
        # ------------------------------------------------------------------
        if base_origin is None:
            base_shift = np.zeros(3, dtype=float)
        else:
            base_shift = np.asarray(base_origin, dtype=float).reshape(3, )
        # Final origin of voxel (0,0,0) in world coordinates
        origin_world = origin_rm + base_shift
        # ------------------------------------------------------------------
        # 5) Create voxel geometry.
        # ------------------------------------------------------------------
        voxel_extent = np.array([voxel_res_vis_xy, voxel_res_vis_xy, voxel_res_vis_z], dtype=float)
        occ_color = occupied_rgb if occupied_rgb is not None else np.array([0.0, 1.0, 0.0])
        free_color = free_rgb if free_rgb is not None else None
        mc = gm.mmc.ModelCollection(name="reachability_volume")
        nz, ny, nx = occ_3d.shape
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    val = occ_3d[iz, iy, ix]
                    if val:
                        color = occ_color
                    elif free_color is not None:
                        color = free_color
                    else:
                        continue
                    # Voxel center in world coordinates. ``origin_world`` is
                    # the minimum corner of voxel (0,0,0) in the *world* frame
                    # after shifting by ``base_origin``.
                    cx = origin_world[0] + (ix + 0.5) * voxel_res_vis_xy
                    cy = origin_world[1] + (iy + 0.5) * voxel_res_vis_xy
                    cz = origin_world[2] + (iz * step_z + 0.5 * step_z) * self.voxel_res

                    box = gm.gen_box(
                        xyz_lengths=voxel_extent,
                        pos=np.array([cx, cy, cz], dtype=float),
                        rgb=color,
                        alpha=alpha,
                    )
                    mc.add_gm(box)

        return mc




