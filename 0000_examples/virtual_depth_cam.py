"""
Examples to use virtual depth camera in Panda3D World.
Author: Chen Hao (chen960216@gmail.com)
Date: 20251203fuzhou
"""

import colorsys
from panda3d.core import (
    Geom, GeomNode, GeomPoints, GeomVertexFormat, GeomVertexData,
    GeomVertexWriter
)

def heat_map_color(z, min_z=0.0, max_z=0.3):
    """Generates a rainbow color based on height."""
    # Normalize z
    t = (z - min_z) / (max_z - min_z)
    t = np.clip(t, 0, 1)
    # Hue: 0.0 (Red) to 0.66 (Blue).
    # Let's go Blue(0.66) -> Red(0.0) for height (Standard heatmap)
    hue = (1.0 - t) * 0.66
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return r, g, b, 1.0


from panda3d.core import OmniBoundingVolume, GeomEnums


class PointCloudVisualizer:
    def __init__(self, base, max_points=100000, point_size=4):
        self.max_points = max_points

        # 1. Create Node and disable culling
        self.node = GeomNode('point_cloud')
        self.np = base.render.attachNewNode(self.node)
        self.np.setRenderModeThickness(point_size)
        self.np.setLightOff()

        # Force "Infinite" bounds so it never disappears
        self.node.setBounds(OmniBoundingVolume())
        self.node.setFinal(True)

        # 2. Setup Data Container (Fixed Size Pool)
        # usage=GeomEnums.UHDynamic ensures the GPU expects frequent updates
        self.vdata = GeomVertexData('pc_data', GeomVertexFormat.getV3c4(), GeomEnums.UHDynamic)

        # CRITICAL: Pre-allocate the maximum size ONCE.
        # We will never resize this buffer again. Resizing is slow.
        self.vdata.setNumRows(self.max_points)

        # 3. Setup Primitive (The "Drawer")
        self.geom = Geom(self.vdata)
        self.points = GeomPoints(GeomEnums.UHStatic)  # The indices don't change often, the data does
        self.geom.addPrimitive(self.points)
        self.node.addGeom(self.geom)

        # 4. Create Writers (Keep these persistent)
        self.writer_v = GeomVertexWriter(self.vdata, 'vertex')
        self.writer_c = GeomVertexWriter(self.vdata, 'color')

    def update(self, points_world):
        """
        Updates the point cloud geometry.
        points_world: (N, 3) numpy array
        """
        num_new_points = len(points_world)
        # Safety Check
        if num_new_points > self.max_points:
            print(f"Warning: Point cloud truncated! {num_new_points} > {self.max_points}")
            num_new_points = self.max_points
            points_world = points_world[:self.max_points]
        if num_new_points == 0:
            # If no points, just clear the drawing instruction
            self.points.clearVertices()
            return
        # 1. Reset Writers to the beginning of the buffer
        self.writer_v.setRow(0)
        self.writer_c.setRow(0)
        # 2. Write Data (Only the active points)
        # This overwrites the first N rows of the buffer.
        # Rows N to Max are left as stale garbage (we just won't draw them).
        for p in points_world:
            self.writer_v.addData3f(p[0], p[1], p[2])
            # Simple Height coloring logic
            z_norm = (p[2] - 0.0) / 0.2
            z_norm = max(0.0, min(1.0, z_norm))
            self.writer_c.addData4f(z_norm, 0, 1.0 - z_norm, 1.0)
        # 3. Update the Drawing Primitive
        # CRITICAL: We clear the "Draw List" and tell it to only draw indices 0 to N
        self.points.clearVertices()
        self.points.addNextVertices(num_new_points)


if __name__ == "__main__":
    import cv2
    import numpy as np
    from wrs import wd, rm, mgm
    from wrs.visualization.panda.panda3d_utils import VirtualDepthCamera, ExtraWindow

    base = wd.World(cam_pos=[1, 0, 1], lookat_pos=[0, 0, .05],
                    w=640, h=480)
    obj = mgm.GeometricModel("objects/bunnysim.stl", alpha=1)
    obj.attach_to(base)
    mgm.gen_frame(ax_length=0.2, ax_radius=0.005).attach_to(base)
    resolution = (640, 480)
    vcam = VirtualDepthCamera(cam_pos=[0.4, 0, 0.2],
                              lookat_pos=[0, 0, 0.05],
                              resolution=resolution,
                              screen_size=np.array((0.05, 0.05)),
                              fov=45,
                              depth_far=5.0)  # Clip depth at 5 meters
    # Create the visualization of the camera (Frustum)
    # We will update its pose in the loop
    vcam_viz_node = [vcam.gen_meshmodel(alpha=0.5)]
    vcam_viz_node[0].attach_to(base)
    vcam_pcd_node = [mgm.gen_pointcloud(vcam.get_point_cloud(), point_size=.001)]
    vcam_pcd_node[0].attach_to(base)
    target_point = obj.trm_mesh.center_mass
    vcam_pcd_node[0].detach()

    ew = ExtraWindow(base,
                     cam_pos=[.5, 0, .5], lookat_pos=[0, 0, .03],
                     w=640, h=480,)
    ew.set_origin((np.array([0, 40])))
    pc_viz = PointCloudVisualizer(ew)
    pc_viz.np.setPos(0, 0, 0)


    def update(task):
        t = task.time
        vcam_viz_node[0].detach()
        # --- A. Orbit Logic ---
        radius = 0.4
        speed = 1
        angle = t * speed
        # Calculate new position (Circular orbit at height 0.2)
        new_x = radius * np.cos(angle)
        new_y = radius * np.sin(angle)
        new_z = 0.1 + 0.1 * np.sin(angle * 0.5)  # Bobbing up and down slightly

        # Update Camera Pose
        vcam.cam_pos = np.array([new_x, new_y, new_z])
        vcam.look_at(target_point)
        # Update Visualization
        vcam_viz_node[0] = vcam.gen_meshmodel(alpha=0.5)
        vcam_viz_node[0].attach_to(base)
        # Get Images
        img_rgb = vcam.get_rgb_image()  # (H, W, 3) uint8
        img_depth = vcam.get_depth_image()  # (H, W) float32 in meters
        pcd = vcam.get_point_cloud()
        step = 10
        pcd_sparse = pcd[::step]
        pc_viz.update(pcd_sparse)
        depth_vis = img_depth.copy()
        img_depth[img_depth > 4] = 0
        max_vis_dist = np.max(img_depth)
        depth_vis = np.clip(depth_vis, 0, max_vis_dist)
        depth_norm = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        # Apply false color: Blue = Close, Red = Far
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        K = vcam.intrinsics
        p_world = target_point
        # Transform world point to camera frame
        # P_cam = R^T * (P_world - t_cam)
        p_cam = vcam.cam_rotmat.T @ (p_world - vcam.cam_pos)
        verts = obj.trm_mesh.vertices
        faces = obj.trm_mesh.faces
        # Manual projection math (using K)
        verts_world = verts @ obj.rotmat.T + obj.pos
        # Vectorized World -> Camera (Panda convention: X-Right, Z-Up, Y-Fwd)
        # We need Camera Space for projection (X-Right, Y-Down, Z-Fwd)
        # But we can just use the manual logic from previous code or K directly
        # Let's use the manual logic which matches K derivation
        verts_cam_panda = (verts_world - vcam.cam_pos) @ vcam.cam_rotmat
        verts_cv_x = verts_cam_panda[:, 0]
        verts_cv_y = -verts_cam_panda[:, 2]  # Panda Z is CV -Y
        verts_cv_z = verts_cam_panda[:, 1]  # Panda Y is CV Z
        z_safe = verts_cv_z.copy()
        z_safe[z_safe < 0.001] = 0.001

        u_coords = (K[0, 0] * verts_cv_x) / z_safe + K[0, 2]
        v_coords = (K[1, 1] * verts_cv_y) / z_safe + K[1, 2]

        uv_coords = np.stack((u_coords, v_coords), axis=1).astype(np.int32)
        face_polygons = uv_coords[faces]

        # Draw contours on RGB
        mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, face_polygons, 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_rgb, contours, -1, (0, 255, 0), 2)

        # --- G. Final Display ---
        # Add Overlay Text
        info_txt = f"Pos: {new_x:.2f}, {new_y:.2f}, {new_z:.2f}"
        cv2.putText(img_rgb, info_txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
        cv2.putText(depth_colored, "Depth (Jet Colormap)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        img_rgb = img_rgb[:, :, :3]
        combined_img = np.hstack((img_rgb, depth_colored))
        cv2.imshow("Virtual Depth Camera (RGB | Depth)", combined_img)
        key = cv2.waitKey(1) & 0xFF
        return task.cont


    # Add task to Panda3D manager
    base.taskMgr.add(update, "vcam_update_loop")
    print("Running... Press Ctrl+C in console to stop, or close the main window.")
    base.run()
