"""
Examples to use virtual camera in Panda3D World.
Author: Chen Hao (chen960216@gmail.com)
Date: 20251202fuzhou
"""
if __name__ == "__main__":
    import cv2
    import numpy as np
    from wrs import wd, rm, mgm
    from wrs.visualization.panda.panda3d_utils import VirtualCamera

    base = wd.World(cam_pos=[1, 0, 1], lookat_pos=[0, 0, .05])
    obj = mgm.GeometricModel("objects/bunnysim.stl")
    obj.attach_to(base)
    mgm.gen_frame(ax_length=0.2, ax_radius=0.005).attach_to(base)
    resolution = (640, 480)
    vcam = VirtualCamera(cam_pos=[0.4, 0, 0.2],
                         lookat_pos=[0, 0, 0.05],
                         resolution=resolution,
                         screen_size=np.array((0.05, 0.05)),
                         fov=45)
    # Create the visualization of the camera (Frustum)
    # We will update its pose in the loop
    vcam_viz_node = [vcam.gen_meshmodel(alpha=0.5)]
    vcam_viz_node[0].attach_to(base)

    target_point = obj.trm_mesh.center_mass
    def update(task):
        t = task.time
        vcam_viz_node[0].detach()
        # --- A. Orbit Logic ---
        radius = 0.3
        speed = 2
        angle = t * speed
        # Calculate new position (Circular orbit at height 0.2)
        new_x = radius * np.cos(angle)
        new_y = radius * np.sin(angle)
        new_z = 0.2 + 0.1 * np.sin(angle * 0.5)  # Bobbing up and down slightly

        # Update Camera Pose
        vcam.cam_pos = np.array([new_x, new_y, new_z])
        vcam.look_at(target_point)
        # Update Visualization
        vcam_viz_node[0] = vcam.gen_meshmodel(alpha=0.5)
        vcam_viz_node[0].attach_to(base)
        # Get Image and Intrinsic Matrix
        img = vcam.get_image()
        K = vcam.intrinsics
        # Project 3D Point to 2D
        # Let's project the target_point (center of bunny) onto the image
        # Transform world point to camera frame
        # P_cam = R^T * (P_world - t_cam)
        p_world = target_point
        p_cam = vcam.cam_rotmat.T @ (p_world - vcam.cam_pos)
        # Project using Intrinsic Matrix
        # [u, v, 1]^T = K * [x/z, y/z, 1]^T
        if p_cam[2] > 0.001:  # Avoid division by zero and points behind camera
            u = (K[0, 0] * p_cam[0] + K[0, 2] * p_cam[2]) / p_cam[2]
            v = (K[1, 1] * p_cam[1] + K[1, 2] * p_cam[2]) / p_cam[2]
            # Draw a crosshair on the target
            u, v = int(u), int(v)
            cv2.drawMarker(img, (u, v), (0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=2)
            cv2.putText(img, "Target", (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # Generate Mask from Mesh Projection
        verts = obj.trm_mesh.vertices
        faces = obj.trm_mesh.faces  # (M, 3) array
        verts_world = verts @ obj.rotmat.T + obj.pos
        verts_cam_panda = (verts_world - vcam.cam_pos) @ vcam.cam_rotmat
        verts_cv_x = verts_cam_panda[:, 0]
        verts_cv_y = -verts_cam_panda[:, 2]
        verts_cv_z = verts_cam_panda[:, 1]
        # u = fx * x/z + cx, v = fy * y/z + cy
        z_safe = verts_cv_z.copy()
        z_safe[z_safe < 0.001] = 0.001
        u_coords = (K[0, 0] * verts_cv_x) / z_safe + K[0, 2]
        v_coords = (K[1, 1] * verts_cv_y) / z_safe + K[1, 2]
        # Stack into (N, 2) integer coordinates
        uv_coords = np.stack((u_coords, v_coords), axis=1).astype(np.int32)
        # 4. Rasterize Mask
        # Create polygon array from faces: (M, 3, 2)
        face_polygons = uv_coords[faces]
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # fillPoly is very efficient for drawing multiple polygons
        cv2.fillPoly(mask, face_polygons, 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

        # Add Overlay Text
        cv2.putText(img, f"Pos: {new_x:.2f}, {new_y:.2f}, {new_z:.2f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
        # Show using OpenCV
        cv2.imshow("Virtual Camera Feed", img)
        cv2.waitKey(1)  # Required to refresh cv2 window
        return task.cont
    # Add task to Panda3D manager
    base.taskMgr.add(update, "vcam_update_loop")
    print("Running... Press Ctrl+C in console to stop, or close the main window.")
    base.run()
