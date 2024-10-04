"""
Example for utilize the Zivid driver
Author: Hao Chen <chen960216@gmail.com>, 20230213, osaka
"""
import wrs.visualization.panda.world as wd
from wrs import modeling as gm
from zivid_sdk import Zivid

if __name__ == "__main__":
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, 0])
    cam = Zivid()

    # Get point cloud
    raw_pcd, pcd_no_nan_indices, rgba = cam.get_pcd_rgba()

    # Remove nan point from raw mph data
    pcd = raw_pcd[pcd_no_nan_indices]
    # change to panda3d color
    pcd_rgb = rgba.reshape(-1, 4)[pcd_no_nan_indices] / 255
    gm.gen_pointcloud(pcd, rgbas=pcd_rgb).attach_to(base)

    base.run()
