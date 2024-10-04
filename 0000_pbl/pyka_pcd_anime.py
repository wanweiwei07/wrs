import numpy as np

import wrs.visualization.panda.world as wd
from wrs.vision import load_calibration_data
from wrs import basis as rm, drivers as pk, modeling as gm

affine_matrix, _, _ = load_calibration_data()

base = wd.World(cam_pos=[10, 2, 7], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
pkx = pk.PyKinectAzure()
pcd_list = []
ball_center_list = []
para_list = []
ball_list = []
counter = [0]

def update(pk_obj, pcd_list, ball_center_list, counter, task):
    if len(pcd_list) != 0:
        for pcd in pcd_list:
            pcd.detach()
        pcd_list.clear()
    # if len(ball_center_list) != 0:
    #     for ball_center in ball_center_list:
    #         ball_center.detach()
    #     ball_center_list.clear()
    while True:
        pk_obj.device_get_capture()
        color_image_handle = pk_obj.capture_get_color_image()
        depth_image_handle = pk_obj.capture_get_depth_image()
        if color_image_handle and depth_image_handle:
            break
    point_cloud = pk_obj.transform_depth_image_to_point_cloud(depth_image_handle)
    point_cloud = rm.transform_points_by_homomat(affine_matrix, point_cloud)
    ball = []
    for id, point_cloud_sub in enumerate(point_cloud):
        if 0.3 < point_cloud_sub[0] < 3.3 and -1.3 < point_cloud_sub[1] < .3 and 0.5 < point_cloud_sub[2] < 2.5:
            ball.append(point_cloud_sub)

    mypoint_cloud = gm.GeometricModel(initor=point_cloud)
    mypoint_cloud.attach_to(base)
    pcd_list.append(mypoint_cloud)

    if len(ball) > 20:
        center = np.mean(np.array(ball), axis=0)
        ball_center_list.append([center, counter[0]])
        ball = gm.gen_sphere(center, radius=.1)
        ball.attach_to(base)
        ball_list.append(ball)
        if len(ball_center_list) > 2:
            x = []
            y = []
            z = []
            t = []
            for cen, f_id in ball_center_list:
                x.append(cen[0])
                y.append(cen[1])
                z.append(cen[2])
                t.append(f_id)
            para_x = np.polyfit(t, x, 1)
            para_y = np.polyfit(t, y, 1)
            para_z = np.polyfit(t, z, 2)
            f_x = np.poly1d(para_x)
            f_y = np.poly1d(para_y)
            f_z = np.poly1d(para_z)
            orbit=[]
            for t in np.linspace(ball_center_list[0][1],ball_center_list[0][1]+5,100):
                orbit.append(np.array([f_x(t), f_y(t), f_z(t)]))
            for id in range(len(orbit)):
                if id > 0:
                    tmp_stick = gm.gen_stick(spos=orbit[id - 1], epos=orbit[id], radius=.01, type="round")
                    tmp_stick.attach_to(base)
                    para_list.append(tmp_stick)
            return task.done
    pk_obj.image_release(color_image_handle)
    pk_obj.image_release(depth_image_handle)
    pk_obj.capture_release()
    counter[0] += 1
    return task.cont


taskMgr.doMethodLater(0.001, update, "update",
                      extraArgs=[pkx, pcd_list, ball_center_list, counter],
                      appendTask=True)
base.run()
