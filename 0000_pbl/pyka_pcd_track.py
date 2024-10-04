import numpy as np
import math
import wrs.visualization.panda.world as wd
from wrs.vision import load_calibration_data
from wrs import basis as rm, drivers as pk, robot_sim as xsm, modeling as gm
import wrs.robot_con.xarm_shuidi_grpc.xarm_shuidi_client as xsc

affine_matrix, _, _ = load_calibration_data()
base = wd.World(cam_pos=[10, 2, 7], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
pkx = pk.PyKinectAzure()
pcd_list = []
ball_center_list = []
para_list = []
ball_list = []
counter = [0]
# get background
background_image = None
while True:
    pkx.device_get_capture()
    depth_image_handle = pkx.capture_get_depth_image()
    if depth_image_handle:
        background_image = pkx.image_convert_to_numpy(depth_image_handle)
        pkx.image_release(depth_image_handle)
        pkx.capture_release()
        break
rbts = xsm.XArmShuidi()
pos_start, rot_start = (np.array([-0.06960152, 0.01039783, 1.41780278]),
                        np.array([[-1, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, 1]]))
jnts = rbts.ik(component_name="arm", tgt_pos=pos_start, tgt_rotmat=rot_start, max_niter=1000)
rbts.fk(jnt_values=jnts)
rbts.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
rbtx = xsc.XArmShuidiClient(host="10.2.0.201:18300")
rbtx.move_jnts(component_name="arm", jnt_values=jnts)


def update(pkx, rbtx, background_image, pcd_list, ball_center_list, counter, task):
    if len(pcd_list) != 0:
        for pcd in pcd_list:
            pcd.detach()
        pcd_list.clear()
    while True:
        pkx.device_get_capture()
        color_image_handle = pkx.capture_get_color_image()
        depth_image_handle = pkx.capture_get_depth_image()
        if color_image_handle and depth_image_handle:
            break
    point_cloud = pkx.transform_depth_image_to_point_cloud(depth_image_handle)
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
            orbit = []
            for t in np.linspace(ball_center_list[0][1], ball_center_list[0][1] + 15, 100):
                point = np.array([f_x(t), f_y(t), f_z(t)])
                orbit.append(point)
                if abs(point[0]) < .7 and abs(point[1]) < .7 and abs(point[2] - 1.1) < .3:
                    # last_point = orbit[-1]
                    jnt_values = rbts.ik(tgt_pos=point, tgt_rotmat=np.array([[-1, 0, 0],
                                                                             [0, -1, 0],
                                                                             [0, 0, 1]]))
                    if jnt_values is None:
                        continue
                    rbts.fk(component_name="arm", jnt_values=jnt_values)
                    rbts.gen_meshmodel().attach_to(base)
                    rbtx.arm_move_jspace_path([rbtx.get_jnt_values(), jnt_values], max_jntspeed=math.pi * 1.3)
                    break
            for id in range(len(orbit)):
                if id > 0:
                    tmp_stick = gm.gen_stick(spos=orbit[id - 1], epos=orbit[id], radius=.01, type="round")
                    tmp_stick.attach_to(base)
                    para_list.append(tmp_stick)
            return task.done
    pkx.image_release(color_image_handle)
    pkx.image_release(depth_image_handle)
    pkx.capture_release()
    counter[0] += 1
    return task.cont


taskMgr.doMethodLater(0.001, update, "update",
                      extraArgs=[pkx, rbtx, background_image, pcd_list, ball_center_list, counter],
                      appendTask=True)
base.run()
