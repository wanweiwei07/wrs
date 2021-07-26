import cv2
import drivers.devices.kinect_azure.pykinectazure as pk
import drivers.devices.kinect_azure.helper as pkhlpr
import visualization.panda.world as wd
import modeling.geometric_model as gm
import pickle
import vision.rgb_camera.util_functions as rgbu
import cv2.aruco as aruco
import basis.robot_math as rm

pcd_list = pickle.load(open("pcd_list.pkl", "rb"))
base = wd.World(cam_pos=[0, 0, -10], lookat_pos=[0, 0, 10])
gm.gen_frame().attach_to(base)


shown_pcd_list = []
def update(pcd_list, task):
    if len(pcd_list)==0:
        return task.again
    if len(shown_pcd_list) != 0:
        for pcd in shown_pcd_list:
            pcd.detach()
        shown_pcd_list.clear()
    pcd = gm.GeometricModel(pcd_list[0])
    pcd.attach_to(base)
    shown_pcd_list.append(pcd)
    pcd_list.pop(0)
    return task.again

taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[pcd_list],
                      appendTask=True)
base.run()