from wrs import modeling as gm
import wrs.visualization.panda.world as wd
import cv2
import img_to_depth as itd

base = wd.World(cam_pos=[.03, .03, .07], lookat_pos=[0.015, 0.015, 0])
itd_cvter = itd.ImageToDepth()

video1 = cv2.VideoCapture(1)
width = (int(video1.get(cv2.CAP_PROP_FRAME_WIDTH)))
height = (int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
pointcloud = None

pcdm = []
def update(video1, pcdm, task):
    if len(pcdm) != 0:
        for one_pcdm in pcdm:
            one_pcdm.detach()
    else:
        pcdm.append(None)
    key = cv2.waitKey(1)
    if int(key) == 113:
        video1.release()
        return task.done
    ret, frame = video1.read()
    frame = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
    depth, hm = itd_cvter.convert(frame)
    pcdm[0] = gm.GeometricModel(depth*.001)
    pcdm[0].attach_to(base)
    return task.again
taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[video1, pcdm],
                      appendTask=True)
base.run()
