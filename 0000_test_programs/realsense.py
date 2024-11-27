import wrs.drivers.devices.realsense.realsense_d400s as rs
from wrs import wd, rm, mgm

base = wd.World(cam_pos=rm.vec(2, 1, 1), lookat_pos=rm.vec(0, 0, 0))
mgm.gen_frame().attach_to(base)

d405 = rs.RealSenseD400()
onscreen = []

def update(d405, onscreen, task):
    if len(onscreen) > 0:
        for ele in onscreen:
            ele.detach()
    pcd, pcd_color = d405.get_pcd(return_color=True)
    onscreen.append(mgm.gen_pointcloud(pcd, pcd_color))
    onscreen[-1].attach_to(base)
    return task.cont

base.taskMgr.add(update, "update", extraArgs=[d405, onscreen], appendTask=True)
base.run()