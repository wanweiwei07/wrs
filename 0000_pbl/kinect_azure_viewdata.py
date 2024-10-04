import wrs.visualization.panda.world as wd
from wrs import modeling as gm
import pickle

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