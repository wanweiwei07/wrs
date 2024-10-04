import wrs.visualization.panda.world as wd
import pickle
from wrs import modeling as cm

base = wd.World(cam_pos=[0, -2, -5], lookat_pos=[0, -1, 5])
pcd_list = pickle.load(open("pcdlist.pkl", "rb"))

attached_list = []
counter = [0]
def update(attached_list, pcd_list, counter, task):
    if counter[0] >= len(pcd_list):
        counter[0] = 0
    if len(attached_list) != 0:
        for objcm in attached_list:
            objcm.detach()
        attached_list.clear()
    pcd = pcd_list[counter[0]]
    attached_list.append(cm.CollisionModel(pcd))
    attached_list[-1].attach_to(base)
    counter[0]+=1
    return task.again

taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[attached_list, pcd_list, counter],
                      appendTask=True)
base.run()