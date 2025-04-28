from wrs import wd, rm, rtqhe, mgm, mcm, gg
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import itertools
import random

class GraspGroup:
    def __init__(self, grasps, color, avg_y_dir):
        self.grasps = grasps
        self.color = color
        self.avg_y_dir = avg_y_dir


base = wd.World(cam_pos=[1.7, -1, 1.2], lookat_pos=[0, 0, .3])
obj_cmodel = mcm.CollisionModel("lshape.stl")
obj_cmodel.rgba = rm.vec(.7, .7, 0, 1)
obj_cmodel.attach_to(base)

gripper = rtqhe.RobotiqHE()
grasp_collection = gg.GraspCollection.load_from_disk(file_name="robotiqhe_grasps.pickle")

# grouping considering grasping directions
y_axes = rm.np.array([grasp.ac_rotmat[:, 1] for grasp in grasp_collection])
y_dirs = rm.np.array([y / rm.np.linalg.norm(y) for y in y_axes])
sim_mat = rm.np.abs(cosine_similarity(y_dirs))
dist_mat = 1 - sim_mat
clustering = DBSCAN(eps=0.05, min_samples=1, metric='precomputed')  # eps is the distance threshold
labels = clustering.fit_predict(dist_mat)
cluster_indices = defaultdict(list)
for i, label in enumerate(labels):
    cluster_indices[label].append(i)
grasp_groups = []
for label, idx_list in cluster_indices.items():
    grasps = [grasp_collection[i] for i in idx_list]
    y_vecs = [rm.np.abs(grasp.ac_rotmat[:, 1]) for grasp in grasps]
    avg_y = rm.np.mean(y_vecs, axis=0)
    avg_y /= rm.np.linalg.norm(avg_y)
    color = plt.get_cmap("tab10")(label % 10)[:3]
    group = GraspGroup(grasps=grasps, color=color, avg_y_dir=avg_y)
    grasp_groups.append(group)
# debug info
for group in grasp_groups:
    print(f"Group with {len(group.grasps)} grasps, avg_y_dir: {group.avg_y_dir}")
# show grasp groups
for group in grasp_groups:
    for grasp in group.grasps:
        gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat,
                                jaw_width=grasp.ee_values)
        gripper.gen_meshmodel(rgb=group.color, alpha=.3).attach_to(base)
base.run()