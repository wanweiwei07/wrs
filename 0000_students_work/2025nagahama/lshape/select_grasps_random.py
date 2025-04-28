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

# best score
combo_scores = []
for i1, i2, i3 in itertools.permutations(range(len(grasp_groups)), 3):
    y1 = grasp_groups[i1].avg_y_dir
    if rm.np.dot(y1, rm.const.y_ax) < 0.5:
        continue
    y2 = grasp_groups[i2].avg_y_dir
    y3 = grasp_groups[i3].avg_y_dir
    score = abs(rm.np.dot(y1, y2)) + abs(rm.np.dot(y2, y3)) + abs(rm.np.dot(y3, y1))
    combo_scores.append((score, (i1, i2, i3)))
    print(i1, i2, i3, score)
    break
combo_scores.sort(key=lambda x: x[0])
best_score, best_triplet = combo_scores[0]


gripper1 = rtqhe.RobotiqHE()
gripper2 = rtqhe.RobotiqHE()
gripper3 = rtqhe.RobotiqHE()
target_dir_1 = rm.np.array([0.5, 0, -0.5])
target_dir_2 = rm.np.array([0,1,0])
target_dir_3 = rm.np.array([0,-1,0])
i1, i2, i3 = best_triplet
while True:
    grasp1 = random.choice(grasp_groups[i1].grasps)
    z1 = grasp1.ac_rotmat[:, 2]
    if rm.np.dot(z1, target_dir_1) < 0.5:
        continue
    grasp2 = random.choice(grasp_groups[i2].grasps)
    z2 = grasp2.ac_rotmat[:, 2]
    if rm.np.dot(z2, target_dir_2) < 0.5:
        continue
    grasp3 = random.choice(grasp_groups[i3].grasps)
    z3 = grasp3.ac_rotmat[:, 2]
    if rm.np.dot(z3, target_dir_3) < 0.5:
        continue
    # check collision
    gripper1.grip_at_by_pose(jaw_center_pos=grasp1.ac_pos, jaw_center_rotmat=grasp1.ac_rotmat,
                            jaw_width=grasp1.ee_values)
    gripper2.grip_at_by_pose(jaw_center_pos=grasp2.ac_pos, jaw_center_rotmat=grasp2.ac_rotmat,
                            jaw_width=grasp2.ee_values)
    gripper3.grip_at_by_pose(jaw_center_pos=grasp3.ac_pos, jaw_center_rotmat=grasp3.ac_rotmat,
                            jaw_width=grasp3.ee_values)
    if gripper1.is_mesh_collided(gripper2.cdmesh_list) or gripper2.is_mesh_collided(gripper3.cdmesh_list):
        continue
    # show grasp1
    gripper1.gen_meshmodel(rgb=grasp_groups[i1].color, alpha=0.5).attach_to(base)
    mgm.gen_arrow(spos=grasp1.ac_pos, epos=target_dir_1, rgb=grasp_groups[i1].color).attach_to(base)
    # show grasp2
    gripper2.gen_meshmodel(rgb=grasp_groups[i2].color, alpha=0.5).attach_to(base)
    mgm.gen_arrow(spos=grasp2.ac_pos, epos=target_dir_2, rgb=grasp_groups[i2].color).attach_to(base)
    # show grasp3
    gripper3.gen_meshmodel(rgb=grasp_groups[i3].color, alpha=0.5).attach_to(base)
    mgm.gen_arrow(spos=grasp3.ac_pos, epos=target_dir_3, rgb=grasp_groups[i3].color).attach_to(base)
    # planes
    mgm.gen_box(xyz_lengths=rm.vec(.3, 0.001, .3), pos=grasp1.ac_pos, rotmat=grasp1.ac_rotmat,
                rgb=grasp_groups[i1].color, alpha=.3).attach_to(base)
    mgm.gen_box(xyz_lengths=rm.vec(.3, 0.001, .3), pos=grasp2.ac_pos, rotmat=grasp2.ac_rotmat,
            rgb=grasp_groups[i2].color, alpha=.3).attach_to(base)
    mgm.gen_box(xyz_lengths=rm.vec(.3, 0.001, .3), pos=grasp3.ac_pos, rotmat=grasp3.ac_rotmat,
                rgb=grasp_groups[i3].color, alpha=.3).attach_to(base)
    base.run()
