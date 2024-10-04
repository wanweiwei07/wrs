import wrs.visualization.panda.world as wd
import math
import numpy as np
from wrs import basis as rm, robot_sim as cbt, modeling as gm, modeling as cm
import gcn_ik
import torch
import itertools

base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
gm.gen_frame().attach_to(base)
# ground
ground = cm.gen_box(xyz_lengths=[5, 5, 1], rgba=[.7, .7, .7, .7])
ground.set_pos(np.array([0, 0, -.51]))
ground.attach_to(base)

model = gcn_ik.GCN(6, 32, 6)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

a = [0, 1, 2, 3, 4, 5]
p = itertools.product(a, a)
edge_index = torch.tensor(np.array(list(p)).T.tolist(), dtype=torch.long)
print(edge_index)

robot_s = cbt.Cobotta()
robot_s.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
seed_jnt_values = None
for z in np.linspace(.1, .6, 5):
    goal_pos = np.array([.25, -.1, z])
    goal_rot = rm.rotmat_from_axangle(np.array([0, 1, 0]), math.pi * 1 / 2)
    gm.gen_frame(goal_pos, goal_rot).attach_to(base)

    jnt_values = robot_s.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rot, seed_jnt_values=seed_jnt_values)
    # print(jnt_values)
    if jnt_values is not None:
        robot_s.fk(jnt_values=jnt_values)
        seed_jnt_values = jnt_values
    robot_s.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    tcp_input = torch.Tensor(goal_pos.tolist() + rm.rotmat_to_euler(goal_rot).tolist())
    print(tcp_input.shape)
    print(tcp_input)

    jnt_values = model(tcp_input, edge_index=edge_index)
    print(jnt_values)
    robot_s.fk(jnt_values=jnt_values)
    seed_jnt_values = jnt_values
    robot_s.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)

base.run()