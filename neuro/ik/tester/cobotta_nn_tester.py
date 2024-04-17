import torch
import math
import time
import numpy as np
import pandas as pd
import neuro.ik.cobotta_fitting as cbf
import basis.robot_math as rm
import modeling.geometric_model as mgm
import visualization.panda.world as world
import robot_sim.robots.cobotta.cobotta as cbt_s

model = cbf.Net(n_hidden=1000, n_jnts=6)
model.load_state_dict(torch.load("cobotta_model.pth", map_location=torch.device('cpu')))
model.eval()

print(model.parameters())

min_max = np.load(file="../data_gen/cobotta_ik_min_max.npy")

base = world.World(cam_pos=np.array([1.5, 1, .1]))
cobotta = cbt_s.Cobotta()

jnt_values = cobotta.rand_conf()
tgt_pos, tgt_rotmat = cobotta.goto_given_conf(jnt_values=jnt_values)
rpy = rm.rotmat_to_euler(tgt_rotmat)
xyzrpy = (tgt_pos[0], tgt_pos[1], tgt_pos[2], rpy[0], rpy[1], rpy[2])
normalized_xyzrpy = (xyzrpy - min_max[0]) / (min_max[1] - min_max[0])

print(torch.tensor(normalized_xyzrpy))

jnt_values = model(torch.tensor(normalized_xyzrpy, dtype=torch.float32))
cobotta.goto_given_conf(jnt_values=jnt_values)
cobotta.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)

mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

base.run()
