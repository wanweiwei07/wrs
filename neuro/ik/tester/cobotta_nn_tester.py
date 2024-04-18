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

model = cbf.Net(n_hidden=128, n_jnts=6)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

base = world.World(cam_pos=np.array([1.5, 1, .1]))
cobotta = cbt_s.Cobotta()

jnt_values = cobotta.rand_conf()
tgt_pos, tgt_rotmat = cobotta.goto_given_conf(jnt_values=jnt_values)
wvec = rm.rotmat_to_wvec(tgt_rotmat)
xyzwvec = (tgt_pos[0], tgt_pos[1], tgt_pos[2], wvec[0], wvec[1], wvec[2])

print(torch.tensor(xyzwvec))

jnt_values = model(torch.tensor(xyzwvec, dtype=torch.float32))
cobotta.goto_given_conf(jnt_values=jnt_values)
cobotta.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)

mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

base.run()
