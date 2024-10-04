import torch
import numpy as np
import wrs.neuro.ik.cobotta_fitting as cbf
from wrs import basis as rm, robot_sim as cbt_s, modeling as mgm
import wrs.visualization.panda.world as world
import torch.nn as nn

model = cbf.Net(n_hidden=128, n_jnts=6)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

base = world.World(cam_pos=np.array([1.5, 1, .1]))
cobotta = cbt_s.Cobotta()

random_jv = cobotta.rand_conf()
tgt_pos, tgt_rotmat = cobotta.goto_given_conf(jnt_values=random_jv)
wvec = rm.rotmat_to_wvec(tgt_rotmat)
xyzwvec = (tgt_pos[0], tgt_pos[1], tgt_pos[2], wvec[0], wvec[1], wvec[2])

jnt_values = model(torch.tensor(xyzwvec, dtype=torch.float32))
result_jv = jnt_values.data.numpy()

print(xyzwvec)
pred_pos, pred_rotmat = cobotta.goto_given_conf(jnt_values=jnt_values)
result_xyzwvec =np.hstack((pred_pos, rm.rotmat_to_wvec(pred_rotmat)))
print(result_xyzwvec)
print("tcp difference ", (result_xyzwvec-xyzwvec).T@(result_xyzwvec-xyzwvec), np.mean((result_xyzwvec-xyzwvec)**2))
print("jnt difference ", random_jv, result_jv, (result_jv-random_jv).T@(result_jv-random_jv), np.mean((result_jv-random_jv)**2))
cobotta.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)

mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
mgm.gen_frame(pos=pred_pos, rotmat=pred_rotmat).attach_to(base)

loss_fn = nn.MSELoss()
loss = loss_fn(jnt_values, torch.tensor(random_jv, dtype=torch.float32))
print("loss ", loss.item())
base.run()
