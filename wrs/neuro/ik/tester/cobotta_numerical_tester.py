import torch
import math
import numpy as np
import wrs.neuro.ik.cobotta_fitting as cbf
from wrs import basis as rm, robot_sim as cbt_s, modeling as gm
import wrs.visualization.panda.world as world

if __name__ == '__main__':
    base = world.World(cam_pos=np.array([1.5, 1, .7]))
    gm.gen_frame().attach_to(base)
    rbt_s = cbt_s.Cobotta()
    rbt_s.fk(jnt_values=np.zeros(6))
    rbt_s.gen_meshmodel(toggle_tcp_frame=True, rgba=[.5, .5, .5, .3]).attach_to(base)
    rbt_s.gen_stickmodel(toggle_tcp_frame=True).attach_to(base)
    tgt_pos = np.array([.25, .2, .15])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

    # numerical ik
    jnt_values = rbt_s.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    rbt_s.fk(jnt_values=jnt_values)
    rbt_s.gen_meshmodel(toggle_tcp_frame=True, rgba=[.5, .5, .5, .3]).attach_to(base)

    # neural ik
    model = cbf.Net(n_hidden=100, n_jnts=6)
    model.load_state_dict(torch.load("cobotta_model.pth"))
    tgt_rpy = rm.rotmat_to_euler(tgt_rotmat)
    xyzrpy = torch.from_numpy(np.hstack((tgt_pos,tgt_rpy)))
    jnt_values = model(xyzrpy.float()).to('cpu').detach().numpy()
    rbt_s.fk(jnt_values=jnt_values)
    rbt_s.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)

    base.run()
