import time
import math
import numpy as np
from wrs import basis as rm, robot_sim as jlc, modeling as gm
import wrs.visualization.panda.world as wd

if __name__ == "__main__":
    base = wd.World(cam_pos=[2, 0, 2], lookat_pos=[0, 0, 0])
    gm.gen_frame(axis_length=.2).attach_to(base)

    jlinstance = jlc.JLChain(home_conf=np.array([0, 0, 0, 0, 0]))
    jlinstance.jnts[4]['end_type'] = 'prismatic'
    jlinstance.jnts[4]['loc_motionax'] = np.array([1, 0, 0])
    jlinstance.jnts[4]['motion_value'] = .2
    jlinstance.jnts[4]['motion_range'] = [-.5, .5]
    jlinstance.finalize()
    jlinstance.gen_stickmodel(toggle_jntscs=True, rgba=[1, 0, 0, .15]).attach_to(base)
    tgt_pos0 = np.array([.3, .1, 0])
    tgt_rotmat0 = np.eye(3)
    gm.gen_myc_frame(pos=tgt_pos0, rotmat=tgt_rotmat0, axis_length=.15, axis_radius=.01).attach_to(base)
    jlinstance.set_flange(loc_flange_pos=np.array([.2, -.13, 0]),
                          loc_flange_rotmat=rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi / 8))
    tic = time.time()
    jnt_values = jlinstance.ik(tgt_pos0,
                               tgt_rotmat0,
                               seed_jnt_values=None,
                               local_minima="accept",
                               toggle_debug=False)
    toc = time.time()
    print('ik cost: ', toc - tic, jnt_values)
    jlinstance.fk(joint_values=jnt_values)
    jlinstance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    base.run()
