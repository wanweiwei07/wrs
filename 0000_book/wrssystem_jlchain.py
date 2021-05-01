import time
import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import robot_sim._kinematics.jlchain as jlc
import modeling.geometric_model as gm

if __name__ == "__main__":
    base = wd.World(cam_pos=[2, 0, 2], lookat_pos=[0, 0, 0])
    gm.gen_frame(length=.2).attach_to(base)

    jlinstance = jlc.JLChain(homeconf=np.array([0, 0, 0, 0, 0]))
    jlinstance.jnts[4]['type'] = 'prismatic'
    jlinstance.jnts[4]['loc_motionax'] = np.array([1, 0, 0])
    jlinstance.jnts[4]['motion_val'] = .2
    jlinstance.jnts[4]['motion_rng'] = [-.5,.5]
    jlinstance.reinitialize()
    jlinstance.gen_stickmodel(toggle_jntscs=True, rgba=[1, 0, 0, .15]).attach_to(base)
    tgt_pos0 = np.array([.3, .1, 0])
    tgt_rotmat0 = np.eye(3)
    gm.gen_mycframe(pos=tgt_pos0, rotmat=tgt_rotmat0, length=.15, thickness=.01).attach_to(base)
    jlinstance.set_tcp(tcp_jntid=4, tcp_loc_pos=np.array([.2, -.13, 0]), tcp_loc_rotmat=rm.rotmat_from_axangle(np.array([0,0,1]), math.pi/8))
    tic = time.time()
    jnt_values = jlinstance.ik(tgt_pos0,
                               tgt_rotmat0,
                               seed_jnt_values=None,
                               local_minima="accept",
                               toggle_debug=False)
    toc = time.time()
    print('ik cost: ', toc - tic, jnt_values)
    jlinstance.fk(jnt_values=jnt_values)
    jlinstance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    base.run()
