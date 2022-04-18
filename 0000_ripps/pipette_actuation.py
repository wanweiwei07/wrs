import os
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.cobotta.cobotta_ripps as cbtr
import robot_sim.end_effectors.gripper.cobotta_pipette.cobotta_pipette as cbtg
import modeling.collision_model as cm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import utils

if __name__ == '__main__':
    base = wd.World(cam_pos=[.25, -1, .4], lookat_pos=[.25, 0, .3])
    gm.gen_frame().attach_to(base)

    rbt_s = cbtr.CobottaRIPPS()
    # rbt_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    rbt_s.jaw_to(jaw_width=0.03)

    tgt_pos = np.array([.25, .0, .1])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], np.pi)
    jnt_values = rbt_s.ik(component_name="arm", tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    print(jnt_values)
    if jnt_values is not None:
        rbt_s.fk(component_name="arm", jnt_values=jnt_values)
        rbt_s.gen_meshmodel().attach_to(base)

    base.run()