import wrs.robot_sim.end_effectors.single_contact.suction.mvfln40.mvfln40 as suction
import numpy as np
import wrs.visualization.panda.world as wd
from wrs import basis as rm, robot_sim as ur3d, modeling as gm, modeling as cm
import math

base = wd.World(cam_pos=[3.5, 1, 2.5], lookat_pos=[.5,0,1.2])
ur3d_s = ur3d.UR3Dual()
ur3d_s.gen_meshmodel().attach_to(base)

ground = gm.gen_box(np.array([2, 2, .001]))
ground.set_rgba([1, 253 / 255, 219 / 255, 1])
ground.attach_to(base)

box = cm.gen_box(np.array([.3, .3, .04]))
box.set_rgba([153 / 255, 183 / 255, 1, 1])
box.set_pos(np.array([.3, 0, 1.126]))
box.attach_to(base)

# box = mcm.gen_box(np.array([.3, .3, .04]))
# box.set_rgba([153 / 255, 183 / 255, 1, 1])
# box.set_pos(np.array([.35, 0, 1.126+0.05]))
# box.set_rotmat(rm.rotmat_from_axangle([0,0,1], math.pi/8))
# box.attach_to(base)
#
# support_box = mcm.gen_box(np.array([.5, .3, .05]))
# support_box.set_rgba([133 / 255, 94 / 255, 66/255, 1])
# support_box.set_pos(np.array([.35, 0, 1.126]))
# support_box.set_rotmat(rm.rotmat_from_axangle([0,0,1], math.pi/8))
# support_box.attach_to(base)

suction_s = suction.MVFLN40()
loc_pos_box = np.array([-.1, 0, .02])
# loc_pos_box = np.array([-.1, 0, .02])
# loc_pos_box = np.array([0, -.1, .02])
loc_rotmat_box = rm.rotmat_from_euler(math.pi, 0, 0)
gl_pos_box = box.get_pos() + box.get_rotmat().dot(loc_pos_box)
gl_rotmat_box = box.get_rotmat().dot(loc_rotmat_box)
suction_s.attach_to_with_cpose(gl_pos_box, gl_rotmat_box)
suction_s.gen_meshmodel().attach_to(base)
gm.gen_stick(
    suction_s.pos,
    np.array([0,0,2.4])).attach_to(base)


ar = np.array([[0.94, -0.34, 0], [0.34, 0.94, 0], [0, 0, 1]])
print(rm.axangle_between_rotmat(ar, np.eye(3)))
print(np.degrees(0.347))
base.run()