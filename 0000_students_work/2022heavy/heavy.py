import robot_sim.end_effectors.suction.mvfln40.mvfln40 as suction
import robot_sim.end_effectors.gripper.robotiq85_gelsight.robotiq85_gelsight_pusher as rtqgel
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq
import modeling.geometric_model as gm
import modeling.collision_model as cm
import numpy as np
import visualization.panda.world as wd
import basis.robot_math as rm
import math

base = wd.World(cam_pos=[1.5, 1, 1], lookat_pos=[-.5, -.5, 0])

ground = gm.gen_box(np.array([2, 2, .001]))
ground.set_rgba([1, 253 / 255, 219 / 255, 1])
ground.attach_to(base)

# box_0 = cm.gen_box(np.array([.3, .3, .04]))
# box_0.set_rgba([153 / 255, 183 / 255, 1, .3])
# box_0.set_pos(np.array([0, 0, .02]))
# box_0.attach_to(base)

# suction_s = suction.MVFLN40()
# loc_pos_box = np.array([.1, 0, .02])
# loc_rotmat_box = rm.rotmat_from_euler(math.pi, 0, 0)
# gl_pos_box = box_0.get_pos() + box_0.get_rotmat().dot(loc_pos_box)
# gl_rotmat_box = box_0.get_rotmat().dot(loc_rotmat_box)
# suction_s.suction_to_with_scpose(gl_pos_box, gl_rotmat_box)
# suction_s.gen_meshmodel(rgba=[.55, .55, .55, .3]).attach_to(base)
# gm.gen_stick(
#     suction_s.pos,
#     np.array([0,0,1]),rgba=[1,0,0,.3]).attach_to(base)

box = cm.gen_box(np.array([.3, .3, .04]))
box.set_rgba([153 / 255, 183 / 255, 1, 1])
box.set_pos(np.array([0, 0, .02]))
# box.set_rotmat(rm.rotmat_from_axangle([0, 1, 0], -math.pi / 12))
box.attach_to(base)

# box = cm.gen_box(np.array([.3, .3, .04]))
# box.set_rgba([153 / 255, 183 / 255, 1, 1])
# box.set_pos(np.array([0, 0, .32]))
# box.set_rotmat(rm.rotmat_from_axangle([0, 1, 0], -math.pi / 12))
# box.attach_to(base)

suction_s = suction.MVFLN40()
loc_pos_box = np.array([.1, 0, .02])
loc_rotmat_box = rm.rotmat_from_euler(math.pi, 0, 0)
gl_pos_box = box.get_pos() + box.get_rotmat().dot(loc_pos_box)
gl_rotmat_box = box.get_rotmat().dot(loc_rotmat_box)
suction_s.suction_to_with_scpose(gl_pos_box, gl_rotmat_box)
suction_s.gen_meshmodel().attach_to(base)
# gm.gen_stick(
#     suction_s.pos,
#     suction_s.pos - suction_s.rotmat[:,2]*10).attach_to(base)
gm.gen_stick(
    suction_s.pos,
    np.array([0,0,1])).attach_to(base)


# loc_pos_box = np.array([-.12, .12, .02])
# loc_rotmat_box = rm.rotmat_from_euler(math.pi*2/3, -math.pi/6, 0)
loc_pos_box = np.array([-.12, .03, .02])
loc_rotmat_box = rm.rotmat_from_euler(math.pi*5/7, -math.pi / 3, math.pi/3)
# loc_pos_box = np.array([-.12, .03, -.02])
# loc_rotmat_box = rm.rotmat_from_euler(0, -math.pi / 3, math.pi/3)
# loc_pos_box = np.array([0, -.12, -.02])
# loc_rotmat_box = rm.rotmat_from_euler(0, math.pi / 3, math.pi/3)
gl_pos_box = box.get_pos() + box.get_rotmat().dot(loc_pos_box)
gl_rotmat_box = box.get_rotmat().dot(loc_rotmat_box)
print(gl_pos_box)

rtqgel_s = rtqgel.Robotiq85GelsightPusher()
print(rtqgel_s.jaw_center_pos)
rtqgel_s.grip_at_with_jcpose(gl_pos_box, gl_rotmat_box, jaw_width=rtqgel_s.jawwidth_rng[1])
rtqgel_s.gen_meshmodel(toggle_tcpcs=False).attach_to(base)

rtq_s = rtq.Robotiq85()
# loc_pos_box = np.array([-.11, .12, .0])
# loc_rotmat_box = rm.rotmat_from_euler(-math.pi/2,0,-math.pi*4/5)
loc_pos_box = np.array([.11, -.12, .0])
loc_rotmat_box = rm.rotmat_from_euler(-math.pi/2,0,-math.pi/6)
gl_pos_box = box.get_pos() + box.get_rotmat().dot(loc_pos_box)
gl_rotmat_box = box.get_rotmat().dot(loc_rotmat_box)
rtq_s.grip_at_with_jcpose(gl_pos_box, gl_rotmat_box, jaw_width=.04)
# rtq_s.gen_meshmodel(toggle_tcpcs=False).attach_to(base)
base.run()
