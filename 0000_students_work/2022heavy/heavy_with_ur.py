import wrs.robot_sim.end_effectors.single_contact.suction.mvfln40.mvfln40 as suction
import wrs.robot_sim.end_effectors.grippers.robotiq85_gelsight.robotiq85_gelsight_pusher as rtqgel
import numpy as np
import wrs.visualization.panda.world as wd
from wrs import basis as rm, robot_sim as rtq, robot_sim as ur3d, modeling as gm, modeling as cm
import math

base = wd.World(cam_pos=[3.5, 1, 2.5], lookat_pos=[.5,0,1.2])
ur3d_s = ur3d.UR3Dual()
# ur3d_s.gen_meshmodel().attach_to(base)

ground = gm.gen_box(np.array([2, 2, .001]))
ground.set_rgba([1, 253 / 255, 219 / 255, 1])
ground.attach_to(base)

box_0 = cm.gen_box(np.array([.3, .3, .04]))
box_0.set_rgba([153 / 255, 183 / 255, 1, .3])
box_0.set_pos(np.array([.4, -.1, 1.126]))
box_0.set_rotmat(rm.rotmat_from_axangle([0, 0, 1], -math.pi / 6))
box_0.attach_to(base)

# suction_s = suction.MVFLN40()
# loc_pos_box = np.array([.1, 0, .02])
# loc_rotmat_box = rm.rotmat_from_euler(math.pi, 0, 0)
# gl_pos_box = box_0.get_pos() + box_0.get_rotmat().dot(loc_pos_box)
# gl_rotmat_box = box_0.get_rotmat().dot(loc_rotmat_box)
# suction_s.suction_to_with_scpose(gl_pos_box, gl_rotmat_box)
# suction_s.gen_meshmodel(rgba=[.55, .55, .55, .3]).attach_to(base)
# mgm.gen_stick(
#     suction_s.pos,
#     np.array([0,0,1]),rgba=[1,0,0,.3]).attach_to(base)

box = cm.gen_box(np.array([.3, .3, .04]))
box.set_rgba([153 / 255, 183 / 255, 1, 1])
box.set_pos(np.array([.4, -.1, 1.126]))
box.set_rotmat(rm.rotmat_from_axangle([0, 0, 1], -math.pi / 6))
current_rotmat = box.get_rotmat()
box.set_rotmat(rm.rotmat_from_axangle(current_rotmat[:,1], -math.pi/6).dot(current_rotmat))
box.set_pos(box.get_pos()+np.array([0,0,.07]-current_rotmat[:,0]*.03))
box.attach_to(base)

suction_s = suction.MVFLN40()
loc_pos_box = np.array([.1, 0, .02])
loc_rotmat_box = rm.rotmat_from_euler(math.pi, 0, 0)
gl_pos_box = box.get_pos() + box.get_rotmat().dot(loc_pos_box)
gl_rotmat_box = box.get_rotmat().dot(loc_rotmat_box)
suction_s.attach_to_with_cpose(gl_pos_box, gl_rotmat_box)
suction_s.gen_meshmodel().attach_to(base)
# mgm.gen_stick(
#     suction_s.pos,
#     suction_s.pos - suction_s.rotmat[:,2]*10).attach_to(base)
gm.gen_stick(
    suction_s.pos,
    np.array([0,0,2.4])).attach_to(base)


loc_pos_box = np.array([-.125, .03, .02])
loc_rotmat_box = rm.rotmat_from_axangle([0,1,0], math.pi*11/12).dot(rm.rotmat_from_axangle([0,0,1], math.pi/2))
loc_rotmat_box=rm.rotmat_from_axangle([1,0,0], -math.pi/6).dot(loc_rotmat_box)
# loc_pos_box = np.array([-.12, .03, .02])
# loc_rotmat_box = rm.rotmat_from_euler(math.pi*5/7, -math.pi / 3, math.pi/3)
# loc_pos_box = np.array([-.12, .03, -.02])
# loc_rotmat_box = rm.rotmat_from_euler(0, -math.pi / 3, math.pi/3)
# loc_pos_box = np.array([0, -.12, -.02])
# loc_rotmat_box = rm.rotmat_from_euler(0, math.pi / 3, math.pi/3)
gl_pos_box = box.get_pos() + box.get_rotmat().dot(loc_pos_box)
gl_rotmat_box = box.get_rotmat().dot(loc_rotmat_box)
jnt_angles = ur3d_s.ik(component_name='lft_arm', tgt_pos=gl_pos_box, tgt_rotmat=gl_rotmat_box)
if jnt_angles is not None:
    ur3d_s.fk(component_name='lft_arm', jnt_values=jnt_angles)
    # ur3d_s.gen_meshmodel().attach_to(base)
print(gl_pos_box)

rtqgel_s = rtqgel.Robotiq85GelsightPusher()
print(rtqgel_s.jaw_center_pos)
rtqgel_s.grip_at_by_pose(gl_pos_box, gl_rotmat_box, jaw_width=rtqgel_s.jaw_range[1])
rtqgel_s.gen_meshmodel(toggle_tcp_frame=False).attach_to(base)

rtq_s = rtq.Robotiq85()
# loc_pos_box = np.array([-.11, .12, .0])
# loc_rotmat_box = rm.rotmat_from_euler(-math.pi/2,0,-math.pi*4/5)
loc_pos_box = np.array([.1, -.12, .0])
loc_rotmat_box = rm.rotmat_from_euler(-math.pi/2,0,-math.pi/12)
gl_pos_box = box.get_pos() + box.get_rotmat().dot(loc_pos_box)
gl_rotmat_box = box.get_rotmat().dot(loc_rotmat_box)
rtq_s.grip_at_by_pose(gl_pos_box, gl_rotmat_box, jaw_width=.04)
# rtq_s.gen_meshmodel(toggle_flange_frame=False).attach_to(base)
# jnt_angles = ur3d_s.ik(component_name='rgt_arm', tgt_pos=gl_pos_box, tgt_rotmat=gl_rotmat_box)
# if jnt_angles is not None:
#     ur3d_s.fk(component_name='rgt_arm', jnt_values=jnt_angles)
ur3d_s.gen_meshmodel().attach_to(base)
base.run()
