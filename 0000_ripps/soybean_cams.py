import numpy as np
from wrs import basis as rm, robot_sim as rbt, modeling as cm
import wrs.visualization.panda.world as wd

aluminium_rgba = [132 / 255, 135 / 255, 137 / 255, 1]
matt_red = [162 / 255, 32 / 255, 65 / 255, 1]
matt_blue = [30 / 255, 80 / 255, 162 / 255, 1]
matt_black = [44 / 255, 44 / 255, 44 / 255, 1]
reflective_black = [74 / 255, 74 / 255, 74 / 255, 1]

base = wd.World(cam_pos=[1.5, 1.5, .75], auto_cam_rotate=False)

rbt_s = rbt.XArm7()
rbt_s.fk(jnt_values=[0, -np.pi/2, 0, np.pi/12, 0, np.pi/6, 0])
rbt_model = rbt_s.gen_meshmodel(toggle_tcp_frame=False)
rbt_model.attach_to(base)
# base.run()

pos, rotmat = rbt_s.get_gl_tcp()
r_rotmat = rotmat.dot(rm.rotmat_from_axangle([0, 1, 0], -np.pi / 2))
# pos = np.zeros(3)
# rotmat = np.eye(3)

cam_frame = cm.CollisionModel(initor="objects/camera_frame.stl")
cam_frame.set_rgba(rgba=aluminium_rgba)
cam_frame.set_pose(pos, r_rotmat)
cam_frame.attach_to(base)

cam_0 = cm.CollisionModel(initor="objects/flircam.stl")
cam_1 = cam_0.copy()
cam_2 = cam_0.copy()
phoxi = cm.CollisionModel(initor="objects/phoxi_m.stl")

cam_0.set_rgba(rgba=matt_red)
cam_1.set_rgba(rgba=matt_blue)
cam_2.set_rgba(rgba=matt_black)
phoxi.set_rgba(rgba=reflective_black)

phoxi_loc_pos = np.array([.12, 0, 0.015])
phoxi_loc_rotmat = rm.rotmat_from_axangle(r_rotmat[:, 0], np.pi)
phoxi_gl_rotmat = phoxi_loc_rotmat.dot(r_rotmat)
phoxi_gl_pos = phoxi_gl_rotmat.dot(phoxi_loc_pos) + pos
phoxi.set_pose(pos=phoxi_gl_pos, rotmat=phoxi_gl_rotmat)
phoxi.attach_to(base)

cam_0_loc_pos = np.array([.075, .075, .03])
cam_0_loc_rotmat = np.eye(3)
cam_1_loc_pos = np.array([.075, -.075, .03])
cam_1_loc_rotmat = np.eye(3)
cam_2_loc_pos = np.array([.095, 0, .03])
cam_2_loc_rotmat = np.eye(3)

cam_0_gl_rotmat = cam_0_loc_rotmat.dot(r_rotmat)
cam_0_gl_pos = cam_0_gl_rotmat.dot(cam_0_loc_pos) + pos
cam_1_gl_rotmat = cam_1_loc_rotmat.dot(r_rotmat)
cam_1_gl_pos = cam_1_gl_rotmat.dot(cam_1_loc_pos) + pos
cam_2_gl_rotmat = cam_2_loc_rotmat.dot(r_rotmat)
cam_2_gl_pos = cam_2_gl_rotmat.dot(cam_2_loc_pos) + pos
cam_0.set_pose(pos=cam_0_gl_pos, rotmat=cam_0_gl_rotmat)
cam_1.set_pose(pos=cam_1_gl_pos, rotmat=cam_1_gl_rotmat)
cam_2.set_pose(pos=cam_2_gl_pos, rotmat=cam_2_gl_rotmat)

cam_0.attach_to(base)
cam_1.attach_to(base)
cam_2.attach_to(base)

base.run()
