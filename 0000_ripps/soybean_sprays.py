import numpy as np
import wrs.visualization.panda.world as wd
from wrs import robot_sim as rbt, modeling as cm

aluminium_rgba = [132 / 255, 135 / 255, 137 / 255, 1]
matt_red = [93 / 255, 53 / 255, 53 / 255, 1]
matt_blue = [53 / 255, 53 / 255, 93 / 255, 1]
matt_black = [44 / 255, 44 / 255, 44 / 255, 1]
reflective_black = [74 / 255, 74 / 255, 74 / 255, 1]

matt_purple = [133 / 255, 53 / 255, 153 / 255, 1]
matt_green = [53 / 255, 133 / 255, 53 / 255, 1]

base = wd.World(cam_pos=[1, 1, .5], auto_cam_rotate=False)

rbt_s = rbt.XArm7()
rbt_s.fk(jnt_values=[0, np.pi / 3, np.pi / 3, 0, 0, 0, 0])
rbt_model = rbt_s.gen_meshmodel(toggle_tcp_frame=True)
rbt_model.attach_to(base)
# base.run()

pos, rotmat = rbt_s.get_gl_tcp()
r_rotmat = rotmat
# pos = np.zeros(3)
# rotmat = np.eye(3)

spray_host = cm.CollisionModel(initor="objects/airgun_host.stl")
spray_host.set_rgba(rgba=aluminium_rgba)
spray_host.set_pose(pos, r_rotmat)
spray_host.attach_to(base)

spray = cm.CollisionModel(initor="objects/spray.stl")
spray_loc_pos = np.array([.0, -.07, 0.07])
spray_loc_rotmat = np.eye(3)
spray_gl_rotmat = spray_loc_rotmat.dot(r_rotmat)
spray_gl_pos = spray_gl_rotmat.dot(spray_loc_pos) + pos
spray.set_pose(pos=spray_gl_pos, rotmat=spray_gl_rotmat)
spray.attach_to(base)
spray.set_rgba(matt_green)

container = cm.CollisionModel(initor="objects/spray_container.stl")
container_loc_pos = np.array([.0, -.01, 0.15])
container_loc_rotmat = np.eye(3)
container_gl_rotmat = container_loc_rotmat.dot(r_rotmat)
container_gl_pos = container_gl_rotmat.dot(container_loc_pos) + pos
container.set_pose(pos=container_gl_pos, rotmat=container_gl_rotmat)
container.attach_to(base)
container.set_rgba(matt_purple)

base.run()
