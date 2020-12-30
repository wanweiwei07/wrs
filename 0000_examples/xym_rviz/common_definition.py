import os
import math
import basis
import numpy as np
import modeling.geometricmodel as gm
import modeling.collisionmodel as cm
import robotsim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xav
import re

global_frame = gm.gen_frame()
# define robot and robot anime info
robot_instance = xav.XArm7YunjiMobile()
print(str(robot_instance.__class__))
print(re.findall(r"'(.*?)'", str(robot_instance.__class__)))
robot_jlc_name = 'arm'
robot_meshmodel_parameters = [None,  # tcp_jntid
                              None,  # tcp_loc_pos
                              None,  # tcp_loc_rotmat
                              False,  # toggle_tcpcs
                              False,  # toggle_jntscs
                              [0, .7, 0, .3],  # rgba
                              'auto']  # name
# define object and object anime info
objfile = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
obj = cm.CollisionModel(objfile)
obj_parameters = [[.3, .2, .1, 1]]  # rgba
obj_path = [[np.array([.85, 0, .17]), np.eye(3)]]  # [pos, rotmat]
