import time
import math
import numpy as np
from basis import robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.xarm_shuidi.xarm_shuidi as xss
import motion.probabilistic.rrt_differential_wheel_connect as rrtdwc

base = wd.World(cam_pos=[10, 1, 5], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# robot_s
component_name='agv'
robot_s = xss.XArmShuidi()
robot_s.gen_meshmodel().attach_to(base)
m_mat = robot_s.manipulability_axmat("arm", type="rotational")
print(m_mat)
gm.gen_ellipsoid(pos=robot_s.get_gl_tcp("arm")[0], axmat=m_mat).attach_to(base)
base.run()
