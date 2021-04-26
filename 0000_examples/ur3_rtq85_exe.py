import math
import numpy as np
import robot_con.ur.ur3_rtq85_x as u3r85x

rbtx = u3r85x.UR3Rtq85X(robot_ip='10.2.0.51')
current_jnt_values = rbtx.get_jnt_values()
n_jnt_values = (current_jnt_values + (np.random.rand(6) - .5) * 1 / 18 * math.pi).tolist()
nn_jnt_values = (n_jnt_values + (np.random.rand(6) - .5) * 1 / 18 * math.pi).tolist()
nnn_jnt_values = (nn_jnt_values + (np.random.rand(6) - .5) * 1 / 18 * math.pi).tolist()
rbtx.move_jntspace_path([current_jnt_values, n_jnt_values, nn_jnt_values, nnn_jnt_values], control_frequency=0.008)