import math
import numpy as np
import basis.robot_math as rm
import modeling.mesh_tools as mt

if __name__ == '__main__':
    mt.convert_to_stl("robotiq_arg2f_140_pad.stl", "robotiq_arg2f_140_pad.stl", scale_ratio=.001)