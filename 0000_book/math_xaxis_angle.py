import math
import numpy as np
import basis.robot_math as rm

if __name__ == '__main__':
    x_axis = np.array([1,0,0])
    x_30_rotmat = rm.rotmat_from_axangle(x_axis, math.radians(30))
    print(x_30_rotmat)