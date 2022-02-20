import math
import numpy as np
import basis.robot_math as rm
import modeling.mesh_tools as mt

if __name__ == '__main__':
    '''
    author: weiwei
    date: 20201208osaka
    '''
    mt.convert_to_stl("shuidi_agv.stl", "shuidi_agv_meter.stl", pos=np.array([.2542, -.2542, 0]),
                      rotmat=rm.rotmat_from_axangle([0, 0, 1], math.pi / 2), scale_ratio=.001)
