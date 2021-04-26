import math
import numpy as np
import basis.robot_math as rm
import modeling.mesh_tools as mt

if __name__ == '__main__':
    '''
    author: weiwei
    date: 20201207osaka
    '''
    mt.convert_to_stl("tubebig.stl", "tubebig.stl", scale_ratio=.001)