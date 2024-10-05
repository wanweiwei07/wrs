import numpy as np
import wrs.modeling.mesh_tools as mt

if __name__ == '__main__':
    '''
    author: weiwei
    date: 20201207osaka
    '''
    mt.convert_to_stl("block.stl", "block.stl", scale_ratio=np.ones(3)*.001)