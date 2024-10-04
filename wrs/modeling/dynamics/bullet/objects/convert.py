import numpy as np
from wrs import modeling as mt

if __name__ == '__main__':
    '''
    author: weiwei
    date: 20201207osaka
    '''
    mt.convert_to_stl("bunnysim.stl", "bunnysim_mm.stl", scale_ratio=np.ones(3)*1000)