import math
import numpy as np
import basis.robot_math as rm
import modeling.mesh_tools as mt

if __name__ == '__main__':
    '''
    There is an inherent transformation in the official Universal Robot UR3E dae file.
    The old trimesh used by our system does not support applying the transoformation.
    This file was designed to make up the failure.
    As 20201208, I have included the transformation in the io/dae.py file. This conversion is no longer needed.
    It is kept here for reference.
    author: weiwei
    date: 20201207osaka
    '''
    mt.convert_to_stl("base.dae", "base.stl")
    mt.convert_to_stl("shoulder.dae", "shoulder.stl")
    mt.convert_to_stl("upperarm.dae", "upperarm.stl")
    mt.convert_to_stl("forearm.dae", "forearm.stl")
    mt.convert_to_stl("wrist1.dae", "wrist1.stl")
    mt.convert_to_stl("wrist2.dae", "wrist2.stl")
    mt.convert_to_stl("wrist3.dae", "wrist3.stl")