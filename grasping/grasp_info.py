# TODO not used as 20210113

import numpy as np


class GraspInfo(object):

    def __init__(self):
        self.jaw_width = 0.0
        self.gl_jaw_center = np.zeros(3)
        self.hnd_pos = np.zero(3)
        self. hnd_rotmat = np.eye(3)