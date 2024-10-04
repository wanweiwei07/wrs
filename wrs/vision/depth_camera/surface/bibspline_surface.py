import scipy.interpolate as si
from wrs import vision as sfc
import numpy as np

class BiBSpline(sfc.Surface):

    def __init__(self,
                 xydata,
                 zdata,
                 degree_x=3,
                 degree_y=3):
        """
        :param xydata:
        :param zdata:
        :param degree_x: the degrees of the spline in x, 1~5, 3 recommended
        :param degree_y:
        author: weiwei
        date: 20210707
        """
        super().__init__(xydata, zdata)
        self._tck = si.bisplrep(xydata[:, 0],
                                xydata[:, 1],
                                zdata,
                                kx=degree_x,
                                ky=degree_y)

    def get_zdata(self, xydata):
        return_value = []
        for each_xy in xydata:
            each_z = si.bisplev(each_xy[0], each_xy[1], self._tck)
            return_value.append(each_z)
        return np.array(return_value)
        # return si.bisplev(xydata[:, 0], xydata[:, 1], self._tck)