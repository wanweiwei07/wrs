import numpy as np
from scipy.linalg import lstsq
from wrs import vision as sfc


class PlaneSurface(sfc.Surface):
    """
    Plane surface fitting
    author: weiwei
    date: 20210707
    """

    def __init__(self,
                 xydata,
                 zdata):
        super().__init__(xydata, zdata)
        A = np.c_[np.ones(xydata.shape[0]), xydata[:,0], xydata[:,1]]
        self.coef, _, _, _ = lstsq(A, zdata)

    @staticmethod
    def func(xydata, *parameters):
        xydata = np.asarray(xydata)

        def plane(xdata, ydata, a, x, y):
            return a + x * xdata + y * ydata

        z = np.zeros(len(xydata))
        for single_parameters in np.array(parameters).reshape(-1, 3):
            z += plane(xydata[:, 0], xydata[:, 1], *single_parameters)
        return z

    def get_zdata(self, xydata):
        zdata = PlaneSurface.func(xydata, self.coef)
        return zdata
