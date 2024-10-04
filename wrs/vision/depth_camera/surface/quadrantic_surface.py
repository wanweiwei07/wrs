import numpy as np
from scipy.linalg import lstsq
from wrs import vision as sfc


class QuadraticSurface(sfc.Surface):
    """
    Quadratic surface fitting
    author: ruishuang
    date: 20210625
    """

    def __init__(self,
                 xydata,
                 zdata):
        super().__init__(xydata, zdata)
        A = np.c_[np.ones(xydata.shape[0]), xydata, np.prod(xydata, axis=1), xydata ** 2]
        self.coef, _, _, _ = lstsq(A, zdata)

    @staticmethod
    def func(xydata, *parameters):
        xydata = np.asarray(xydata)

        def quad(xdata, ydata, a, x, y, xy, x2, y2):
            return a + x * xdata + y * ydata + xy * xdata * ydata + x2 * xdata ** 2 + y2 * ydata ** 2

        z = np.zeros(len(xydata))
        for single_parameters in np.array(parameters).reshape(-1, 6):
            z += quad(xydata[:, 0], xydata[:, 1], *single_parameters)
        return z

    def get_zdata(self, xydata):
        zdata = QuadraticSurface.func(xydata, self.coef)
        return zdata
