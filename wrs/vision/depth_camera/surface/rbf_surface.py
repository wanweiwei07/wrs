from scipy.interpolate import RBFInterpolator
from wrs import vision as sfc


class RBFSurface(sfc.Surface):

    def __init__(self,
                 xydata,
                 zdata,
                 neighbors=None,
                 smoothing=0.0,
                 kernel='thin_plate_spline',
                 epsilon=None,
                 degree=None):
        """
        :param xydata:
        :param zdata:
        :param neighbors:
        :param smoothing:
        :param kernel:
        :param epsilon:
        :param degree:
        :param range: [[xmin, xmax], [ymin, ymax]],
                    min(xydata[:,0]), max(xydata[:,0]) and min(xydata[:,1]), max(xydata[:,1]) will be used in case of None
        author: weiwei
        date: 20210624
        """
        super().__init__(xydata, zdata)
        self._surface = RBFInterpolator(xydata,
                                        zdata,
                                        neighbors=neighbors,
                                        smoothing=smoothing,
                                        kernel=kernel,
                                        epsilon=epsilon,
                                        degree=degree)

    def get_zdata(self, xydata):
        return self._surface(xydata)
