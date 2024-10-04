import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.linalg import lstsq
from scipy.optimize import curve_fit

from wrs import basis as trm, modeling as cm


class Surface(object):

    def __init__(self, xydata, zdata):
        self.xydata = xydata
        self.zdata = zdata

    def _gen_surface(self, surface_callback, rng, granularity=.01):
        """
        :param surface_callback:
        :param rng: [[dim0_min, dim0_max], [dim1_min, dim1_max]]
        :return:
        author: weiwei
        date: 20210624
        """

        def _mesh_from_domain_grid(domain_grid, vertices):
            domain_0, domain_1 = domain_grid
            nrow = domain_0.shape[0]
            ncol = domain_0.shape[1]
            faces = np.empty((0, 3))
            for i in range(nrow - 1):
                urgt_pnt0 = np.arange(i * ncol, i * ncol + ncol - 1).T
                urgt_pnt1 = np.arange(i * ncol + 1 + ncol, i * ncol + ncol + ncol).T
                urgt_pnt2 = np.arange(i * ncol + 1, i * ncol + ncol).T
                faces = np.vstack((faces, np.column_stack((urgt_pnt0, urgt_pnt2, urgt_pnt1))))
                blft_pnt0 = np.arange(i * ncol, i * ncol + ncol - 1).T
                blft_pnt1 = np.arange(i * ncol + ncol, i * ncol + ncol + ncol - 1).T
                blft_pnt2 = np.arange(i * ncol + 1 + ncol, i * ncol + ncol + ncol).T
                faces = np.vstack((faces, np.column_stack((blft_pnt0, blft_pnt2, blft_pnt1))))
            return trm.Trimesh(vertices=vertices, faces=faces)

        a_min, a_max = rng[0]
        b_min, b_max = rng[1]
        n_a = round((a_max - a_min) / granularity)
        n_b = round((b_max - b_min) / granularity)
        domain_grid = np.meshgrid(np.linspace(a_min, a_max, n_a, endpoint=True),
                                  np.linspace(b_min, b_max, n_b, endpoint=True))
        domain_0, domain_1 = domain_grid
        domain = np.column_stack((domain_0.ravel(), domain_1.ravel()))
        codomain = surface_callback(domain)
        vertices = np.column_stack((domain, codomain))
        return _mesh_from_domain_grid(domain_grid, vertices)

    def get_zdata(self, domain):
        raise NotImplementedError

    def get_gometricmodel(self,
                          rng=None,
                          granularity=.01,
                          rgba=[.7, .7, .3, 1]):
        if rng is None:
            rng = [[min(self.xydata[:, 0])-.01, max(self.xydata[:, 0])+.01],
                   [min(self.xydata[:, 1])-.01, max(self.xydata[:, 1])+.01]]
        surface_trm = self._gen_surface(self.get_zdata, rng=rng, granularity=granularity)
        surface_cm = cm.CollisionModel(initor=surface_trm, toggle_twosided=True)
        surface_cm.set_rgba(rgba)
        return surface_cm


class RBFSurface(Surface):

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


class MixedGaussianSurface(Surface):
    def __init__(self,
                 xydata,
                 zdata,
                 n_mix=4):
        """
        :param xydata:
        :param zdata:
        :param neighbors:
        :param smoothing:
        :param kernel:
        :param epsilon:
        :param degree:
        :param range: [[a0min, a0max], [a1min, a1max]],
                    min(domain[:,0]), max(domain[:,0]) and min(domain[:,1]), max(domain[:,1]) will be used in case of None
        author: weiwei
        date: 20210624
        """
        super().__init__(xydata, zdata)
        guess_prms = np.array([[0, 0, .05, .05, .01]] * n_mix)
        self.popt, pcov = curve_fit(MixedGaussianSurface.mixed_gaussian, xydata, zdata, guess_prms.ravel())

    @staticmethod
    def mixed_gaussian(xydata, *parameters):
        """
        :param n_mix: number of gaussian for mixing
        :param xy:
        :param parameters: first gaussian parameters, second gaussian parameters, ...
                            parameter includs: x_mean, y_mean, x_delta, y_delta, attitude
        :return:
        author: weiwei
        date; 20210624
        """
        def gaussian(xdata, ydata, xmean, ymean, xdelta, ydelta, attitude):
            return attitude * np.exp(-((xdata - xmean) / xdelta) ** 2 - ((ydata - ymean) / ydelta) ** 2)

        z = np.zeros(len(xydata))
        xydata = np.asarray(xydata)
        for single_parameters in np.array(parameters).reshape(-1, 5):
            z += gaussian(xydata[:, 0], xydata[:, 1], *single_parameters)
        return z

    def get_zdata(self, xydata):
        zdata = MixedGaussianSurface.mixed_gaussian(xydata, self.popt)
        return zdata


class QuadraticSurface(Surface):
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
