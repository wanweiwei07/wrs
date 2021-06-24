import numpy as np
from scipy.interpolate import RBFInterpolator
import basis.trimesh_generator as tg
import modeling.geometric_model as gm


class RBFSurface(object):

    def __init__(self, y, d,
                 neighbors=None,
                 smoothing=0.0,
                 kernel='thin_plate_spline',
                 epsilon=None,
                 degree=None,
                 rng=None,
                 granularity=.01,
                 toggle_debug=False):
        """

        :param y:
        :param d:
        :param neighbors:
        :param smoothing:
        :param kernel:
        :param epsilon:
        :param degree:
        :param rng: [[a0min, a0max], [a1min, a1max]],
                    min(y[:,0]), max(y[:,0]) and min(y[:,1]), max(y[:,1]) will be used in case of None
        """
        self._surface = RBFInterpolator(y, d,
                                        neighbors=neighbors,
                                        smoothing=smoothing,
                                        kernel=kernel,
                                        epsilon=epsilon,
                                        degree=degree)
        if rng is None:
            a0min = min(y[:, 0])
            a0max = max(y[:, 0])
            a1min = min(y[:, 1])
            a1max = max(y[:, 1])
        else:
            a0min, a0max = rng[0]
            a1min, a1max = rng[1]
        na0 = round((a0max - a0min) / granularity)
        na1 = round((a1max - a1min) / granularity)
        print(na0, na1)
        xgrid = np.meshgrid(np.linspace(a0min, a0max, na0, endpoint=True),
                            np.linspace(a1min, a1max, na1, endpoint=True))
        xg0, xg1 = xgrid
        xg = np.column_stack((xg0.ravel(), xg1.ravel()))
        zg = self._surface(xg)
        vertices = np.column_stack((xg, zg))
        if toggle_debug:
            for p in vertices:
                gm.gen_sphere(p, rgba=[0, 1, 0, 1], radius=.0005).attach_to(base)
        self._surface_trm = tg.mesh_from_domain_grid(xgrid, vertices)
        self._surface_gm = gm.GeometricModel(self._surface_trm, btwosided=True)

    def get_gometricmodel(self, rgba=[.7, .7, .3, 1]):
        return_gm = self._surface_gm.copy()
        return_gm.set_rgba(rgba)
        return return_gm

    def get_trimesh(self):
        return self._surface_trm