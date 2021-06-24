import numpy as np
import modeling.geometric_model as gm


class Surface(object):

    def __init__(self):
        pass

    def _surface(self, X):
        raise NotImplementedError

    def get_gometricmodel(self,
                          rng=None,
                          granularity=.01,
                          rgba=[.7, .7, .3, 1]):
        surface_trm = self.get_trimesh(rng=rng, granularity=granularity)
        surface_gm = gm.GeometricModel(surface_trm, btwosided=True)
        surface_gm.set_rgba(rgba)
        return surface_gm

    def get_trimesh(self,
                    rng=None,
                    granularity=.01):
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
        domain_grid = np.meshgrid(np.linspace(a0min, a0max, na0, endpoint=True),
                                  np.linspace(a1min, a1max, na1, endpoint=True))
        return gm.gen_surface(domain_grid, self._surface)
