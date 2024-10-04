from wrs import modeling as gm


class Surface(object):

    def __init__(self, xydata, zdata):
        self.xydata = xydata
        self.zdata = zdata

    def get_zdata(self, domain):
        raise NotImplementedError

    def get_gometricmodel(self,
                          rng=None,
                          granularity=.003,
                          rgba=[.7, .7, .3, 1]):
        if rng is None:
            rng = [[min(self.xydata[:,0]), max(self.xydata[:,0])],
                   [min(self.xydata[:,1]), max(self.xydata[:,1])]]
        surface_gm = gm.gen_surface(self.get_zdata, rng=rng, granularity=granularity)
        surface_gm.set_rgba(rgba=rgba)
        return surface_gm
