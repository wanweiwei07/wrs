from wrs import vision as sfc
from scipy.optimize import curve_fit


class MixedGaussianSurface(sfc.Surface):

    def __init__(self,
                 xydata,
                 zdata,
                 n_mix=1,
                 init_guess=[0, 0, .05, .05, .01]):
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
        guess_prms = np.array([init_guess] * n_mix)
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
        for single_parameters in np.array(parameters).reshape(-1, 5):
            z += gaussian(xydata[:, 0], xydata[:, 1], *single_parameters)
        return z

    def get_zdata(self, xydata):
        zdata = MixedGaussianSurface.mixed_gaussian(xydata, self.popt)
        return zdata


if __name__ == '__main__':
    import numpy as np
    # from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt

    # The two-dimensional domain of the fit.
    xmin, xmax, nx = -5, 4, 75
    ymin, ymax, ny = -3, 7, 150
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)


    # Our function to fit is going to be a sum of two-dimensional Gaussians
    def gaussian(x, y, x0, y0, xalpha, yalpha, A):
        return A * np.exp(-((x - x0) / xalpha) ** 2 - ((y - y0) / yalpha) ** 2)


    # A list of the Gaussian parameters: x0, y0, xalpha, yalpha, A
    gprms = [(0, 2, 2.5, 5.4, 1.5),
             (-1, 4, 6, 2.5, 1.8),
             (-3, -0.5, 1, 2, 4),
             (3, 0.5, 2, 1, 5)]
    # Standard deviation of normally-distributed noise to add in generating
    # our test function to fit.
    noise_sigma = 0.1
    # The function to be fit is Z.
    Z = np.zeros(X.shape)
    for p in gprms:
        Z += gaussian(X, Y, *p)
    Z += noise_sigma * np.random.randn(*Z.shape)
    # Plot the 3D figure of the fitted function and the residuals.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma')
    ax.set_zlim(0, np.max(Z) + 2)
    plt.show()
    xdata = np.vstack((X.ravel(), Y.ravel())).T
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=np.array([7, 7, 20]), lookat_pos=np.array([0, 0, 0.05]))
    surface = MixedGaussianSurface(xdata, Z.ravel())
    surface_gm = surface.get_gometricmodel()
    surface_gm.attach_to(base)
    base.run()
