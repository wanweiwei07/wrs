import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from scipy import integrate
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
# from sympy import symbols, diff, solve, exp

from wrs import basis as rm
import math

# import quadpy

PLOT_EDGE = 1


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj3d.proj_transform((x1, x2), (y1, y2), (z1, z2), renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


class Annotation3D(Annotation):
    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj3d.proj_transform(*self._xyz, renderer.M)
        self.xy = (x2, y2)
        super().draw(renderer)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """
    Add an 3d arrow to an `Axes3D` instance.
    """

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


def _annotate3D(ax, text, xyz, *args, **kwargs):
    """Add anotation `text` to an `Axes3d` instance."""

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)


setattr(Axes3D, 'annotate3D', _annotate3D)
setattr(Axes3D, 'arrow3D', _arrow3D)


def get_pcd_center(pcd):
    return np.array((np.mean(pcd[:, 0]), np.mean(pcd[:, 1]), np.mean(pcd[:, 2])))


def linear_interp_2d(p1, p2, step=1.0):
    diff = np.array(p1) - np.array(p2)
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    if abs(diff[0]) > abs(diff[1]):
        f = interpolate.interp1d(x, y, kind='linear')
        if x[0] < x[1]:
            x_new = list(np.arange(x[0], x[1], step)) + [x[1]]
        else:
            x_new = list(np.arange(x[1], x[0], step)) + [x[0]]
            x_new = x_new[::-1]
        y_new = list(f(x_new))
    else:
        f = interpolate.interp1d(y, x, kind='linear')
        if y[0] < y[1]:
            y_new = list(np.arange(y[0], y[1], step)) + [y[1]]
        else:
            y_new = list(np.arange(y[1], y[0], step)) + [y[0]]
            y_new = y_new[::-1]
        x_new = list(f(y_new))
    return list(zip(x_new, y_new))


def linear_interp_3d(p1, p2, step=1.0):
    def __find_2d(p1, p2, v, axis):
        p = []
        for current_axis in [0, 1, 2]:
            if current_axis != axis:
                p.append(np.interp(v, (p1[axis], p2[axis]), (p1[current_axis], p2[current_axis])))
            else:
                p.append(v)
        return p

    def __sort_p1_p2(p1, p2, axis=0):
        if p1[axis] > p2[axis]:
            return p2, p1
        return p1, p2

    result = []
    diff = np.abs(np.array(p1) - np.array(p2))
    if list(diff).count(0) == 2:
        axis = diff.argmax()
        p1, p2 = __sort_p1_p2(p1, p2, axis)
        for v in np.arange(p1[axis], p2[axis], step):
            p = __find_2d(p1, p2, v, axis)
            result.append(p)

    return result


def bilinear_interp_2d(target_point, pts, values):
    quad_xs = [p[0] for p in pts]
    quad_ys = [p[1] for p in pts]
    interp_f = interpolate.interp2d(quad_xs, quad_ys, values, kind='linear')
    result = interp_f(target_point[0], target_point[1])[0]
    return result


def interp_3d():
    # 3D example
    total_rad = 10
    z_factor = 3
    noise = 0.1

    num_true_pts = 200
    s_true = np.linspace(0, total_rad, num_true_pts)
    x_true = np.cos(s_true)
    y_true = np.sin(s_true)
    z_true = s_true / z_factor

    num_sample_pts = 80
    s_sample = np.linspace(0, total_rad, num_sample_pts)
    x_sample = np.cos(s_sample) + noise * np.random.randn(num_sample_pts)
    y_sample = np.sin(s_sample) + noise * np.random.randn(num_sample_pts)
    z_sample = s_sample / z_factor + noise * np.random.randn(num_sample_pts)

    tck, u = interpolate.splprep([x_sample, y_sample, z_sample], s=2)
    x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
    u_fine = np.linspace(0, 1, num_true_pts)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

    fig2 = plt.figure(2)
    ax3d = fig2.add_subplot(111, projection='3d')
    ax3d.plot(x_true, y_true, z_true, 'b')
    ax3d.plot(x_sample, y_sample, z_sample, 'r*')
    ax3d.plot(x_knots, y_knots, z_knots, 'go')
    ax3d.plot(x_fine, y_fine, z_fine, 'g')
    fig2.show()
    plt.show()


def fit_qua_surface(data, axis=2, toggledebug=False):
    """
    :param data:
    :param toggledebug:
    :return:
    """
    # best-fit quadratic curve
    # l = [max(data[:, 0]) - min(data[:, 0]), max(data[:, 1]) - min(data[:, 1]), max(data[:, 2]) - min(data[:, 2])]
    # axis = l.index(min(l))
    x, y, z = symbols('x, y, z')
    if axis == 0:
        A = np.c_[np.ones(data.shape[0]), data[:, 1:], np.prod(data[:, 1:], axis=1), data[:, 1:] ** 2]
        coef, _, _, _ = scipy.linalg.lstsq(A, data[:, 0])
        f = coef[0] + coef[1] * y + coef[2] * z + coef[3] * y * z + coef[4] * y ** 2 + coef[5] * z ** 2 - x
    elif axis == 1:
        A = np.c_[np.ones(data.shape[0]), np.hstack((data[:, :1], data[:, 2:])),
                  np.prod(np.hstack((data[:, :1], data[:, 2:])), axis=1), np.hstack((data[:, :1], data[:, 2:])) ** 2]
        coef, _, _, _ = scipy.linalg.lstsq(A, data[:, 1])
        f = coef[0] + coef[1] * x + coef[2] * z + coef[3] * x * z + coef[4] * x ** 2 + coef[5] * z ** 2 - y
    else:
        A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
        coef, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
        f = coef[0] + coef[1] * x + coef[2] * y + coef[3] * x * y + coef[4] * x ** 2 + coef[5] * y ** 2 - z
    if toggledebug:
        x_range = (min(data[:, 0].flatten()) - PLOT_EDGE, max(data[:, 0].flatten()) + PLOT_EDGE)
        y_range = (min(data[:, 1].flatten()) - PLOT_EDGE, max(data[:, 1].flatten()) + PLOT_EDGE)
        z_range = (min(data[:, 2].flatten()) - PLOT_EDGE, max(data[:, 2].flatten()) + PLOT_EDGE)
        plot_surface_f(ax, f, x_range, y_range, z_range, dense=.5)
        # ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='k', s=5, alpha=.5)
    return f


def trans_data_pcv(data, toggledebug=False, random_rot=True):
    pcv, pcaxmat = rm.compute_pca(data)
    inx = sorted(range(len(pcv)), key=lambda k: pcv[k])
    x_v = pcaxmat[:, inx[2]]
    y_v = pcaxmat[:, inx[1]]
    z_v = pcaxmat[:, inx[0]]
    pcaxmat = np.asarray([y_v, -x_v, -z_v]).T
    if random_rot:
        pcaxmat = np.dot(rm.rotmat_from_axangle([1, 0, 0], math.radians(5)), pcaxmat)
        pcaxmat = np.dot(rm.rotmat_from_axangle([0, 1, 0], math.radians(5)), pcaxmat)
        pcaxmat = np.dot(rm.rotmat_from_axangle([0, 0, 1], math.radians(5)), pcaxmat)
    data_tr = np.dot(pcaxmat.T, data.T).T
    if toggledebug:
        center = get_pcd_center(data)
        print('center:', center)
        scale = 2
        ax.arrow3D(center[0], center[1], center[2], scale * x_v[0], scale * x_v[1], scale * x_v[2],
                   mutation_scale=10, arrowstyle='->', color='r')
        ax.arrow3D(center[0], center[1], center[2], scale * y_v[0], scale * y_v[1], scale * y_v[2],
                   mutation_scale=10, arrowstyle='->', color='g')
        ax.arrow3D(center[0], center[1], center[2], scale * z_v[0], scale * z_v[1], scale * z_v[2],
                   mutation_scale=10, arrowstyle='->', color='b')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=5, alpha=.5)
        ax.scatter(data_tr[:, 0], data_tr[:, 1], data_tr[:, 2], c='g', s=5, alpha=.5)
    center = np.mean(data_tr, axis=0)
    data_tr = data_tr - center
    transmat = np.eye(4)
    transmat[:3,:3]=pcaxmat
    transmat[:3,3]=np.mean(data, axis=0)
    return data_tr, transmat


def gaussian(x, y, x0, y0, xalpha, yalpha, A):
    """
    Our function to fit is going to be a sum of two-dimensional Gaussians
    :param x:
    :param y:
    :param x0:
    :param y0:
    :param xalpha:
    :param yalpha:
    :param A:
    :return:
    """
    return A * np.exp(-((x - x0) / xalpha) ** 2 - ((y - y0) / yalpha) ** 2)


def fit_gss_surface(data, toggledebug=False):
    def _gaussian(M, *args):
        """
        This is the callable that is passed to curve_fit. M is a (2,N) array
        where N is the total number of data points in Z, which will be ravelled
        to one dimension.

        :param M:
        :param args:
        :return:
        """
        x, y = M
        arr = np.zeros(x.shape)
        for i in range(len(args) // 5):
            arr += gaussian(x, y, *args[i * 5:i * 5 + 5])
        return arr

    X, Y = np.meshgrid(np.unique(data[:, 0]), np.unique(data[:, 1]))
    Z = np.zeros(X.shape)
    for x, y, z in zip(data[:, 0], data[:, 1], data[:, 2]):
        Z[np.bitwise_and(y == Y, x == X)] = z

    Z = griddata(data[:, :2], data[:, 2], (X, Y), method='nearest')

    # Initial guesses to the fit parameters.
    guess_prms = [(np.mean(data[:, 0]), np.mean(data[:, 1]), 1, 1, 2)]
    p0 = [p for prms in guess_prms for p in prms]

    # Ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    xdata = np.vstack((X.ravel(), Y.ravel()))
    # Do the fit, using _gaussian function which understands flattened (ravelled) ordering of the data points.
    popt, pcov = curve_fit(_gaussian, xdata, Z.ravel(), p0)
    fit = np.zeros(Z.shape)
    for i in range(len(popt) // 5):
        fit += gaussian(X, Y, *popt[i * 5:i * 5 + 5])
    print('Fitted parameters:\n', popt)

    rms = np.sqrt(np.mean((Z - fit) ** 2))
    print('RMS residual =', rms)

    x, y, z = symbols('x, y, z')
    f = popt[4] * exp(-((x - popt[0]) / popt[2]) ** 2 - ((y - popt[1]) / popt[3]) ** 2) - z

    if toggledebug:
        x_range = (min(data[:, 0].flatten()) - PLOT_EDGE, max(data[:, 0].flatten()) + PLOT_EDGE)
        y_range = (min(data[:, 1].flatten()) - PLOT_EDGE, max(data[:, 1].flatten()) + PLOT_EDGE)
        z_range = (min(data[:, 2].flatten()) - PLOT_EDGE, max(data[:, 2].flatten()) + PLOT_EDGE)
        plot_surface_f(ax, f, x_range, y_range, z_range, axis='z', dense=.5, c=cm.coolwarm)

        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='k')
        # ax.plot_surface(X, Y, Z)

        # ax.contourf(X, Y, Z - fit, zdir='z', offset=-4, cmap='plasma')
        # ax.set_zlim(-4, np.max(fit))

    return f


def fit_plane(data, toggledebug=False):
    # best-fit linear plane
    A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    coef, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients
    x, y, z = symbols('x, y, z')
    print(coef)
    f = coef[0] * x + coef[1] * y + coef[2] - z

    if toggledebug:
        x_range = (min(data[:, 0].flatten()), max(data[:, 0].flatten()))
        y_range = (min(data[:, 1].flatten()), max(data[:, 1].flatten()))
        z_range = (min(data[:, 2].flatten()), max(data[:, 2].flatten()))
        plot_surface_f(ax, f, x_range, y_range, z_range, dense=.5)

    return f


def plot_surface_f(ax, f, x_range, y_range, z_range, dense=.5, alpha=.8, c=None, axis=None):
    x, y, z = symbols('x, y, z')
    coef_dict = f.as_coefficients_dict()
    if axis is None:
        if z in coef_dict.keys():
            axis = 'z'
        elif y in coef_dict.keys():
            axis = 'y'
        else:
            axis = 'x'
    if axis == 'z':
        s = solve(f, z)[0]
        X, Y = np.meshgrid(np.arange(x_range[0] - PLOT_EDGE, x_range[1] + PLOT_EDGE, dense),
                           np.arange(y_range[0] - PLOT_EDGE, y_range[1] + PLOT_EDGE, dense))
        Z = []
        for y in Y[:, 0]:
            z_res = []
            for x in X[0]:
                z_res.append(s.subs({'x': x, 'y': y}))
            Z.append(z_res)
        Z = np.asarray(Z).astype(dtype=float)
    elif axis == 'y':
        s = solve(f, y)[0]
        X, Z = np.meshgrid(np.arange(x_range[0] - PLOT_EDGE, x_range[1] + PLOT_EDGE, dense),
                           np.arange(z_range[0] - PLOT_EDGE, z_range[1] + PLOT_EDGE, dense))
        Y = []
        for z in Z[:, 0]:
            y = []
            for x in X[0]:
                y.append(s.subs({'x': x, 'z': z}))
            Y.append(y)
        Y = np.asarray(Y).astype(dtype=float)
    else:
        s = solve(f, x)[0]
        Y, Z = np.meshgrid(np.arange(y_range[0] - PLOT_EDGE, y_range[1] + PLOT_EDGE, dense),
                           np.arange(z_range[0] - PLOT_EDGE, z_range[1] + PLOT_EDGE, dense))
        X = []
        for y in Y[:, 0]:
            x = []
            for z in Z[0]:
                x.append(s.subs({'y': y, 'z': z}))
            X.append(x)
        X = np.asarray(X).astype(dtype=float)
    if c is None:
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=alpha, color='k')
    elif type(c) == str:
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=alpha, color=c)
    else:
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=alpha, cmap=c)
    # ax.set_zlim(z_range[0] - PLOT_EDGE, z_range[1] + PLOT_EDGE)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def get_nrml(coef, p):
    if len(coef) == 6:
        nrml = np.asarray([coef[1] + coef[3] * p[1] + 2 * coef[4] * p[0],
                           coef[2] + coef[3] * p[0] + 2 * coef[5] * p[1],
                           -1.0], dtype=np.float64)
        nrml = nrml / np.linalg.norm(nrml)
        return nrml
    else:
        print('wrong coef axis_length!')
        return None


def get_plane(v1, v2, p, transmat=None, toggledebug=False):
    if transmat is not None:
        v1 = np.dot(transmat, v1)
        v2 = np.dot(transmat, v2)
        p = np.dot(transmat, p)
    n = np.cross(v1, v2)
    coef = [n[0], n[1], n[2], -sum(n * p)]
    x, y, z = symbols('x, y, z')
    f = coef[0] * x + coef[1] * y + coef[2] * z + coef[3]
    if toggledebug:
        x_range = (p[0] - PLOT_EDGE, p[0] + PLOT_EDGE)
        y_range = (p[1] - PLOT_EDGE, p[1] + PLOT_EDGE)
        z_range = (p[2] - PLOT_EDGE, p[2] + PLOT_EDGE)
        ax.scatter([p[0]], [p[1]], [p[2]], c='g', s=10, alpha=1)
        ax.arrow3D(p[0], p[1], p[2], v1[0], v1[1], v1[2], mutation_scale=10, arrowstyle='->', color='g')
        ax.arrow3D(p[0], p[1], p[2], v2[0], v2[1], v2[2], mutation_scale=10, arrowstyle='->', color='g')
        plot_surface_f(ax, f, x_range, y_range, z_range, dense=.5)
        # plt.show()
    return f


def bisection(func, tgt, torlerance, lb, ub, mode='lb'):
    l = lb
    r = ub
    while True:
        v = l + (r - l) / 2
        tmpv, _ = func(v)
        # print(tmpv, v)
        if abs(tmpv - tgt) < torlerance:
            # print(v, tmpv)
            return v, tmpv
        else:
            if mode == 'lb':
                if tmpv < tgt:
                    r = v
                else:
                    l = v
            else:
                if tmpv < tgt:
                    l = v
                else:
                    r = v


def plot_intersc(ax, f, intersc, p, transmat=None, plot_edge=2, c='k', alpha=.5):
    xs = []
    ys = []
    zs = []
    x, y, z = symbols('x, y, z')

    if x in list(f.free_symbols):
        for x_value in np.arange(p[0] - plot_edge, p[0] + plot_edge, 0.1):
            y_value = f.subs({'x': x_value})
            z_value = solve(intersc.subs({'x': x_value, 'y': y_value}), z)[0]
            xs.append(x_value)
            ys.append(y_value)
            zs.append(z_value)
    else:
        for y_value in np.arange(p[1] - plot_edge, p[1] + plot_edge, 0.1):
            x_value = f.subs({'y': y_value})
            z_value = solve(intersc.subs({'x': x_value, 'y': y_value}), z)[0]
            xs.append(x_value)
            ys.append(y_value)
            zs.append(z_value)

    ps = np.hstack(([[x] for x in xs], [[y] for y in ys], [[z] for z in zs]))
    if transmat is not None:
        ps = np.dot(transmat, ps.T).T
    ax.plot(ps[:, 0], ps[:, 1], ps[:, 2], c=c, alpha=alpha)


def cal_surface_intersc(F, G, p, tgtlen, mode='ub', method='bisec', torlerance=1e-5, itg_axis='x', toggledebug=False):
    """

    :param F:
    :param G:
    :param p:
    :param tgtlen:
    :param mode: 'ub', 'lb'
    :param method: 'bisec', 'primitive'
    :param torlerance:
    :param itg_axis: 'x', 'y'
    :param toggledebug:
    :return:
    """

    def _func_L_x(x):
        return L_x.subs({'x': x})

    def _func_L_y(y):
        return L_y.subs({'y': y})

    def _elimination(F1, F2, p, itg_axis=itg_axis):
        if itg_axis == 'x':
            x_y1 = solve(F1, y)[0]
            x_y2 = solve(F1, y)[1]
            y_s1 = x_y1.subs({'x': p[0]})
            y_s2 = x_y2.subs({'x': p[0]})
            if abs(p[1] - y_s1) < abs(p[1] - y_s2):
                return x_y1, F2.subs({'y': x_y1})
            else:
                return x_y2, F2.subs({'y': x_y2})
        else:
            y_x1 = solve(F1, x)[0]
            y_x2 = solve(F1, x)[1]
            x_s1 = y_x1.subs({'y': p[1]})
            x_s2 = y_x2.subs({'y': p[1]})
            if abs(p[0] - x_s1) < abs(p[0] - x_s2):
                return y_x1, F2.subs({'x': y_x1})
            else:
                return y_x2, F2.subs({'x': y_x2})

    def _solve_bnd(tgt_func, tgtlen, mode, start_p, eliminated_func, itg_axis=itg_axis):
        if mode not in ['ub', 'lb']:
            print('wrong mode!')

        step = tgtlen * 1.5
        if itg_axis == 'x':
            if mode == 'ub':
                func = lambda x: integrate.quad(tgt_func, start_p[0], x)
                x_final, real_len = bisection(func, tgtlen, torlerance, start_p[0], start_p[0] + step, 'ub')
            else:
                func = lambda x: integrate.quad(tgt_func, x, start_p[0])
                x_final, real_len = bisection(func, tgtlen, torlerance, start_p[0] - step, start_p[0], 'lb')
            y_final = eliminated_func.subs({'x': x_final})
            z_final = float(F.subs({'x': x_final, 'y': y_final, 'z': 0}))

        else:
            if mode == 'ub':
                func = lambda y: integrate.quad(tgt_func, start_p[1], y)
                y_final, real_len = bisection(func, tgtlen, torlerance, start_p[1], start_p[1] + step, 'ub')
            else:
                func = lambda y: integrate.quad(tgt_func, y, start_p[1])
                y_final, real_len = bisection(func, tgtlen, torlerance, start_p[1] - step, start_p[1], 'lb')
            x_final = eliminated_func.subs({'y': y_final})
            z_final = float(F.subs({'x': x_final, 'y': y_final, 'z': 0}))

        return np.asarray([x_final, y_final, z_final], dtype=np.float64), real_len

    if itg_axis not in ['x', 'y']:
        print('wrong integration axis!')
        return None

    x, y, z = symbols('x, y, z')

    F_dx = diff(F, x)
    F_dy = diff(F, y)
    F_dz = diff(F, z)
    G_dx = diff(G, x)
    G_dy = diff(G, y)
    G_dz = diff(G, z)

    J1 = F_dy * G_dz - F_dz * G_dy
    J2 = F_dz * G_dx - F_dx * G_dz
    J3 = F_dx * G_dy - F_dy * G_dx

    if itg_axis == 'x':
        L = (1 + (J2 / J1) ** 2 + (J3 / J1) ** 2) ** (1 / 2)
    else:
        L = (1 + (J1 / J2) ** 2 + (J3 / J2) ** 2) ** (1 / 2)

    intersc = F.subs({'z': solve(G, z)[0]})

    # print('intersc', intersc)
    # print('f_l', G)

    if method == 'bisec':
        if itg_axis == 'x':
            f_x, L_x = _elimination(intersc, L, p)
            if toggledebug:
                plot_intersc(ax, f_x, F, p)
                plot_intersc(ax, f_x, F, p, transmat=transmat)
            result_p, result_len = _solve_bnd(_func_L_x, tgtlen, mode, p, f_x)
            return result_p, result_len, F, G, f_x
        else:
            f_y, L_y = _elimination(intersc, L, p)
            if toggledebug:
                plot_intersc(ax, f_y, F, p)
                plot_intersc(ax, f_y, F, p, transmat=transmat)
            result_p, result_len = _solve_bnd(_func_L_y, tgtlen, mode, p, f_y)
            return result_p, result_len, F, G, f_y

    # elif method == 'primitive':
    #     x_y, L = _elimination(intersc, L, p, obj='y')
    #     # Get origin and calculate the upper bound
    #     prim_L = itg(nsimplify(simplify(L)), x)
    #     print(f'origin L: {prim_L}')
    #     line_length = 5
    #     F_p0 = prim_L.subs({'x': p[0]})
    #     x_final = solve(prim_L - F_p0 - line_length, x)
    #     y_final = x_y.subs({'x': x_final})
    #     z_final = float(F.subs({'x': x_final, 'y': y_final, 'z': 0}))
    #     if toggledebug:
    #         plot_intersc(ax, x_y, F, p)
    #     return np.asarray([x_final, y_final, z_final]), tgtlen, F, G, x_y


def cal_surface_intersc_p2p(F, G, p1, p2, itg_axis='x'):
    """
    :param coef1:
    :param coef2:
    :param p:
    :param tgtlen:
    :param mode: 'ub', 'lb'
    :param method: 'bisec', 'primitive'
    :param torlerance:
    :param itg_axis: 'x', 'y'
    :param toggledebug:
    :return:
    """

    def _elimination(F1, F2, p, obj='y'):
        if obj == 'x':
            y_x1 = solve(F1, x)[0]
            y_x2 = solve(F1, x)[1]
            x_s1 = y_x1.subs({'y': p[1]})
            x_s2 = y_x2.subs({'y': p[1]})
            if abs(p[0] - x_s1) < abs(p[0] - x_s2):
                return y_x1, F2.subs({'x': y_x1})
            else:
                return y_x2, F2.subs({'x': y_x2})
        else:
            x_y1 = solve(F1, y)[0]
            x_y2 = solve(F1, y)[1]
            y_s1 = x_y1.subs({'x': p[0]})
            y_s2 = x_y2.subs({'x': p[0]})
            if abs(p[1] - y_s1) < abs(p[1] - y_s2):
                return x_y1, F2.subs({'y': x_y1})
            else:
                return x_y2, F2.subs({'y': x_y2})

    def _func_L_x(x):
        return L_x.subs({'x': x})

    def _func_L_y(y):
        return L_y.subs({'y': y})

    if itg_axis not in ['x', 'y']:
        print('wrong integration axis!')
        return None

    x, y, z = symbols('x, y, z')

    F_dx = diff(F, x)
    F_dy = diff(F, y)
    F_dz = diff(F, z)
    G_dx = diff(G, x)
    G_dy = diff(G, y)
    G_dz = diff(G, z)

    J1 = F_dy * G_dz - F_dz * G_dy
    J2 = F_dz * G_dx - F_dx * G_dz
    J3 = F_dx * G_dy - F_dy * G_dx

    intersc = F.subs({'z': solve(G, z)[0]})

    if itg_axis == 'x':
        L = (1 + (J2 / J1) ** 2 + (J3 / J1) ** 2) ** (1 / 2)
        x_y, L_x = _elimination(intersc, L, p1, obj='y')
        gd, _ = integrate.quad(_func_L_x, p1[0], p2[0])
        return abs(gd)
    else:
        L = (1 + (J1 / J2) ** 2 + (J3 / J2) ** 2) ** (1 / 2)
        y_x, L_y = _elimination(intersc, L, p1, obj='x')
        gd, _ = integrate.quad(_func_L_y, p1[1], p2[1])
        return abs(gd)


if __name__ == '__main__':
    TB = True
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # p1 = (-50, -50, 100)
    # p2 = (-50, -50, 0)
    # linear_interp_3d(p1, p2, 1)
    # interp_3d()

    """
    fit surface
    """
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.array([[1.0, -0.5, 0.8], [-0.5, 1.1, 0.0], [0.8, 0.0, 1.0]])
    data = np.random.multivariate_normal(mean, cov, 50)

    # f_q_base = fit_qua_surface(data, toggledebug=TB)
    # print(f_q_base)

    data_tr, transmat = trans_data_pcv(data)
    f_q = fit_plane(data_tr, toggledebug=TB)
    print(f_q)

    f_gaussian = fit_gss_surface(data_tr, toggledebug=TB)
    print(f_gaussian)
    plt.show()

    v1 = (1, 0, 0)
    v2 = (0, 1, 0)
    p = data[0]
    ax.scatter([p[0]], [p[1]], [p[2]], c='r', s=10, alpha=1)
    ax.arrow3D(p[0], p[1], p[2], v1[0], v1[1], v1[2], mutation_scale=10, arrowstyle='->', color='r')
    ax.arrow3D(p[0], p[1], p[2], v2[0], v2[1], v2[2], mutation_scale=10, arrowstyle='->', color='r')

    f_l = get_plane(v1, v2, p, transmat=transmat.T, toggledebug=TB)
    print(f_l)

    f_l_base = get_plane(v1, v2, p, toggledebug=TB)
    print(f_l_base)

    p_ub_new, _, _, _, _ = \
        cal_surface_intersc(f_gaussian, f_l, np.dot(transmat.T, p), tgtlen=3, toggledebug=TB, mode='ub', itg_axis='y')
    print(p, p_ub_new)

    p_ub = np.dot(transmat, p_ub_new)
    p_new = np.dot(transmat.T, p)
    ax.scatter([p_ub[0]], [p_ub[1]], [p_ub[2]], c='m', s=10, alpha=.5)
    ax.scatter([p_ub_new[0]], [p_ub_new[1]], [p_ub_new[2]], c='y', s=10, alpha=1)

    length = cal_surface_intersc_p2p(f_gaussian, f_l, p_new, p_ub_new, itg_axis='y')
    length_base = cal_surface_intersc_p2p(f_q_base, f_l_base, p, p_ub, itg_axis='y')

    print('result:', length, length_base)
    plt.show()

    p_lb, _, _, _, _ = cal_surface_intersc(f_q, f_l, p, tgtlen=3, toggledebug=TB, mode='lb', itg_axis='y')
    print(p, p_lb)

    p_ub_new, _, _, _, _ = cal_surface_intersc(f_q, f_l, p, tgtlen=3, toggledebug=TB, mode='ub', itg_axis='x')
    print(p, p_ub_new)

    p_lb, _, _, _, _ = cal_surface_intersc(f_q, f_l, p, tgtlen=3, toggledebug=TB, mode='lb', itg_axis='x')
    print(p, p_lb)
