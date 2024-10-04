# from commpy import pnsequence
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import pyprimes
import random
from scipy.signal import correlate2d


class codedaperture():
    """
    Coded Aperture

    Parameters
    ----------
    type : str
        "1d", "rectangular", or "hexagonal"

    Attributes
    ----------
    base : array_like
        the "base" pattern; for rectangular arrays, this is the base pattern
        prior to tiling; for non-random hexagonal arrays, this is the rhombus
    aperture : array_like
        the physical coded aperture pattern; 1=open, 0=closed
    decoder : array_like
        the decoding matrix
    density : float
        ratio of closed to total number of pixels
    """

    def __init__(self, type):
        self.type = type
        self.base = None
        self.aperture = None
        self.decoder = None
        self.density = None

    @property
    def type(self):
        return self.__type

    @type.setter
    def type(self, value):
        if value not in ['1d', 'rectangular', 'hexagonal']:
            raise ValueError("Invalid value for type")
        self.__type = value

    @property
    def aperture(self):
        return self.__aperture

    @aperture.setter
    def aperture(self, value):
        self.__aperture = value

        # determine size
        if not self.__aperture is None:

            # determine size
            if self.type == "1d":
                self.length = len(self.__aperture)
            elif self.type == "rectangular":
                self.width = self.__aperture.shape[0]
                self.height = self.__aperture.shape[1]
            elif self.type == "hexagonal":
                pass

            # determine density
            self.density = 1.0 - (np.sum(self.__aperture) / self.__aperture.size)

    def gen_decoder(self, method='matched'):
        """
        Generate decoder matrix

        Parameters
        ----------
        method : str
            "matched" or "balanced"
        """
        self.decoder = np.zeros(self.aperture.shape)
        for i in range(self.decoder.shape[0]):
            for j in range(self.decoder.shape[1]):
                if method == "matched":
                    self.decoder[i, j] = self.aperture[i, j]
                if method == "balanced":
                    if self.aperture[i, j] == 1:
                        self.decoder[i, j] = 1
                    else:
                        self.decoder[i, j] = -1

    def gen_psf(self):
        """
        Generate Point Spread Function
        """

        self.psf = correlate2d(self.aperture, self.decoder)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        x = np.arange(self.psf.shape[1])
        y = np.arange(self.psf.shape[0])
        x, y = np.meshgrid(x, y)
        ax.plot_surface(x, y, self.psf, cmap=cm.get_cmap('coolwarm'))
        plt.title('Point Spread Function')

    def pre_plot(self, size=None):
        if size is None: size = 5
        plt.rcParams['figure.figsize'] = [size, size]
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        return fig, ax

    def post_plot(self, axis="on", title=None):
        plt.axis(axis)
        if title: plt.title(title)
        plt.show()


class randa1d(codedaperture):
    """
    Random Array 1-Dimensional

    Parameters
    ----------
    length : int
        number of elements in the array
    density : float
        fill factor fraction
    quiet : bool
        if True, will print mask info upon creation
    """

    def __init__(self, length=30, density=0.5, quiet=False):

        super().__init__("1d")
        self.name = "Random Array 1D"

        # randomly fill
        self.desired_density = density
        self.base = np.zeros(length, dtype=np.int8) + 1
        for i in range(length):
            if random.random() < self.desired_density:
                self.base[i] = 0
        self.aperture = self.base

        if not quiet: self.report()

    def report(self):
        """
        Report on the mask information
        """
        print("Random Array 1D")
        print("---------------")
        print(f"length:          {self.length}")
        print(f"desired density: {self.desired_density:.2f}")
        print(f"actual  density: {self.density:.2f}")

    def plot(self, height=10, size=None, axis="on"):
        """
        Plot the coded aperture to the screen

        Parameters
        ----------
        inverse : bool, optional
            if True, will invert the array before plotting
        size : int, optional
            size of the plot (default 8)
        axis : str
            "on" or "off
        """

        super().pre_plot(size=size)
        plot_aperture = np.transpose(np.tile(self.aperture, (height, 1)))
        plt.imshow(np.transpose(plot_aperture), cmap="binary_r", aspect=1)
        super().post_plot(axis=axis, title=f"{self.name}")


class randa2d(codedaperture):
    """
    Random Array 2-Dimensional

    Parameters
    ----------
    width : int
        width in elements
    height : int
        height in elements
    density : float
        fill factor fraction
    quiet : bool
        if True, will print mask info upon creation
    """

    def __init__(self, width=10, height=10, density=0.5, quiet=False):

        super().__init__("rectangular")
        self.name = "Random Array 2D"

        self.desired_density = density

        # randomly fill
        self.base = np.zeros([width, height]) + 1
        for i in range(width):
            for j in range(height):
                if random.random() < self.desired_density:
                    self.base[i, j] = 0
        self.aperture = self.base

        if not quiet: self.report()

    def report(self):
        """
        Report on the mask information
        """
        print("Random Array 2D")
        print("---------------")
        print(f"width:           {self.width}")
        print(f"height:          {self.height}")
        print(f"desired density: {self.desired_density:.2f}")
        print(f"actual  density: {self.density:.2f}")

    def plot(self, axis="on", size=None):
        """
        Plots the mask to the screen

        Parameters
        ----------
        axis : str
            "on" or "off"
        size : int
            size of the plot (default 8)
        """

        super().pre_plot(size=size)
        plt.imshow(np.transpose(self.aperture), cmap="binary_r", aspect=1)
        super().post_plot(axis=axis,
                          title=f"{self.name}")


class ura(codedaperture):
    """
    Uniformly Redundant Array

    Parameters
    ----------
    rank : int
        the rank of prime pairs to use (0 -> [5,3] 1 -> [13,11], etc.)
    tile : None or int or tuple of ints
        how to tile the pattern
    quiet : bool
        if True, will print information about the array upon creation
    center: bool
        if True, will center the base pattern in the array
    """

    def __init__(self, rank=4, tile=None, quiet=False, center=False):

        # initialize
        super().__init__("rectangular")
        self.name = "Uniformly Redundant Array"
        self.rank = rank
        self.tile = tile
        self.s, self.r = self.__get_prime_pairs(self.rank)

        # generate C_r(i) and C_r(j)
        C_r_i = np.zeros(self.r) - 1
        C_s_j = np.zeros(self.s) - 1
        for x in range(1, self.r):
            C_r_i[x ** 2 % self.r] = 1
        for y in range(1, self.s):
            C_s_j[y ** 2 % self.s] = 1

        # generate base (A_ij)
        A_ij = np.zeros((self.r, self.s))
        for i in range(self.r):
            for j in range(self.s):
                if i == 0:
                    A_ij[i, j] = 0
                elif j == 0:
                    A_ij[i, j] = 1
                elif C_r_i[i] * C_s_j[j] == 1:
                    A_ij[i, j] = 1
                else:
                    A_ij[i, j] = 0
        self.base = A_ij

        # determine tiling pattern
        if tile:
            if isinstance(tile, int):
                self.tile = (tile, tile)
            else:
                self.tile = tile
            self.aperture = np.tile(self.base, self.tile)
            if center:
                if not (self.tile[0] % 2):
                    self.aperture = np.roll(self.aperture, int((self.r + 1) / 2), axis=0)
                if not (self.tile[1] % 2):
                    self.aperture = np.roll(self.aperture, int((self.s + 1) / 2), axis=1)
        else:
            self.tile = (1, 1)
            self.aperture = self.base

        if not quiet: self.report()

    def report(self):
        """
        Report the array info
        """
        print("Uniformly Redundant Array")
        print("-------------------------")
        print(f"rank:   {self.rank}")
        print(f"r, s:   {self.r}, {self.s}")
        print(f"tile:   {self.tile}")
        print(f"width:  {self.width}")
        print(f"height: {self.height}")

    def __get_prime_pairs(self, rank):
        """
        Determine prime pairs at specified rank

        Parmeters
        ---------
        rank : int
            the rank of prime pairs to determine (0 -> 5, 1 -> 13, etc.)
        """

        assert rank >= 0, f"rank must be great than or equal to zero, got {rank}"

        pit = pyprimes.primes()

        # intialize
        p1 = next(pit)
        this_rank = -1

        # find primes
        while True:
            p2 = next(pit)
            if (p2 - p1) == 2:
                this_rank += 1
            else:
                p1 = p2
            if this_rank == rank:
                break

        return p1, p2

    def plot(self, border=0, axis="on", size=None):
        """
        Plots the mask to the screen

        Parameters
        ----------
        border : int
            width of border
        axis : str
            "on" or "off"
        size : int
            size of the plot (default 8)
        """

        super().pre_plot(size=size)
        if border > 0:
            pass
        plt.imshow(np.transpose(self.aperture), cmap="binary_r", aspect=1)
        plt.xlabel(f"{self.tile[0]}r")
        plt.ylabel(f"{self.tile[1]}s")
        super().post_plot(axis=axis, title=f"{self.name}")


class mura(codedaperture):
    """
    Modified Uniformly Redundant Array

    Parameters
    ----------
    rank : int
        the rank of prime to use
    tile : None or int or tuple of ints
        how to tile the pattern
    quiet : bool
        if True, will print information about the array upon creation
    center: bool
        if True, will center the base pattern in the array
    """

    def __init__(self, rank=5, quiet=False, tile=None, center=False):

        # initialize
        super().__init__("rectangular")
        self.name = "Modified Uniformly Redundant Array"
        self.rank = rank
        self.tile = tile
        self.L = self.__get_prime(rank)

        # generate C_r(I)
        C_r_i = np.zeros(self.L) - 1
        C_s_j = np.zeros(self.L) - 1
        for x in range(1, self.L):
            C_r_i[x ** 2 % self.L] = 1
        for y in range(1, self.L):
            C_s_j[y ** 2 % self.L] = 1

        # generate A_IJ
        A_ij = np.zeros([self.L, self.L])
        for i in range(self.L):
            for j in range(self.L):
                if i == 0:
                    A_ij[i, j] = 0
                elif j == 0:
                    A_ij[i, j] = 1
                elif C_r_i[i] * C_s_j[j] == 1:
                    A_ij[i, j] = 1
        self.base = A_ij

        # determine tiling pattern
        if tile:
            if isinstance(tile, int):
                self.tile = (tile, tile)
            else:
                self.tile = tile
            self.aperture = np.tile(self.base, self.tile)
            if center:
                if not (self.tile[0] % 2):
                    self.aperture = np.roll(self.aperture, int((self.L + 1) / 2), axis=0)
                if not (self.tile[1] % 2):
                    self.aperture = np.roll(self.aperture, int((self.L + 1) / 2), axis=1)
        else:
            self.tile = (1, 1)
            self.aperture = self.base
        self.aperture = np.transpose(self.aperture)

        if not quiet: self.report()

    def __get_prime(self, rank):
        """
        Determine prime of specified rank

        Parameters
        ----------
        rank : int
            the rank of the prime number
        """

        assert rank >= 0, f"rank must be great than or equal to zero, got {rank}"

        m = 1
        this_rank = -1
        while True:
            L = 4 * m + 1
            if pyprimes.is_prime(L):
                this_rank += 1
            if this_rank == rank:
                break
            m += 1
        return L

    def report(self):
        """
        Report on the mask information
        """
        print("Modified Uniformly Redundant Array")
        print("----------------------------------")
        print(f"rank:   {self.rank}")
        print(f"L:      {self.L}")
        print(f"tile:   {self.tile}")
        print(f"width:  {self.width}")
        print(f"height: {self.height}")

    def plot(self, border=0, axis="on", size=None):
        """
        Plots the coded aperture to the screen

        Parameters
        ----------
        border : int
            width of border
        axis : str
            "on" or "off"
        size : int
            size of the plot (default 8)
        """

        super().pre_plot(size=size)
        if border > 0:
            pass
        plt.imshow(np.transpose(self.aperture), cmap="binary_r", aspect=1)
        plt.xlabel(f"{self.tile[0]}L")
        plt.ylabel(f"{self.tile[1]}L")
        super().post_plot(axis=axis, title=f"{self.name}")


class pnp(codedaperture):
    """
    Pseudo-Noise Product Array

    Parameters
    ----------
    m : int
        degree of a_i
    n : int
        degree of b_i
    tile : None or int or tuple of ints
        how to tile the pattern
    quiet : bool
        if True, will print information about the array upon creation
    center: bool
        if True, will center the base pattern in the array
    """

    def __init__(self, m=10, n=10, quiet=False, tile=None, center=False):

        # initialize
        super().__init__("rectangular")
        self.name = "Pseudo-Noise Product Array"
        self.m = m
        self.n = n
        a = self.__prim_poly(m)
        b = self.__prim_poly(n)
        self.len_a = len(a)
        self.len_b = len(b)

        # generate mask
        self.base = np.zeros((self.len_a, self.len_b))
        for i in range(self.len_a):
            for j in range(self.len_b):
                self.base[i, j] = a[i] * b[j]
        # self.aperture = np.roll(self.aperture,(-(m-2),-(n-2)), axis=(0,1))

        # determine tiling pattern
        if tile:
            if isinstance(tile, int):
                self.tile = (tile, tile)
            else:
                self.tile = tile
            self.aperture = np.tile(self.base, self.tile)
            if center:
                if not (self.tile[0] % 2):
                    self.aperture = np.roll(self.aperture, int((self.len_a + 1) / 2), axis=0)
                if not (self.tile[1] % 2):
                    self.aperture = np.roll(self.aperture, int((self.len_b + 1) / 2), axis=1)
        else:
            self.tile = (1, 1)
            self.aperture = self.base

        self.report()

    def report(self):
        """
        Report the array info
        """
        print("Pseudo-Noise Product Array")
        print("--------------------------")
        print(f"m:               {self.m}")
        print(f"n:               {self.n}")
        print(f"length of m seq: {self.len_a}")
        print(f"length of n seq: {self.len_b}")
        print(f"width:           {self.width}")
        print(f"height:          {self.height}")

    def plot(self, border=0, axis="on", size=None):
        """
        Plots the coded aperture to the screen

        Parameters
        ----------
        border : int
            width of border
        axis : str
            "on" or "off"
        size : int
            size of the plot (default 8)
        """

        super().pre_plot(size=size)
        if border > 0:
            pass
        plt.imshow(np.transpose(self.aperture), cmap="binary_r", aspect=1)
        super().post_plot(axis=axis, title=f"{self.name}")

    def __prim_poly(self, m):
        """
        Primitive Polynomial

        Parameters
        ----------
        m : int
            degree (between 1 and 40); large numbers will take a very long time

        Returns
        -------
        pnsequence : ndarray
            a pseudo-random sequence satisfying the primitive polynomial of the
            degree specified
        """

        length = 2 ** m - 1

        # define the first 40 primitive polynomial indices here
        h_x = {1: (0), 2: (1, 0), 3: (1, 0), 4: (1, 0), 5: (2, 0), 6: (1, 0), 7: (1, 0),
               8: (6, 5, 1, 0), 9: (4, 0), 10: (3, 0), 11: (2, 0), 12: (7, 4, 3, 0),
               13: (4, 3, 1, 0), 14: (12, 11, 1, 0), 15: (1, 0), 16: (5, 3, 2, 0), 17: (3, 0),
               18: (7, 0), 19: (6, 5, 1, 0), 20: (3, 0), 21: (2, 0), 22: (1, 0), 23: (5, 0),
               24: (4, 3, 1, 0), 25: (3, 0), 26: (8, 7, 1, 0), 27: (8, 7, 1, 0), 28: (3, 0),
               29: (2, 0), 30: (16, 15, 1, 0), 31: (3, 0), 32: (28, 27, 1, 0), 33: (13, 0),
               34: (15, 14, 1, 0), 35: (2, 0), 36: (11, 0), 37: (12, 10, 2, 0), 38: (6, 5, 1, 0),
               39: (4, 0), 40: (21, 19, 2, 0)}
        min_m = np.min(list(int(key) for key in h_x.keys()))
        max_m = np.max(list(int(key) for key in h_x.keys()))

        # check the degree exists
        if (m < min_m) or (m > max_m):
            raise ValueError(f"degree must be betweeen {min_m} and {max_m}")

        # generate mask for this degree
        mask = np.zeros(m)
        for i in h_x[m]:
            mask[m - i - 1] = 1

        # initialize seed to match results from [MacWilliams 1976]
        seed = np.zeros(m)
        seed[0] = 1

        return pnsequence(m, seed, mask, 2 ** m - 1)


class randahex(codedaperture):
    """
    Random Array Hexagonal

    Parameters
    ----------
    radius : int
        vertex-to-vertex radius of the array, minus half the pixel width
    density : float
        fraction fill
    quiet : bool
        if True, will print information about the array upon creation

    Parameters
    ----------
    diameter : int
        due to the nature of hexagonal arrays, this is 2*radius+1
    """

    def __init__(self, radius=3, density=0.5, quiet=False):

        # get/determine mask properties
        super().__init__("hexagonal")
        self.name = "Random Array Hexagonal"
        self.radius = radius
        self.diameter = self.radius * 2 + 1
        self.side_width = radius + 1
        self.base = None
        self.aperture = np.zeros((self.diameter, self.diameter)) + 1
        self.locations = np.zeros((2, self.diameter, self.diameter))
        self.desired_density = density

        # generate mask pattern
        for i in range(self.diameter):
            for j in range(self.diameter):
                if (i + j > (self.radius - 1)) and (i + j < (self.diameter + self.radius)):
                    if random.random() < self.desired_density:
                        self.aperture[i, j] = 0
                else:
                    self.aperture[i, j] = np.nan

        # determine actual fill factor
        self.density = 1 - (np.sum(self.aperture == 1) / np.sum(~np.isnan(self.aperture)))

        # generate locations
        for i in range(self.diameter):
            for j in range(self.diameter):
                if not np.isnan(self.aperture[i, j]):
                    self.locations[0, i, j] = i * np.sqrt(3) - abs(j - self.radius) / 2.0
                    self.locations[1, i, j] = -i + j

        self.report()

    def report(self):
        """
        Report the array info
        """
        print("Random Array Hexagonal")
        print("----------------------")
        print(f"radius:          {self.radius}")
        print(f"diameter:        {self.diameter}")
        print(f"side width:      {self.side_width}")
        print(f"desired density: {self.desired_density:.2f}")
        print(f"actual  density: {self.density:.2f}")

    def plot(self, axis="on", size=None):
        """
        Plots the coded aperture to the screen

        Parameters
        ----------
        axis : str
            "on" or "off"
        size : int
            size of the plot (default 8)
        """

        # setup up plotting
        fig, ax = super().pre_plot(size=size)

        # determine hex geometry
        hex_width = 1.0  # face-to-face distance
        hex_vert = (hex_width) * (2.0 / np.sqrt(3))

        # draw hexagon array
        for y in range(self.diameter):
            row_width = self.diameter - abs(self.radius - y)
            start_i = np.max((self.radius - y, 0))
            for x in range(row_width):
                facecolor = 'w' if self.aperture[x + start_i, y] == 1 else 'k'
                alpha = 0.3 if self.aperture[x + start_i, y] == 1 else 0.9
                hex = RegularPolygon((x + 0.5 * abs(y - self.radius) - self.radius,
                                      ((y - self.radius) * ((3 / 2) * hex_vert / 2.0))),
                                     numVertices=6, radius=hex_vert / 2.0,
                                     orientation=np.radians(60),
                                     facecolor=facecolor, alpha=alpha,
                                     edgecolor='k')
                ax.add_patch(hex)

        plt.xlim(-self.radius * hex_vert, self.radius * hex_vert)
        plt.ylim(-self.radius, self.radius)
        super().post_plot(axis=axis, title=f"{self.name}")


class shura(codedaperture):
    """
    Skew-Hadamard Uniformly Redundant Array

    Parameters
    ----------
    rank : int
        determines the order, v, a prime of the form v=4n-1
        (default 6)
    r : int
        feeds into pattern (default 5)
    quiet : bool
        if True, will not print information about the array upon creation

    Parameters
    ----------
    diameter : int
        due to the nature of hexagonal arrays, this is 2*radius+1
    """

    def __init__(self, rank=4, r=5, radius=5, quiet=False):

        # get/determine mask properties
        super().__init__("hexagonal")
        self.name = "Skew-Hadamard Uniformly Redundant Array"
        self.rank = rank
        self.v = self.__get_order(self.rank)
        self.n = int((self.v + 1) / 4)
        self.r = r
        self.radius = radius
        self.diameter = self.radius * 2 + 1
        self.side_width = radius + 1
        self.base = np.zeros((self.diameter, self.diameter)) + 1
        self.aperture = np.zeros((self.diameter, self.diameter)) + 1
        self.locations = np.zeros((2, self.diameter, self.diameter))

        # calculate intermediates
        self.v = 4 * self.n - 1
        self.k = 2 * self.n - 1
        self.lam = self.n - 1

        # construct cyclic difference set D
        self.D = np.zeros((int((self.v - 1) / 2)), dtype=np.int32)
        for i in range(len(self.D)):
            self.D[i] = ((i + 1) ** 2) % self.v

        # determine labels
        self.rx = self.diameter
        self.ry = self.diameter
        self.l = np.zeros((self.rx, self.ry), dtype=np.int16)
        for i in range(self.rx):
            for j in range(self.ry):
                i_idx = i - self.radius
                j_idx = j - self.radius
                self.l[i, j] = (i_idx + self.r * j_idx) % self.v

        # calculate aperture
        self.base = np.zeros(self.l.shape, dtype=np.int16) + 1
        for i in range(self.base.shape[0]):
            for j in range(self.base.shape[1]):
                if self.l[i, j] in self.D:
                    self.base[i, j] = 0

        # map to aperture
        for i in range(self.diameter):
            for j in range(self.diameter):
                if (i + j > (self.radius - 1)) and (i + j < (self.diameter + self.radius)):
                    self.aperture[i, j] = self.base[i, j]
                else:
                    self.aperture[i, j] = np.nan

        self.report()

    def report(self):
        """
        Report the array info
        """
        print("Skew-Hadamard Uniformly Redundant Array")
        print("---------------------------------------")
        print(f"rank:        {self.rank}")
        print(f"n:           {self.n}")
        print(f"order (v):   {self.v}")
        print(f"k:           {self.k}")
        print(f"lambda:      {self.lam}")
        print(f"r:           {self.r}")
        print(f"radius:      {self.radius}")
        print(f"diameter:    {self.diameter}")
        print(f"side width:  {self.side_width}")

    def plot_rhombus(self, labels=False, labelsize=8, axis=True, size=None):
        """
        Plot the mask rhombus

        Parameters
        ----------
        labels : bool
            if True, will show the labels on top of each pixel
        labelsize : int
            fontsize for labels
        axis : bool
            if False, will not plot axis
        size : int
            size of the plot
        """

        # setup up plotting
        fig, ax = super().pre_plot(size=size)

        # determine hex vertex-to-vertex and radius
        hex_vert = 1 / (np.sqrt(3) / 2)
        hex_radius = (hex_vert) / 2.0

        for x_i in range(self.base.shape[0]):
            for y_i in range(self.base.shape[1]):

                # determine patch origin
                x = x_i + y_i * 0.5
                y = y_i / hex_vert

                # recenter
                x -= (self.base.shape[0] + self.base.shape[1] / 2.0) / 2.0 - 1 / hex_vert
                y -= (self.base.shape[1] * 1 / hex_vert) / 2.0 - hex_vert / 2.0

                # add hexagon
                facecolor = 'w' if self.base[x_i, y_i] == 1 else 'k'
                labelcolor = 'k' if self.base[x_i, y_i] == 1 else 'w'
                hex = RegularPolygon((x, y), numVertices=6, radius=hex_radius,
                                     orientation=np.radians(60),
                                     facecolor=facecolor, alpha=1.0, edgecolor='k')
                ax.add_patch(hex)

                # add label
                if labels:
                    plt.annotate(self.l[x_i, y_i], (x, y),
                                 ha='center', va='center', fontsize=labelsize,
                                 transform=ax.transAxes, color=labelcolor)

        plt.xlim(-self.rx / 1.2, self.rx / 1.2)
        plt.ylim(-self.ry / 2.0, self.ry / 2.0)
        super().post_plot(axis=axis, title=f"{self.name} Rhombus")

    def plot(self, labels=False, labelsize=8, axis=True, size=None):
        """
        Plot the aperture

        Parameters
        ----------
        labels : bool
            if True, will show the labels on top of each pixel
        labelsize : int
            fontsize for labels
        axis : bool
            if False, will not plot axis
        size : int
            size of the plot
        """

        # setup up plotting
        fig, ax = super().pre_plot(size=size)

        hex_width = 1.0  # face-to-face distance
        hex_vert = (hex_width) * (2.0 / np.sqrt(3))

        # draw hexagon array
        for y in range(self.diameter):
            row_width = self.diameter - abs(self.radius - y)
            start_i = np.max((self.radius - y, 0))
            for x in range(row_width):
                facecolor = 'w' if self.aperture[x + start_i, y] == 1 else 'k'
                labelcolor = 'k' if self.aperture[x + start_i, y] == 1 else 'w'
                alpha = 0.3 if self.aperture[x + start_i, y] == 1 else 0.9
                label = self.l[x + start_i, y]
                hex = RegularPolygon((x + 0.5 * abs(y - self.radius) - self.radius,
                                      ((y - self.radius) * ((3 / 2) * hex_vert / 2.0))),
                                     numVertices=6, radius=hex_vert / 2.0,
                                     orientation=np.radians(60),
                                     facecolor=facecolor, alpha=alpha,
                                     edgecolor='k')
                ax.add_patch(hex)
                if labels:
                    plt.annotate(label, (x + 0.5 * abs(y - self.radius) - self.radius,
                                         (y - self.radius) * ((3 / 2) * hex_vert / 2.0)),
                                 ha='center', va='center', fontsize=labelsize,
                                 transform=ax.transAxes, color=labelcolor)

        # set axis limits
        plt.xlim(-self.radius * hex_vert, self.radius * hex_vert)
        plt.ylim(-self.radius, self.radius)
        super().post_plot(axis=axis, title=f"{self.name}")

    def __get_order(self, rank):
        """
        Determine order from the given rank, n. Order is defined
        as a prime satisfying the condition v=4n-1

        Parameters
        ----------
        rank : int
            rank; atleast 1

        Returns
        -------
        v : int
            prime order
        """

        n = 1
        this_rank = -1
        while True:
            v = 4 * n - 1
            if pyprimes.isprime(v):
                this_rank += 1
            if this_rank == rank:
                break
            n += 1
        return v


class hura(codedaperture):
    """
    Hexagonal Uniformly Redundant Array

    Parameters
    ----------
    rank : int
        determines the order, v, a prime of the form 3 or 12n+7
        radius
    quiet : bool
        if True, will not print information about the array upon creation

    """

    def __init__(self, rank=4, radius=5, quiet=False):

        # get/determine mask properties
        super().__init__("hexagonal")
        self.name = "Hexagonal Uniformly Redundant Array"

        self.rank = rank
        self.v = self.__get_order_from_rank(self.rank)
        self.n = int((self.v - 7) / 12)
        self.r = self.__get_valid_r()

        # calculate mask size
        self.radius = radius
        self.diameter = self.radius * 2 + 1
        self.side_width = radius + 1

        # initialize base and locations
        self.aperture = np.zeros((self.diameter, self.diameter))
        self.locations = np.zeros((2, self.diameter, self.diameter))

        # calculate intermediates
        self.k = 2 * self.n - 1
        self.lamda = self.n - 1

        # construct cyclic difference set D
        self.D = np.zeros((int((self.v - 1) / 2)), dtype=np.int32)
        for i in range(len(self.D)):
            self.D[i] = ((i + 1) ** 2) % self.v

        # determine labels
        self.rx = self.diameter
        self.ry = self.diameter
        self.l = np.zeros((self.rx, self.ry), dtype=np.int16)
        for i in range(self.rx):
            for j in range(self.ry):
                i_idx = i - self.radius
                j_idx = j - self.radius
                self.l[i, j] = (i_idx + self.r * j_idx) % self.v

        # calculate base
        self.base = np.zeros(self.l.shape, dtype=np.int16) + 1
        for i in range(self.base.shape[0]):
            for j in range(self.base.shape[1]):
                if self.l[i, j] in self.D:
                    self.base[i, j] = 0

        # map to axial matrix
        for i in range(self.diameter):
            for j in range(self.diameter):
                if (i + j > (self.radius - 1)) and (i + j < (self.diameter + self.radius)):
                    self.aperture[i, j] = self.base[i, j]
                else:
                    self.aperture[i, j] = np.nan

        if not quiet: self.report()

    def __get_order_from_rank(self, rank):
        """
        Get the Order, v, from specified rank

        Parameters
        ----------
        rank : int
            rank, or nth order that satisfies 12n+7
        """
        if rank == 0:
            return 3
        else:
            n = 0
            this_rank = -1
            while True:
                v = 12 * n + 7
                if pyprimes.isprime(v):
                    this_rank += 1
                if this_rank == rank:
                    break
                n += 1
        return v

    def __get_valid_r(self):
        """
        Determines the valid r from v

        Returns
        r : int
            the value that satifies r**2 % v == (r-1) % v
        """
        r_not_found = True;
        r = 0
        while r_not_found:
            if (r ** 2 % self.v) == ((r - 1) % self.v):
                r_not_found = False
                break
            r += 1
        return r

    def report(self):
        """
        Report the array info
        """
        print("Hexagonal Uniformly Redundant Array")
        print("-----------------------------------")
        print(f"rank:         {self.rank}")
        print(f"n:            {self.n}")
        print(f"order (v):    {self.v}")
        print(f"k:            {self.k}")
        print(f"lambda:       {self.lamda}")
        print(f"r:            {self.r}")
        print(f"diameter:     {self.diameter}")
        print(f"side width:   {self.side_width}")

    def plot_rhombus(self, labels=False, labelsize=8, axis=True, size=None):
        """
        Plot the mask rhombus

        Parameters
        ----------
        labels : bool
            if True, will show the labels on top of each pixel
        labelsize : int
            fontsize for labels
        axis : bool
            if False, will not plot axis
        size : int
            size of the plot
        """

        # setup up plotting
        fig, ax = super().pre_plot(size=size)

        # determine hex vertex-to-vertex and radius
        hex_vert = 1 / (np.sqrt(3) / 2)
        hex_radius = (hex_vert) / 2.0

        for x_i in range(self.base.shape[0]):
            for y_i in range(self.base.shape[1]):

                # determine patch origin
                x = x_i + y_i * 0.5
                y = y_i / hex_vert

                # recenter
                x -= (self.base.shape[0] + self.base.shape[1] / 2.0) / 2.0 - 1 / hex_vert
                y -= (self.base.shape[1] * 1 / hex_vert) / 2.0 - hex_vert / 2.0

                # add hexagon
                facecolor = 'w' if self.base[x_i, y_i] == 1 else 'k'
                labelcolor = 'k' if self.base[x_i, y_i] == 1 else 'w'
                hex = RegularPolygon((x, y), numVertices=6, radius=hex_radius,
                                     orientation=np.radians(60),
                                     facecolor=facecolor, alpha=1.0, edgecolor='k')
                ax.add_patch(hex)

                # add label
                if labels:
                    plt.annotate(self.l[x_i, y_i], (x, y),
                                 ha='center', va='center', fontsize=labelsize,
                                 transform=ax.transAxes, color=labelcolor)

        plt.xlim(-self.rx / 1.2, self.rx / 1.2)
        plt.ylim(-self.ry / 2.0, self.ry / 2.0)
        super().post_plot(axis=axis, title=f"{self.name} Rhombus")

    def plot(self, labels=False, labelsize=8, axis=True, size=None):
        """
        Plot the aperture

        Parameters
        ----------
        labels : bool
            if True, will show the labels on top of each pixel
        labelsize : int
            fontsize for labels
        axis : bool
            if False, will not plot axis
        size : int
            size of the plot
        """

        # setup up plotting
        fig, ax = super().pre_plot(size=size)

        hex_width = 1.0  # face-to-face distance
        hex_vert = (hex_width) * (2.0 / np.sqrt(3))

        # draw hexagon array
        for y in range(self.diameter):
            row_width = self.diameter - abs(self.radius - y)
            start_i = np.max((self.radius - y, 0))
            for x in range(row_width):
                facecolor = 'w' if self.aperture[x + start_i, y] == 1 else 'k'
                labelcolor = 'k' if self.aperture[x + start_i, y] == 1 else 'w'
                alpha = 0.3 if self.aperture[x + start_i, y] == 1 else 0.9
                label = self.l[x + start_i, y]
                hex = RegularPolygon((x + 0.5 * abs(y - self.radius) - self.radius,
                                      ((y - self.radius) * ((3 / 2) * hex_vert / 2.0))),
                                     numVertices=6, radius=hex_vert / 2.0,
                                     orientation=np.radians(60),
                                     facecolor=facecolor, alpha=alpha,
                                     edgecolor='k')
                ax.add_patch(hex)
                if labels:
                    plt.annotate(label, (x + 0.5 * abs(y - self.radius) - self.radius,
                                         (y - self.radius) * ((3 / 2) * hex_vert / 2.0)),
                                 ha='center', va='center', fontsize=labelsize,
                                 transform=ax.transAxes, color=labelcolor)

        # set axis limits
        plt.xlim(-self.radius * hex_vert, self.radius * hex_vert)
        plt.ylim(-self.radius, self.radius)
        super().post_plot(axis=axis, title=f"{self.name}")


class fzp(codedaperture):
    """
    Fresnel Zone Plate

    Parameters
    ----------
    radius : int
        radius of the aperture to build
    resolution : int
        number of pixels per radius
    transmission : str
        "theoretical": transmission is floating point number
        "practical": transmission is 1/open or 0/closed
    quiet : bool
        if True, will not print information about the array upon creation
    """

    def __init__(self, radius=4, resolution=10, transmission="theoretical",
                 quiet=False):

        # get/determine mask properties
        super().__init__("rectangular")
        self.name = "Fresnel Zone Plate"

        # calculate mask size
        self.radius = radius
        self.resolution = resolution
        self.width = self.radius * self.resolution * 2 + 1
        self.height = self.radius * self.resolution * 2 + 1
        self.transmission = transmission

        # calculate locations
        self.r = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                r = np.sqrt((x - self.radius * self.resolution) ** 2 +
                            (y - self.radius * self.resolution) ** 2)
                # r = np.sqrt((x)**2 +
                #            (y)**2)
                self.r[x, y] = r / self.resolution

        # determine aperture
        self.aperture = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                self.aperture[x, y] = np.cos(self.r[x, y] ** 2)

        # modify if practical
        if self.transmission == "practical":
            self.aperture[np.where(self.aperture >= 0.5)] = 1
            self.aperture[np.where(self.aperture < 0.5)] = 0

        if not quiet: self.report()

    def report(self):
        """
        Report the array info
        """
        print("Fresnel Zone Plate")
        print("------------------")
        print(f"radius:       {self.radius}")
        print(f"resolution:   {self.resolution}")
        print(f"transmission: {self.transmission}")
        print(f"width:        {self.width}")
        print(f"height:       {self.height}")

    def plot(self, border=0, axis="on", size=None):
        """
        Plots the coded aperture to the screen

        Parameters
        ----------
        border : int
            width of border
        axis : str
            "on" or "off"
        size : int
            size of the plot (default 8)
        """

        super().pre_plot(size=size)
        if border > 0:
            pass
        plt.imshow(self.aperture, cmap="gray", aspect=1)
        plt.xlabel(f"width")
        plt.ylabel(f"height")
        super().post_plot(axis=axis, title=f"{self.name}")