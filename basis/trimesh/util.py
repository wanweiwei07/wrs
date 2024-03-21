'''
trimesh.util: utility functions

Only imports from numpy and the standard library are allowed in this file.
'''

import numpy as np
import logging
import hashlib
import base64
from collections import defaultdict, deque
from sys import version_info

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

if version_info.major >= 3:
    basestring = str

log = logging.getLogger('trimesh')
log.addHandler(logging.NullHandler())

# include constants here so we don't have to import a floating point threshold for 0.0
# we are setting it to 100x the resolution of a float64 which works out to be 1e-13
TOL_ZERO = np.finfo(np.float64).resolution * 100
# how close to merge vertices
TOL_MERGE = 1e-8


def unitize(vectors, check_valid=False, threshold=None):
    """
    Unitize a vector or an array or row-vectors.

    Parameters
    ------------
    vectors : (n,m) or (j) float
       Vector or vectors to be unitized
    check_valid :  bool
       If set, will return mask of nonzero vectors
    threshold : float
       Cutoff for a value to be considered zero.

    Returns
    ---------
    unit :  (n,m) or (j) float
       Input vectors but unitized
    valid : (n,) bool or bool
        Mask of nonzero vectors returned if `check_valid`
    """
    # make sure we have a numpy array
    vectors = np.asanyarray(vectors)
    # allow user to set zero threshold
    if threshold is None:
        threshold = TOL_ZERO
    if len(vectors.shape) == 2:
        # for (m, d) arrays take the per-row unit vector
        # using sqrt and avoiding exponents is slightly faster
        # also dot with ones is faser than .sum(axis=1)
        norm = np.sqrt(np.dot(vectors * vectors, [1.0] * vectors.shape[1]))
        # non-zero norms
        valid = norm > threshold
        # in-place reciprocal of nonzero norms
        norm[valid] **= -1
        # multiply by reciprocal of norm
        unit = vectors * norm.reshape((-1, 1))
    elif len(vectors.shape) == 1:
        # treat 1D arrays as a single vector
        norm = np.sqrt(np.dot(vectors, vectors))
        valid = norm > threshold
        if valid:
            unit = vectors / norm
        else:
            unit = vectors.copy()
    else:
        raise ValueError("vectors must be (n, ) or (n, d)!")
    if check_valid:
        return unit[valid], valid
    return unit



def transformation_2D(offset=[0.0, 0.0], theta=0.0):
    '''
    2D homogeonous transformation matrix
    '''
    T = np.eye(3)
    s = np.sin(theta)
    c = np.cos(theta)

    T[0, 0:2] = [c, s]
    T[1, 0:2] = [-s, c]
    T[0:2, 2] = offset
    return T


def euclidean(a, b):
    '''
    Euclidean linear_distance between vectors a and b
    '''
    return np.sum((np.array(a) - b) ** 2) ** .5


def is_file(obj):
    return hasattr(obj, 'read')


def is_string(obj):
    return isinstance(obj, basestring)


def is_dict(obj):
    return isinstance(obj, dict)


def is_sequence(obj):
    """
    returns True if obj is a sequence.
    :param obj:
    :return:
    """
    seq = (not hasattr(obj, "strip") and
           hasattr(obj, "__getitem__") or
           hasattr(obj, "__iter__"))
    seq = seq and not isinstance(obj, dict)
    # numpy sometimes returns objects that are single float64 values but sure look like sequences, so we check the shape
    if hasattr(obj, 'shape'):
        seq = seq and obj.shape != ()
    return seq


def is_shape(obj, shape):
    """
    Compare the shape of a numpy.ndarray to a target shape,  with any value less than zero being considered a wildcard
    :param obj: np.ndarray to check the shape of
    :param shape: list or tuple of shape. Any negative term will be considered a wildcard
           Any tuple term will be evaluated as an OR
    :return: bool, True if shape of obj matches query shape
    ------------------------ e. g.
    In [1]: a = np.random.random((100,3))
    In [2]: a.shape
    Out[2]: (100, 3)
    In [3]: trimesh.util.is_shape(a, (-1,3))
    Out[3]: True
    In [4]: trimesh.util.is_shape(a, (-1,3,5))
    Out[4]: False
    In [5]: trimesh.util.is_shape(a, (100,-1))
    Out[5]: True
    In [6]: trimesh.util.is_shape(a, (-1,(3,4)))
    Out[6]: True
    In [7]: trimesh.util.is_shape(a, (-1,(4,5)))
    Out[7]: False
    """
    if (not hasattr(obj, 'shape') or
            len(obj.shape) != len(shape)):
        return False
    for i, target in zip(obj.shape, shape):
        # check if current field has multiple acceptable values
        if is_sequence(target):
            if i in target:
                continue
            else:
                return False
        # check if current field is a wildcard
        if target < 0:
            if i == 0:
                return False
            else:
                continue
        # since we have a single target and a single value, if they are not equal we have an answer
        if target != i:
            return False
    # since none of the checks failed, the two shapes are the same
    return True


def make_sequence(obj):
    '''
    Given an object, if it is a sequence return, otherwise
    add it to a axis_length 1 sequence and return.

    Useful for wrapping functions which sometimes return single 
    objects and other times return lists of objects. 
    '''
    if is_sequence(obj):
        return np.array(obj)
    else:
        return np.array([obj])


def vector_hemisphere(vectors, return_sign=False):
    """
    For a set of 3D vectors alter the sign so they are all in the upper hemisphere.
    If the vector lies on the plane all vectors with negative Y will be reversed.
    If the vector has a zero Z and Y value vectors with a negative X value will be reversed.
    :param vectors: (n, 3) float
    :param return_sign: bool Return the sign mask or not
    :return: oriented: (n, 3) float Vectors with same magnitude as source but possibly reversed to ensure all vectors
            are in the same hemisphere.
    sign : (n,) float [OPTIONAL] sign of original vectors
    """
    # vectors as numpy array
    vectors = np.asanyarray(vectors, dtype=np.float64)
    if is_shape(vectors, (-1, 2)):
        # 2D vector case check the Y value and reverse vector motion_vec if negative.
        negative = vectors < -TOL_ZERO
        zero = np.logical_not(np.logical_or(negative, vectors > TOL_ZERO))
        signs = np.ones(len(vectors), dtype=np.float64)
        # negative Y values are reversed
        signs[negative[:, 1]] = -1.0
        # zero Y and negative X are reversed
        signs[np.logical_and(zero[:, 1], negative[:, 0])] = -1.0
    elif is_shape(vectors, (-1, 3)):
        # 3D vector case
        negative = vectors < -TOL_ZERO
        zero = np.logical_not(np.logical_or(negative, vectors > TOL_ZERO))
        # move all                          negative Z to positive
        # then for zero Z vectors, move all negative Y to positive
        # then for zero Y vectors, move all negative X to positive
        signs = np.ones(len(vectors), dtype=np.float64)
        # all vectors with negative Z values
        signs[negative[:, 2]] = -1.0
        # all on-plane vectors with negative Y values
        signs[np.logical_and(zero[:, 2], negative[:, 1])] = -1.0
        # all on-plane vectors with zero Y values
        # and negative X values
        signs[np.logical_and(np.logical_and(zero[:, 2], zero[:, 1]), negative[:, 0])] = -1.0
    else:
        raise ValueError('vectors must be (n, 3)!')
    # apply the signs to the vectors
    oriented = vectors * signs.reshape((-1, 1))
    if return_sign:
        return oriented, signs
    return oriented


def vector_to_spherical(cartesian):
    """
    convert a set of cartesian points to (n,2) spherical vectors
    :param cartesian:
    :return:
    """
    x, y, z = np.array(cartesian).T
    # cheat on divide by zero errors
    x[np.abs(x) < TOL_MERGE] = TOL_ZERO
    spherical = np.column_stack((np.arctan(y / x), np.arccos(z)))
    return spherical


def spherical_to_vector(spherical):
    """
    Convert a set of nx2 spherical vectors to nx3 vectors
    :param spherical:
    :return:
    author: revised by weiwei
    date: 20210120
    """
    spherical = np.asanyarray(spherical, dtype=np.float64)
    if not is_shape(spherical, (-1, 2)):
        raise ValueError('spherical coordinates must be (n, 2)!')
    theta, phi = spherical.T
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    vectors = np.column_stack((ct * sp,
                               st * sp,
                               cp))
    return vectors


def diagonal_dot(a, b):
    '''
    Dot product by row of a and b.

    Same as np.diag(np.dot(a, b.T)) but without the monstrous 
    intermediate matrix.
    '''
    result = (np.array(a) * b).sum(axis=1)
    return result

def row_norm(data):
    """
    Compute the norm per-row of a numpy array.

    This is identical to np.linalg.norm(data, axis=1) but roughly
    three times faster due to being less general.

    In [3]: %timeit trimesh.util.row_norm(a)
    76.3 us +/- 651 ns per loop

    In [4]: %timeit np.linalg.norm(a, axis=1)
    220 us +/- 5.41 us per loop

    Parameters
    -------------
    data : (n, d) float
      Input 2D data to calculate per-row norm of

    Returns
    -------------
    norm : (n,) float
      Norm of each row of input array
    """
    return np.sqrt(np.dot(data ** 2, [1] * data.shape[1]))


def three_dimensionalize(points, return_2D=True):
    '''
    Given a set of (n,2) or (n,3) points, return them as (n,3) points

    Arguments
    ----------
    points:    (n, 2) or (n,3) points
    return_2D: boolean flag

    Returns
    ----------
    if return_2D: 
        is_2D: boolean, True if points were (n,2)
        points: (n,3) set of points
    else:
        points: (n,3) set of points
    '''
    points = np.asanyarray(points)
    shape = points.shape

    if len(shape) != 2:
        raise ValueError('Points must be 2D array!')

    if shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
        is_2D = True
    elif shape[1] == 3:
        is_2D = False
    else:
        raise ValueError('Points must be (n,2) or (n,3)!')

    if return_2D:
        return is_2D, points
    return points


def grid_arange_2D(bounds, step):
    '''
    Return a 2D grid with specified spacing

    Arguments
    ---------
    bounds: (2,2) list of [[minx, miny], [maxx, maxy]]
    step:   float, separation between points
    
    Returns
    -------
    grid: (n, 2) list of 2D points
    '''
    x_grid = np.arange(*bounds[:, 0], step=step)
    y_grid = np.arange(*bounds[:, 1], step=step)
    grid = np.dstack(np.meshgrid(x_grid, y_grid)).reshape((-1, 2))
    return grid


def grid_linspace_2D(bounds, count):
    '''
    Return a n_sec_minor*n_sec_minor 2D grid

    Arguments
    ---------
    bounds: (2,2) list of [[minx, miny], [maxx, maxy]]
    count:  int, number of elements on a side
    
    Returns
    -------
    grid: (n_sec_minor**2, 2) list of 2D points
    '''
    x_grid = np.linspace(*bounds[:, 0], count=count)
    y_grid = np.linspace(*bounds[:, 1], count=count)
    grid = np.dstack(np.meshgrid(x_grid, y_grid)).reshape((-1, 2))
    return grid


def grid_linspace(bounds, count):
    """
    Return a grid spaced inside a bounding box with edges spaced using np.linspace.

    Parameters
    ------------
    bounds: (2,dimension) list of [[min x, min y, etc], [max x, max y, etc]]
    count:  int, or (dimension,) int, number of samples per side

    Returns
    ---------
    grid: (n, dimension) float, points in the specified bounds
    """
    bounds = np.asanyarray(bounds, dtype=np.float64)
    if len(bounds) != 2:
        raise ValueError('bounds must be (2, dimension!')
    count = np.asanyarray(count, dtype=np.int64)
    if count.shape == ():
        count = np.tile(count, bounds.shape[1])
    grid_elements = [np.linspace(*b, num=c) for b, c in zip(bounds.T, count)]
    grid = np.vstack(np.meshgrid(*grid_elements, indexing='ij')).reshape(bounds.shape[1], -1).T
    return grid


def replace_references(data, reference_dict):
    # Replace references in place
    view = np.array(data).view().reshape((-1))
    for i, value in enumerate(view):
        if value in reference_dict:
            view[i] = reference_dict[value]
    return view


def multi_dict(pairs):
    '''
    Given a set of key value pairs, create a dictionary. 
    If a key occurs multiple times, stack the values into an array.

    Can be called like the regular dict(pairs) constructor

    Arguments
    ----------
    pairs: (n,2) array of key, value pairs

    Returns
    ----------
    result: dict, with all values stored (rather than last with regular dict)

    '''
    result = defaultdict(list)
    for k, v in pairs:
        result[k].append(v)
    return result


def tolist_dict(data):
    def tolist(item):
        if hasattr(item, 'tolist'):
            return item.tolist()
        else:
            return item

    result = {k: tolist(v) for k, v in data.items()}
    return result


def is_binary_file(file_obj, probe_sz=1024):
    '''
    Returns True if file has non-ASCII characters (> 0x7F, or 127)
    Should work in both Python 2 and 3
    '''
    try:
        start = file_obj.tell()
        fbytes = file_obj.read(probe_sz)
        file_obj.seek(start)
        is_str = isinstance(fbytes, str)
        for fbyte in fbytes:
            if is_str:
                code = ord(fbyte)
            else:
                code = fbyte
            if code > 127: return True
    except UnicodeDecodeError:
        return True
    return False


def decimal_to_digits(decimal, min_digits=None):
    digits = abs(int(np.log10(decimal)))
    if min_digits is not None:
        digits = np.clip(digits, min_digits, 20)
    return digits


def md5_object(obj):
    '''
    If an object is hashable, return the hex string of the MD5.
    '''
    hasher = hashlib.md5()
    hasher.update(obj)
    hashed = hasher.hexdigest()
    return hashed


def attach_to_log(log_level=logging.DEBUG,
                  blacklist=['TerminalIPythonApp', 'PYREADLINE']):
    '''
    Attach a stream handler to all loggers.
    '''
    try:
        from colorlog import ColoredFormatter
        formatter = ColoredFormatter(
            ("%(log_color)s%(levelname)-8s%(reset)s " +
             "%(filename)17s:%(lineno)-4s  %(blue)4s%(message)s"),
            datefmt=None,
            reset=True,
            log_colors={'DEBUG': 'cyan',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'red'})
    except ImportError:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s",
            "%Y-%m-%d %H:%M:%S")
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    handler_stream.setLevel(log_level)

    for logger in logging.Logger.manager.loggerDict.values():
        if (logger.__class__.__name__ != 'Logger' or
                logger.name in blacklist):
            continue
        logger.addHandler(handler_stream)
        logger.setLevel(log_level)
    np.set_printoptions(precision=5, suppress=True)


def tracked_array(array, dtype=None):
    '''
    Properly subclass a numpy ndarray to track changes. 
    '''
    result = np.ascontiguousarray(array).view(TrackedArray)
    if dtype is None:
        return result
    return result.astype(dtype)


class TrackedArray(np.ndarray):
    '''
    Track changes in a numpy ndarray.

    Methods
    ----------
    md5: returns hexadecimal string of md5 of array
    '''

    def __array_finalize__(self, obj):
        '''
        Sets a modified flag on every TrackedArray
        This flag will be set on every change, as well as during copies
        and certain types of slicing. 
        '''
        self._modified = True
        if isinstance(obj, type(self)):
            obj._modified = True

    def md5(self):
        '''
        Return an MD5 hash of the current array in hexadecimal string form. 
        
        This is quite fast; on a modern i7 desktop a (1000000,3) floating point 
        array was hashed reliably in .03 seconds. 
        
        This is only recomputed if a modified flag is set which may have false 
        positives (forcing an unnecessary recompute) but will not have false 
        negatives which would return an incorrect hash. 
        '''

        if self._modified or not hasattr(self, '_hashed'):
            self._hashed = md5_object(self)
        self._modified = False
        return self._hashed

    def __hash__(self):
        '''
        Hash is required to return an int, so we convert the hex string to int.
        '''
        return int(self.md5(), 16)

    def __setitem__(self, i, y):
        self._modified = True
        super(self.__class__, self).__setitem__(i, y)

    def __setslice__(self, i, j, y):
        self._modified = True
        super(self.__class__, self).__setslice__(i, j, y)


class Cache(object):
    """
    Class to cache values until an id function changes.
    """

    def __init__(self, id_function=None):
        if id_function is None:
            self._id_function = lambda: None
        else:
            self._id_function = id_function
        self.id_current = None
        self._lock = 0
        self.cache = {}

    def decorator(self, function):
        name = function.__name__
        if name in self.cache:
            return self.cache[name]
        result = function()
        self.cache[name] = result
        return result

    def get(self, key):
        """
        Get a key from cache.
        If the key is unavailable or the cache has been invalidated returns None.
        :param key:
        :return:
        author: revised by weiwei
        date: 20201201
        """
        self.verify()
        return self.cache[key] if key in self.cache else None

    def verify(self):
        """
        Verify that the cached values are still for the same value of id_function, and delete all stored items if the
         value of id_function has changed.
        :return:
        author: revised by weiwei
        date: 20201201
        """
        id_new = self._id_function()
        if (self._lock == 0) and (id_new != self.id_current):
            if len(self.cache) > 0:
                log.debug('%d items cleared from cache: %s', len(self.cache), str(self.cache.keys()))
            self.clear()
            self.id_set()

    def clear(self, exclude=None):
        """
        Remove all elements in the cache.
        :param exclude:
        :return:
        author: revised by weiwei
        date: 20201201
        """
        if exclude is None:
            self.cache = {}
        else:
            self.cache = {k: v for k, v in self.cache.items() if k in exclude}

    def update(self, items):
        """
        Update the cache with a set of key, value pairs without checking id_function.
        :param items:
        :return:
        author: revised by weiwei
        date: 20201201
        """
        self.cache.update(items)
        self.id_set()

    def id_set(self):
        self.id_current = self._id_function()

    def set(self, key, value):
        self.verify()
        self.cache[key] = value
        return value

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        return self.set(key, value)

    def __contains__(self, key):
        self.verify()
        return key in self.cache

    def __enter__(self):
        self._lock += 1

    def __exit__(self, *args):
        self._lock -= 1
        self.id_current = self._id_function()


class DataStore(Mapping):

    def __init__(self):
        self.data = {}

    def __iter__(self):
        return iter(self.data)

    def __delitem__(self, key):
        del self.data[key]

    @property
    def mutable(self):
        """
        Is data allowed to be altered or not.
        :return: bool, can data be altered in the DataStore
        """
        if not hasattr(self, '_mutable'):
            return True
        return self._mutable

    @mutable.setter
    def mutable(self, value):
        value = bool(value)
        for i in self.data.value():
            i.flags.writeable = value
        self._mutable = value

    def is_empty(self):
        if len(self.data) == 0:
            return True
        for v in self.data.values():
            if is_sequence(v):
                if len(v) == 0:
                    return True
                else:
                    return False
            else:
                if bool(np.isreal(v)):
                    return False
        return True

    def clear(self):
        self.data = {}

    def __getitem__(self, key):
        if not self.mutable:
            raise ValueError('DataStore is configured immutable!')
        try:
            return self.data[key]
        except KeyError:
            return None

    def __setitem__(self, key, data):
        self.data[key] = tracked_array(data)

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def values(self):
        return self.data.values()

    def md5(self):
        # md5 = ''
        # for key in np.sort(list(self.data.keys())):
        #     md5 += self.data[key].md5()
        # return md5
        hasher = hashlib.md5()
        for key in sorted(self.data.keys()):
            hasher.update(self.data[key].md5().encode('utf-8'))
        md5 = hasher.hexdigest()
        return md5


def stack_lines(indices):
    return np.column_stack((indices[:-1],
                            indices[1:])).reshape((-1, 2))


def append_faces(vertices_seq, faces_seq):
    '''
    Given a sequence of zero- indexed faces and vertices,
    combine them into a single (n,3) list of faces and (m,3) vertices

    Arguments
    ---------
    vertices_seq: (n) sequence of (m,d) vertex arrays
    faces_seq     (n) sequence of (p,j) faces, zero indexed
                  and referencing their counterpoint vertices

    '''
    vertices_len = np.array([len(i) for i in vertices_seq])
    face_offset = np.append(0, np.cumsum(vertices_len)[:-1])

    for offset, faces in zip(face_offset, faces_seq):
        faces += offset

    vertices = np.vstack(vertices_seq)
    faces = np.vstack(faces_seq)

    return vertices, faces


def array_to_encoded(array, dtype=None, encoding='base64'):
    '''
    Export a numpy array to a compact serializable dictionary.

    Arguments
    ---------
    array: numpy array
    dtype: optional, what dtype should array be encoded with.
    encoding: str, 'base64' or 'binary'
    
    Returns
    ---------
    encoded: dict with keys: 
                 dtype: string of dtype
                 shape: int tuple of shape
                 base64: base64 encoded string of flat array
    '''
    array = np.asanyarray(array)
    shape = array.shape
    # ravel also forces contiguous
    flat = np.ravel(array)
    if dtype is None:
        dtype = array.dtype

    encoded = {'dtype': np.dtype(dtype).str,
               'shape': shape}
    if encoding in ['base64', 'dict64']:
        packed = base64.b64encode(flat.astype(dtype))
        if hasattr(packed, 'decode'):
            packed = packed.decode('utf-8')
        encoded['base64'] = packed
    elif encoding == 'binary':
        encoded['binary'] = array.tostring(order='C')
    else:
        raise ValueError('encoding {} is not available!'.format(encoding))
    return encoded


def encoded_to_array(encoded):
    '''
    Turn a dictionary with base64 encoded strings back into a numpy array.

    Arguments
    ----------
    encoded: dict with keys: 
                 dtype: string of dtype
                 shape: int tuple of shape
                 base64: base64 encoded string of flat array
                 binary:  decode result coming from numpy.tostring 
    Returns
    ----------
    array: numpy array
    '''
    shape = encoded['shape']
    dtype = np.dtype(encoded['dtype'])
    if 'base64' in encoded:
        array = np.fromstring(base64.b64decode(encoded['base64']), dtype).reshape(shape)
    elif 'binary' in encoded:
        array = np.fromstring(encoded['binary'],
                              dtype=dtype,
                              count=np.product(shape))
    array = array.reshape(shape)
    return array


def is_instance_named(obj, name):
    '''
    Given an object, if it is a member of the class 'name',
    or a subclass of 'name', return True.

    Arguments
    ---------
    obj: instance of a class
    name: string

    Returns
    ---------
    bool, whether the object is a member of the named class
    '''
    try:
        type_named(obj, name)
        return True
    except ValueError:
        return False


def type_bases(obj, depth=4):
    '''
    Return the bases of the object passed.
    '''
    bases = deque([list(obj.__class__.__bases__)])
    for i in range(depth):
        bases.append([i.__base__ for i in bases[-1] if i is not None])
    try:
        bases = np.hstack(bases)
    except IndexError:
        bases = []
    # we do the hasattr as None/NoneType can be in the list of bases
    bases = [i for i in bases if hasattr(i, '__name__')]
    return np.array(bases)


def type_named(obj, name):
    '''
    Similar to the end_type() builtin, but looks in class bases for named instance.

    Arguments
    ----------
    obj: object to look for class of
    name : str, name of class

    Returns
    ----------
    named class, or None
    '''
    # if obj is a member of the named class, return True
    name = str(name)
    if obj.__class__.__name__ == name:
        return obj.__class__
    for base in type_bases(obj):
        if base.__name__ == name:
            return base
    raise ValueError('Unable to extract class of name ' + name)


def concatenate(a, b=None):
    """
    Concatenate two meshes.
    :param a: Trimesh object
    :param b: Trimesh object
    :return: Trimesh object containing all faces of a and b
    author: weiwei
    date: 20210120
    """
    if b is None:
        b = []
        # stack meshes into flat list
    meshes = np.append(a, b)
    # if there is only one mesh just return the first
    if len(meshes) == 1:
        return meshes[0].copy()
    # extract the trimesh end_type to avoid a circular import
    # and assert that both inputs are Trimesh objects
    trimesh_type = type_named(meshes[0], 'Trimesh')
    # append faces and vertices of meshes
    vertices, faces = append_faces([m.vertices.copy() for m in meshes], [m.faces.copy() for m in meshes])
    # only save face normals if already calculated
    face_normals = None
    if all('face_normals' in m._cache for m in meshes):
        face_normals = np.vstack([m.face_normals for m in meshes])
    try:
        # concatenate visuals
        visual = meshes[0].visual.union([m.visual for m in meshes[1:]])
    except BaseException:
        log.warning('failed to combine visuals', exc_info=True)
        visual = None
    # create the mesh object
    mesh = trimesh_type(vertices=vertices,
                        faces=faces,
                        face_normals=face_normals,
                        visual=visual,
                        process=False)
    return mesh


def submesh(mesh,
            faces_sequence,
            only_watertight=False,
            append=False):
    '''
    Return a subset of a mesh.

    Arguments
    ----------
    mesh: Trimesh object
    faces_sequence: sequence of face indices from mesh
    only_watertight: only return submeshes which are watertight. 
    append: return a single mesh which has the faces specified appended.
            if this flag is set, only_watertight is ignored

    Returns
    ---------
    if append: Trimesh object
    else:      list of Trimesh objects
    '''
    # avoid nuking the cache on the original mesh
    original_faces = mesh.faces.view(np.ndarray)
    original_vertices = mesh.vertices.view(np.ndarray)

    faces = deque()
    vertices = deque()
    normals = deque()
    visuals = deque()

    # for reindexing faces
    mask = np.arange(len(original_vertices))

    for faces_index in faces_sequence:
        # sanitize indices in case they are coming in as a set or tuple
        faces_index = np.array(list(faces_index))
        faces_current = original_faces[faces_index]
        unique = np.unique(faces_current.reshape(-1))

        # redefine face indices from zero
        mask[unique] = np.arange(len(unique))

        normals.append(mesh.face_normals[faces_index])
        faces.append(mask[faces_current])
        vertices.append(original_vertices[unique])
        visuals.extend(mesh.visual.subsets([faces_index]))
    # we use end_type(mesh) rather than importing Trimesh from base
    # as this causes a circular import
    trimesh_type = type_named(mesh, 'Trimesh')
    if append:
        visuals = np.array(visuals)
        vertices, faces = append_faces(vertices, faces)
        appended = trimesh_type(vertices=vertices,
                                faces=faces,
                                face_normals=np.vstack(normals),
                                visual=visuals[0].union(visuals[1:]),
                                process=False)
        return appended
    result = [trimesh_type(vertices=v,
                           faces=f,
                           face_normals=n,
                           visual=c,
                           process=False) for v, f, n, c in zip(vertices,
                                                                faces,
                                                                normals,
                                                                visuals)]
    result = np.array(result)
    if only_watertight:
        watertight = np.array([i.fill_holes() and len(i.faces) > 4 for i in result])
        result = result[watertight]
    return result


def zero_pad(data, count, right=True):
    '''
    Arguments
    --------
    data: (n) axis_length 1D array
    count: int

    Returns
    --------
    padded: (n_sec_minor) axis_length 1D array if (n < n_sec_minor), otherwise axis_length (n)
    '''
    if len(data) == 0:
        return np.zeros(count)
    elif len(data) < count:
        padded = np.zeros(count)
        if right:
            padded[-len(data):] = data
        else:
            padded[:len(data)] = data
        return padded
    else:
        return np.asanyarray(data)


def format_json(data, digits=6):
    '''
    Function to turn a 1D float array into a json string

    The built in json library doesn't have a good way of setting the 
    precision of floating point numbers.

    Arguments
    ----------
    data: (n,) float array
    digits: int, number of digits of floating point numbers to include

    Returns
    ----------
    as_json: string, data formatted into a JSON- parsable string
    '''
    format_str = '.' + str(int(digits)) + 'f'
    as_json = '[' + ','.join(map(lambda o: format(o, format_str), data)) + ']'
    return as_json


class Words:
    '''
    A class to contain a list of words, such as the english language.
    The primary purpose is to create random keyphrases to be used to name
    things without resorting to giant hash strings.
    '''

    def __init__(self, file_name='/usr/share/dict/words', words=None):
        if words is None:
            self.words = np.loadtxt(file_name, dtype=str)
        else:
            self.words = np.array(words, dtype=str)

        self.words_simple = np.array([i.lower() for i in self.words if str.isalpha(i)])
        if len(self.words) == 0:
            log.warning('No words available!')

    def random_phrase(self, length=2, delimiter='-'):
        '''
        Create a random phrase using words containing only charecters. 

        Arguments
        ----------
        length:    int, how many words in phrase
        delimiter: str, what to separate words with

        Returns
        ----------
        phrase: str, axis_length words separated by delimiter

        Examples
        ----------
        In [1]: w = trimesh.util.Words()
        In [2]: for i in range(10): print w.random_phrase()
          ventilate-hindsight
          federating-flyover
          maltreat-patchiness
          puppets-remonstrated
          yoghourts-prut
          inventory-clench
          uncouple-bracket
          hipped-croupier
          puller-demesne
          phenomenally-hairs
        '''
        result = str(delimiter).join(np.random.choice(self.words_simple,
                                                      length))
        return result
