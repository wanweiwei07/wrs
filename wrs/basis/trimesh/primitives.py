import numpy as np

from . import util
from . import points
from . import creation

from .base import Trimesh
from .constants import log
from .triangles import windings_aligned


class Primitive(Trimesh):
    '''
    Geometric primitives which are a subclass of Trimesh.
    Mesh is generated lazily when vertices or faces are requested.
    '''

    def __init__(self, *args, **kwargs):
        super(Primitive, self).__init__(*args, **kwargs)
        self._data.clear()
        self._validate = False

    @property
    def faces(self):
        stored = self._cache['faces']
        if util.is_shape(stored, (-1, 3)):
            return stored
        self._create_mesh()
        # self._validate_face_normals()
        return self._cache['faces']

    @faces.setter
    def faces(self, values):
        log.warning('Primitive faces are immutable! Not setting!')

    @property
    def vertices(self):
        stored = self._cache['vertices']
        if util.is_shape(stored, (-1, 3)):
            return stored

        self._create_mesh()
        return self._cache['vertices']

    @vertices.setter
    def vertices(self, values):
        if values is not None:
            log.warning('Primitive vertices are immutable! Not setting!')

    @property
    def face_normals(self):
        stored = self._cache['face_normals']
        if util.is_shape(stored, (-1, 3)):
            return stored
        self._create_mesh()
        return self._cache['face_normals']

    @face_normals.setter
    def face_normals(self, values):
        if values is not None:
            log.warning('Primitive face normals are immutable! Not setting!')

    def _create_mesh(self):
        raise ValueError('Primitive doesn\'t define mesh creation!')


class Sphere(Primitive):
    def __init__(self, *args, **kwargs):
        '''
        Create a Sphere primitive, which is a subclass of Trimesh.

        Arguments
        ----------
        sphere_radius: float, major_radius of sphere
        sphere_center: (3,) float, center of sphere
        subdivisions: int, number of sphere_ico_level for icosphere. Default is 3
        '''
        super(Sphere, self).__init__(*args, **kwargs)
        if 'radius' in kwargs:
            self.sphere_radius = kwargs['radius']
        if 'center' in kwargs:
            self.sphere_center = kwargs['center']
        if 'ico_level' in kwargs:
            self._data['ico_level'] = int(kwargs['ico_level'])
        else:
            self._data['ico_level'] = 3
        self._unit_sphere = creation.icosphere(subdivisions=self._data['ico_level'][0])

    @property
    def sphere_center(self):
        stored = self._data['center']
        if stored is None:
            return np.zeros(3)
        return stored

    @sphere_center.setter
    def sphere_center(self, values):
        self._data['center'] = np.asanyarray(values, dtype=np.float64)

    @property
    def sphere_radius(self):
        stored = self._data['radius']
        if stored is None:
            return 1.0
        return stored

    @sphere_radius.setter
    def sphere_radius(self, value):
        self._data['radius'] = float(value)

    def _create_mesh(self):
        ico = self._unit_sphere
        self._cache['vertices'] = ((ico.vertices * self.sphere_radius) +
                                   self.sphere_center)
        self._cache['faces'] = ico.faces
        self._cache['face_normals'] = ico.face_normals


class Box(Primitive):
    def __init__(self, *args, **kwargs):
        """
        Create a Box primitive, which is a subclass of Trimesh
        :param kwargs:
                extents:   (3,) float, size of box
                homomat:   (4,4) float, transformation matrix for box
                center:    (3,) float, convience function which updates box_transform
                               with a translation- only matrix
        """
        super(Box, self).__init__(*args, **kwargs)
        if 'extents' in kwargs:
            self.extents = kwargs['extents']
        if 'homomat' in kwargs:
            self.homomat = kwargs['homomat']
        if 'center' in kwargs:
            self.center = kwargs['center']
        self._unit_box = creation.box()

    @property
    def center(self):
        return self.homomat[0:3, 3]

    @center.setter
    def center(self, values):
        transform = self.homomat
        transform[:3, 3] = values
        self._data['homomat'] = transform

    @property
    def extents(self):
        stored = self._data['extents']
        if util.is_shape(stored, (3,)):
            return stored
        return np.ones(3)

    @extents.setter
    def extents(self, values):
        self._data['extents'] = np.asanyarray(values, dtype=np.float64)

    @property
    def homomat(self):
        stored = self._data['homomat']
        if util.is_shape(stored, (4, 4)):
            return stored
        return np.eye(4)

    @homomat.setter
    def homomat(self, matrix):
        matrix = np.asanyarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError('Matrix must be (4,4)!')
        self._data['homomat'] = matrix

    @property
    def is_oriented(self):
        if util.is_shape(self.homomat, (4, 4)):
            return not np.allclose(self.homomat[:3, :3], np.eye(3))
        else:
            return False

    def _create_mesh(self):
        log.debug('Creating mesh for box primitive')
        box = self._unit_box
        vertices, faces, normals = box.vertices, box.faces, box.face_normals
        vertices = points.transform_points(vertices * self.extents, self.homomat)
        normals = np.dot(self.homomat[:3, :3], normals.T).T
        aligned = windings_aligned(vertices[faces[:1]], normals[:1])[0]
        if not aligned:
            faces = np.fliplr(faces)
            # for a primitive the vertices and faces are derived from other information
        # so it goes in the cache, instead of the datastore
        self._cache['vertices'] = vertices
        self._cache['faces'] = faces
        self._cache['face_normals'] = normals


class Cylinder(Primitive):

    def __init__(self, *args, **kwargs):
        """
        Create a Cylinder Primitive, a subclass of Trimesh.
        Parameters
        -------------
        radius : float
          Radius of cylinder
        height : float
          Height of cylinder
        n_sec : int
          Number of facets in circle
        """
        super(Cylinder, self).__init__(*args, **kwargs)
        if 'height' in kwargs:
            self.height = kwargs['height']
        else:
            self.height = 10.0
        if 'radius' in kwargs:
            self.radius = kwargs['radius']
        else:
            self.radius = 1.0
        if 'n_sec' in kwargs:
            self.n_sec = kwargs['n_sec']
        else:
            self.n_sec = 12
        if 'homomat' in kwargs:
            self.homomat = kwargs['homomat']
        else:
            self.homomat = None

    def volume(self):
        """
        The analytic volume of the cylinder primitive.
        Returns
        ---------
        volume : float
          Volume of the cylinder
        """
        volume = ((np.pi * self.radius ** 2) * self.height)
        return volume

    def buffer(self, distance):
        """
        Return a cylinder primitive which covers the source cylinder
        by linear_distance: major_radius is inflated by linear_distance, height by twice
        the linear_distance.
        Parameters
        ------------
        distance : float
          Distance to inflate cylinder major_radius and height
        Returns
        -------------
        buffered : Cylinder
         Cylinder primitive inflated by linear_distance
        """
        distance = float(distance)
        buffered = Cylinder(height=self.height + distance * 2, radius=self.radius + distance, n_sec=self.n_sec,
                            homomat=self.homomat)
        return buffered

    def _create_mesh(self):
        log.debug('Creating cylinder mesh with r=%f, h=%f %d n_sec_minor', self.radius, self.height, self.n_sec)
        mesh = creation.cylinder(radius=self.radius, height=self.height, n_sec=self.n_sec, homomat=self.homomat)
        self._cache['vertices'] = mesh.vertices
        self._cache['faces'] = mesh.faces
        self._cache['face_normals'] = mesh.face_normals


class Extrusion(Primitive):
    def __init__(self, *args, **kwargs):
        '''
        Create an Extrusion primitive, which subclasses Trimesh

        Arguments
        ----------
        extrude_polygon:   shapely.geometry.Polygon, polygon to extrude
        extrude_transform: (4,4) float, transform to apply after extrusion
        extrude_height:    float, height to extrude polygon by
        '''
        super(Extrusion, self).__init__(*args, **kwargs)
        if 'extrude_polygon' in kwargs:
            self.extrude_polygon = kwargs['extrude_polygon']
        if 'extrude_transform' in kwargs:
            self.extrude_transform = kwargs['extrude_transform']
        if 'extrude_height' in kwargs:
            self.extrude_height = kwargs['extrude_height']

    @property
    def extrude_transform(self):
        stored = self._data['extrude_transform']
        if np.shape(stored) == (4, 4):
            return stored
        return np.eye(4)

    @extrude_transform.setter
    def extrude_transform(self, matrix):
        matrix = np.asanyarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError('Matrix must be (4,4)!')
        self._data['extrude_transform'] = matrix

    @property
    def extrude_height(self):
        stored = self._data['extrude_height']
        if stored is None:
            raise ValueError('extrude height not specified!')
        return stored.copy()[0]

    @extrude_height.setter
    def extrude_height(self, value):
        self._data['extrude_height'] = float(value)

    @property
    def extrude_polygon(self):
        stored = self._data['extrude_polygon']
        if stored is None:
            raise ValueError('extrude polygon not specified!')
        return stored[0]

    @extrude_polygon.setter
    def extrude_polygon(self, value):
        polygon = creation.validate_polygon(value)
        self._data['extrude_polygon'] = polygon

    @property
    def extrude_direction(self):
        direction = np.dot(self.extrude_transform[:3, :3],
                           [0.0, 0.0, 1.0])
        return direction

    def slide(self, distance):
        distance = float(distance)
        translation = np.eye(4)
        translation[2, 3] = distance
        new_transform = np.dot(self.extrude_transform.copy(),
                               translation.copy())
        self.extrude_transform = new_transform

    def _create_mesh(self):
        log.debug('Creating mesh for extrude primitive')
        mesh = creation.extrude_polygon(self.extrude_polygon,
                                        self.extrude_height)
        mesh.apply_transform(self.extrude_transform)
        self._cache['vertices'] = mesh.vertices
        self._cache['faces'] = mesh.faces
        self._cache['face_normals'] = mesh.face_normals


class Cone(Primitive):

    def __init__(self, *args, **kwargs):
        """
        Create a Cone Primitive, a subclass of Trimesh.
        Parameters

        :param radius : float, major_radius of cone
        :param height : float, height of cone
        :param n_sec : int, number of facets in circle

        author: weiwei
        date: 20191228
        """

        super(Cone, self).__init__(*args, **kwargs)
        if 'height' in kwargs:
            self.height = kwargs['height']
        else:
            self.height = 10.0
        if 'radius' in kwargs:
            self.radius = kwargs['radius']
        else:
            self.radius = 1.0
        if 'n_sec' in kwargs:
            self.n_sec = kwargs['n_sec']
        else:
            self.n_sec = 12
        if 'homomat' in kwargs:
            self.homomat = kwargs['homomat']
        else:
            self.homomat = None

    def volume(self):
        """
        The analytic volume of the cylinder primitive.

        :return volume, float, volume of the cylinder

        author: weiwei
        date: 20191228osaka
        """
        volume = ((np.pi * self.radius ** 2) * self.height) / 3
        return volume

    def buffer(self, distance):
        """
        Return a cylinder primitive which covers the source cone by linear_distance:
        major_radius is inflated by linear_distance, height by twice the linear_distance.

        :param distance: float, linear_distance to inflate cylinder major_radius and height
        :return buffered: cone primitive inflated by linear_distance

        author: weiwei
        date: 20191228osaka
        """
        distance = float(distance)

        buffered = Cone(height=self.height + distance * 2, radius=self.radius + distance, n_sec=self.n_sec,
                        homomat=self.homomat)
        return buffered

    def _create_mesh(self):
        log.debug('Creating cylinder mesh with r=%f, h=%f %d n_sec', self.radius, self.height, self.n_sec)
        mesh = creation.cone(radius=self.radius, height=self.height, n_sec=self.n_sec, homomat=self.homomat)
        self._cache['vertices'] = mesh.vertices
        self._cache['faces'] = mesh.faces
        self._cache['face_normals'] = mesh.face_normals


class Capsule(Primitive):

    def __init__(self, *args, **kwargs):
        """
        Create a Capsule Primitive, a subclass of Trimesh.
        Parameters

        :param radius : float, major_radius of cone
        :param height : float, height of cone
        :param count : [int, int], number of secs in log and alt
        :param homomat: 4x4 transformation matrix

        author: weiwei
        date: 20191228
        """

        super(Capsule, self).__init__(*args, **kwargs)
        if 'height' in kwargs:
            self.height = kwargs['height']
        else:
            self.height = 10.0
        if 'radius' in kwargs:
            self.radius = kwargs['radius']
        else:
            self.radius = 1.0
        if 'count' in kwargs:
            self.count = kwargs['count']
        else:
            self.count = [8, 8]
        if 'homomat' in kwargs:
            self.homomat = kwargs['homomat']
        else:
            self.homomat = None

    def volume(self):
        """
        The analytic volume of the cylinder primitive.

        :return volume, float, volume of the cylinder

        author: weiwei
        date: 20191228osaka
        """
        volume = (np.pi * self.radius ** 3) * 3 / 4 + (np.pi * self.radius ** 2) * self.height
        return volume

    def buffer(self, distance):
        """
        Return a capsule primitive which covers the source capsule by linear_distance:

        :param distance: float, linear_distance to inflate cylinder major_radius and height
        :return buffered: cone primitive inflated by linear_distance

        author: weiwei
        date: 20191228osaka
        """
        distance = float(distance)

        buffered = Capsule(height=self.height + distance * 2, radius=self.radius + distance, count=self.count,
                           homomat=self.homomat)
        return buffered

    def _create_mesh(self):
        log.debug('Creating cylinder mesh with r=%f, h=%f %d count', self.radius, self.height, self.count)
        mesh = creation.capsule(radius=self.radius, height=self.height, count=self.count, homomat=self.homomat)
        self._cache['vertices'] = mesh.vertices
        self._cache['faces'] = mesh.faces
        self._cache['face_normals'] = mesh.face_normals
