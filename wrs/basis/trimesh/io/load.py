import numpy as np
from ..base import Trimesh
from ..constants import _log_time, log
from ..util import is_file, is_string, make_sequence, is_instance_named, concatenate
from .assimp import _assimp_loaders
from .stl import _stl_loaders
from .misc import _misc_loaders
from .step import _step_loaders
from .ply import _ply_loaders
from .dae import _collada_loaders

try:
    from ..path.io.load import load_path, path_formats
except:
    log.warning('No path functionality available!', exc_info=True)


def load_path(*args, **kwargs):
    raise ImportError('No path functionality available!')


def path_formats():
    return []


def mesh_formats():
    return list(_mesh_loaders.keys())


def available_formats():
    return np.append(mesh_formats(), path_formats())


def load(obj, file_type=None, **kwargs):
    """
    Load a mesh or vectorized path into a Trimesh, Path2D, or Path3D object.
    :param obj: a filename string or a file-like object
    :param file_type: str representing file end_type (eg: 'meshes'), parsed from obj if None
    :param kwargs:
    :return: Trimesh, Path2D, Path3D, or list of same.
    author: revised by weiwei
    date: 20201206
    """
    if isinstance(obj, Trimesh):
        return obj
    if is_instance_named(obj, 'Path'):
        return obj
    if is_string(obj):
        file_type = (str(obj).split('.')[-1]).lower()
        obj = open(obj, 'rb')
    if file_type is None:
        file_type = obj.__class__.__name__
    if file_type in path_formats():
        return load_path(obj, file_type, **kwargs)
    elif file_type in mesh_formats():
        return load_mesh(obj, file_type, **kwargs)
    raise ValueError('File end_type: %s not supported', str(file_type))


@_log_time
def load_mesh(obj, file_type=None, process=True):
    """
    Load a mesh file into a Trimesh object
    :param obj: a filename string or a file-like object
    :param file_type: filename string or a file-like object
    :param process: boolean flag, whether to process the mesh on load
    :return: 
    mesh: a single Trimesh object, or a list of Trimesh objects, 
          depending on the file format.
    author: revised by weiwei
    date: 20220212
    """
    if is_string(obj):
        file_type = (str(obj).split('.')[-1]).lower()
        obj = open(obj, 'rb')
    if file_type is None:
        file_type = obj.__class__.__name__
    file_type = str(file_type).lower()
    loaded = _mesh_loaders[file_type](obj, file_type)
    if is_file(obj):
        obj.close()
    log.debug('loaded mesh using %s', _mesh_loaders[file_type].__name__)
    meshes = [Trimesh(process=process, **i) for i in make_sequence(loaded)]
    # if len(meshes) == 1:
    #     return meshes[0]
    meshes = concatenate(meshes)
    return meshes


_mesh_loaders = {}
# assimp has a lot of loaders, but they are all quite slow
# so we load them first and replace them with native loaders if possible
_mesh_loaders.update(_assimp_loaders)
_mesh_loaders.update(_stl_loaders)
_mesh_loaders.update(_misc_loaders)
_mesh_loaders.update(_step_loaders)
_mesh_loaders.update(_ply_loaders)
_mesh_loaders.update(_collada_loaders)
