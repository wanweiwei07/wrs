# NOTE: This loader assumes no transformation in the dae file 20201207

import io
import uuid
import numpy as np
from .. import transform_points

# try:
    # pip install pycollada
from . import collada
# except BaseException:
#     collada = None
try:
    import PIL.Image
except ImportError:
    pass
from .. import util
from .. import visual
from ..constants import log


def load_collada(file_obj, resolver=None, **kwargs):
    """
    Load a COLLADA (.dae) file into a list of trimesh kwargs.
    Parameters
    ----------
    file_obj : file object
      Containing a COLLADA file
    resolver : trimesh.visual.Resolver or None
      For loading referenced files, like texture images
    kwargs : **
      Passed to trimesh.Trimesh.__init__
    Returns
    -------
    loaded : list of dict
      kwargs for Trimesh constructor
    """
    # load scene using pycollada
    c = collada.Collada(file_obj)
    # Create material map from Material ID to trimesh material
    material_map = {}
    # for m in c.materials:
    #     effect = m.effect
    #     material_map[m.id] = _parse_material(effect, resolver)
    # name : kwargs
    meshes = {}
    # list of dict
    graph = []
    for node in c.scene.nodes:
        _parse_node(node=node,
                    parent_matrix=np.eye(4),
                    material_map=material_map,
                    meshes=meshes,
                    graph=graph,
                    resolver=resolver)
    # create kwargs for load_kwargs
    # result = {'class': 'Scene',
    #           'graph': graph,
    #           'geometry': meshes}
    return list(meshes.values())


def export_collada(mesh, **kwargs):
    """
    Export a mesh or a list of meshes as a COLLADA .dae file.
    Parameters
    -----------
    mesh: Trimesh object or list of Trimesh objects
        The mesh(es) to export.
    Returns
    -----------
    export: str, string of COLLADA format output
    """
    meshes = mesh
    if not isinstance(mesh, (list, tuple, set, np.ndarray)):
        meshes = [mesh]
    c = collada.Collada()
    nodes = []
    for i, m in enumerate(meshes):
        # Load uv, colors, materials
        uv = None
        colors = None
        mat = _unparse_material(None)
        if m.visual.defined:
            if m.visual.kind == 'texture':
                mat = _unparse_material(m.visual.material)
                uv = m.visual.uv
            elif m.visual.kind == 'vertex':
                colors = (m.visual.vertex_colors / 255.0)[:, :3]
        c.effects.append(mat.effect)
        c.materials.append(mat)
        # Create geometry object
        vertices = collada.source.FloatSource(
            'vertices-array', m.vertices.flatten(), ('X', 'Y', 'Z'))
        normals = collada.source.FloatSource(
            'normals-array', m.vertex_normals.flatten(), ('X', 'Y', 'Z'))
        input_list = collada.source.InputList()
        input_list.addInput(0, 'VERTEX', '#vertices-array')
        input_list.addInput(1, 'NORMAL', '#normals-array')
        arrays = [vertices, normals]
        if uv is not None:
            texcoords = collada.source.FloatSource(
                'texcoords-array', uv.flatten(), ('U', 'V'))
            input_list.addInput(2, 'TEXCOORD', '#texcoords-array')
            arrays.append(texcoords)
        if colors is not None:
            idx = 2
            if uv:
                idx = 3
            colors = collada.source.FloatSource('colors-array', colors.flatten(), ('R', 'G', 'B'))
            input_list.addInput(idx, 'COLOR', '#colors-array')
            arrays.append(colors)
        geom = wrs.basis.trimesh.io.collada.geometry.Geometry(c, uuid.uuid4().hex, uuid.uuid4().hex, arrays)
        indices = np.repeat(m.faces.flatten(), len(arrays))
        matref = 'material{}'.format(i)
        triset = geom.createTriangleSet(indices, input_list, matref)
        geom.primitives.append(triset)
        c.geometries.append(geom)
        matnode = wrs.basis.trimesh.io.collada.scene.MaterialNode(matref, mat, inputs=[])
        geomnode = wrs.basis.trimesh.io.collada.scene.GeometryNode(geom, [matnode])
        node = wrs.basis.trimesh.io.collada.scene.Node('node{}'.format(i), children=[geomnode])
        nodes.append(node)
    scene = wrs.basis.trimesh.io.collada.scene.Scene('scene', nodes)
    c.scenes.append(scene)
    c.scene = scene
    b = io.BytesIO()
    c.write(b)
    b.seek(0)
    return b.read()


def _parse_node(node,
                parent_matrix,
                material_map,
                meshes,
                graph,
                resolver=None):
    """
    Recursively parse COLLADA scene nodes.
    """
    # Parse mesh node
    if isinstance(node, collada.scene.GeometryNode):
        geometry = node.geometry
        # Iterate over primitives of geometry
        for i, primitive in enumerate(geometry.primitives):
            if isinstance(primitive, collada.polylist.Polylist):
                primitive = primitive.triangleset()
            if isinstance(primitive, collada.triangleset.TriangleSet):
                vertices = primitive.vertex
                faces = primitive.vertex_index
                normal = primitive.normal
                vertex_normals = normal[primitive.normal_index]
                face_normals = (vertex_normals[:, 0, :] + vertex_normals[:, 1, :] + vertex_normals[:, 2, :]) / 3
                if not np.allclose(parent_matrix, np.eye(4), 1e-8):
                    vertices = transform_points(vertices, parent_matrix)
                    normalized_matrix = parent_matrix/np.linalg.norm(parent_matrix[:,0])
                    face_normals = transform_points(face_normals, normalized_matrix, translate=False)
                primid = '{}.{}'.format(geometry.id, i)
                meshes[primid] = {
                    'vertices': vertices,
                    'faces': faces,
                    'face_normals': face_normals}
                graph.append({'frame_to': primid,
                              'matrix': parent_matrix,
                              'geometry': primid})
    # recurse down tree for nodes with children
    elif isinstance(node, collada.scene.Node):
        if node.children is not None:
            for child in node.children:
                # create the new matrix
                matrix = np.dot(parent_matrix, node.matrix)
                # parse the child node
                _parse_node(
                    node=child,
                    parent_matrix=matrix,
                    material_map=material_map,
                    meshes=meshes,
                    graph=graph,
                    resolver=resolver)
    elif isinstance(node, collada.scene.CameraNode):
        # TODO: convert collada cameras to trimesh cameras
        pass
    elif isinstance(node, collada.scene.LightNode):
        # TODO: convert collada lights to trimesh lights
        pass


def _load_texture(file_name, resolver):
    """
    Load a texture from a file into a PIL image.
    """
    file_data = resolver.get(file_name)
    image = PIL.Image.open(util.wrap_as_stream(file_data))
    return image


def _parse_material(effect, resolver):
    """
    Turn a COLLADA effect into a trimesh material.
    """
    # Compute base color
    baseColorFactor = np.ones(4)
    baseColorTexture = None
    if isinstance(effect.diffuse, wrs.basis.trimesh.io.collada.material.Map):
        try:
            baseColorTexture = _load_texture(
                effect.diffuse.sampler.surface.image.path, resolver)
        except BaseException:
            log.warning('unable to load base texture',
                        exc_info=True)
    elif effect.diffuse is not None:
        baseColorFactor = effect.diffuse
    # Compute emission color
    emissiveFactor = np.zeros(3)
    emissiveTexture = None
    if isinstance(effect.emission, wrs.basis.trimesh.io.collada.material.Map):
        try:
            emissiveTexture = _load_texture(
                effect.diffuse.sampler.surface.image.path, resolver)
        except BaseException:
            log.warning('unable to load emissive texture',
                        exc_info=True)
    elif effect.emission is not None:
        emissiveFactor = effect.emission[:3]
    # Compute roughness
    roughnessFactor = 1.0
    if (not isinstance(effect.shininess, wrs.basis.trimesh.io.collada.material.Map)
            and effect.shininess is not None):
        roughnessFactor = np.sqrt(2.0 / (2.0 + effect.shininess))
    # Compute metallic factor
    metallicFactor = 0.0
    # Compute normal texture
    normalTexture = None
    if effect.bumpmap is not None:
        try:
            normalTexture = _load_texture(
                effect.bumpmap.sampler.surface.image.path, resolver)
        except BaseException:
            log.warning('unable to load bumpmap',
                        exc_info=True)
    return visual.material.PBRMaterial(
        emissiveFactor=emissiveFactor,
        emissiveTexture=emissiveTexture,
        normalTexture=normalTexture,
        baseColorTexture=baseColorTexture,
        baseColorFactor=baseColorFactor,
        metallicFactor=metallicFactor,
        roughnessFactor=roughnessFactor
    )


def _unparse_material(material):
    """
    Turn a trimesh material into a COLLADA material.
    """
    # TODO EXPORT TEXTURES
    if isinstance(material, visual.material.PBRMaterial):
        diffuse = material.baseColorFactor
        if diffuse is not None:
            diffuse = list(diffuse)
        emission = material.emissiveFactor
        if emission is not None:
            emission = [float(emission[0]), float(emission[1]), float(emission[2]), 1.0]
        shininess = material.roughnessFactor
        if shininess is not None:
            shininess = 2.0 / shininess ** 2 - 2.0
        effect = wrs.basis.trimesh.io.collada.material.Effect(
            uuid.uuid4().hex, params=[], shadingtype='phong',
            diffuse=diffuse, emission=emission,
            specular=[1.0, 1.0, 1.0, 1.0], shininess=float(shininess)
        )
        material = wrs.basis.trimesh.io.collada.material.Material(uuid.uuid4().hex, 'pbrmaterial', effect)
    else:
        effect = wrs.basis.trimesh.io.collada.material.Effect(uuid.uuid4().hex, params=[], shadingtype='phong')
        material = wrs.basis.trimesh.io.collada.material.Material(uuid.uuid4().hex, 'defaultmaterial', effect)
    return material


def load_zae(file_obj, resolver=None, **kwargs):
    """
    Load a ZAE file, which is just a zipped DAE file.
    Parameters
    -------------
    file_obj : file object
      Contains ZAE data
    resolver : trimesh.visual.Resolver
      Resolver to load additional assets
    kwargs : dict
      Passed to load_collada
    Returns
    ------------
    loaded : dict
      Results of loading
    """
    # a dict, {file name : file object}
    archive = util.decompress(file_obj, file_type='zip')
    # load the first file with a .dae extension
    file_name = next(i for i in archive.keys() if i.lower().endswith('.dae'))
    # a resolver so the loader can load textures / etc
    resolver = visual.resolvers.ZipResolver(archive)
    # run the regular collada loader
    loaded = load_collada(archive[file_name], resolver=resolver, **kwargs)
    return loaded


# only provide loaders if `pycollada` is installed
_collada_loaders = {}
_collada_exporters = {}
if collada is not None:
    _collada_loaders['dae'] = load_collada
    _collada_loaders['zae'] = load_zae
    _collada_exporters['dae'] = export_collada
