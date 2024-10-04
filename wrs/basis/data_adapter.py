# An adapter file that converts data between panda3d and trimesh
import numpy as np
from panda3d.core import Geom, GeomNode, GeomPoints, GeomTriangles
from panda3d.core import GeomVertexData, GeomVertexFormat, GeomVertexArrayFormat, InternalName
from panda3d.core import GeomEnums
from panda3d.core import NodePath, Vec3, Vec4, Mat3, Mat4, LQuaternion
import wrs.basis.trimesh as trm

# global constant for panda3d
M_TO_PIXEL = 3779.53  # panda3d meter to pixel


# data manipulation
def gen_colorarray(n_colors: int = 1,
                   alpha: float = 1,
                   seed_rgb: np.ndarray = None):
    """
    Generate an array of random colors
    :param n_colors: the number of colors genrated
    :return: 1x4 or nx4 nparray
    author: weiwei
    date: 20161130osaka, 20230811
    """
    if n_colors == 1:
        if seed_rgb:
            return np.asarray([seed_rgb[0], seed_rgb[1], seed_rgb[2], alpha])
        else:
            return np.asarray([np.random.random(), np.random.random(), np.random.random(), alpha])
    color_array = []
    for i in range(n_colors):
        if seed_rgb:
            color_array.append([seed_rgb[0], seed_rgb[1], seed_rgb[2], alpha])
        else:
            color_array.append([np.random.random(), np.random.random(), np.random.random(), alpha])
    return np.asarray(color_array)


def npmat3_to_pdmat3(npmat3: np.ndarray) -> Mat3:
    """
    convert numpy.2darray to LMatrix3f defined in Panda3d
    :param npmat3: a 3x3 numpy ndarray
    :return: a LMatrix3f object, see panda3d
    author: weiwei
    date: 20161107tsukuba
    """
    return Mat3(npmat3[0, 0], npmat3[1, 0], npmat3[2, 0], \
                npmat3[0, 1], npmat3[1, 1], npmat3[2, 1], \
                npmat3[0, 2], npmat3[1, 2], npmat3[2, 2])


def pdmat3_to_npmat3(pdmat3: Mat3) -> np.ndarray:
    """
    convert a mat3 matrix to a numpy 2darray...
    :param pdmat3:
    :return: numpy 2darray
    author: weiwei
    date: 20161216sapporo
    """
    row0 = pdmat3.getRow(0)
    row1 = pdmat3.getRow(1)
    row2 = pdmat3.getRow(2)
    print(row0)
    print(row1)
    print(row2)
    return np.array([[row0[0], row1[0], row2[0]], [row0[1], row1[1], row2[1]], [row0[2], row1[2], row2[2]]])


def npv3mat3_to_pdmat4(npvec3: np.ndarray = np.array([0, 0, 0]),
                       npmat3: np.ndarray = np.eye(3)):
    """
    convert numpy.2darray to LMatrix4 defined in Panda3d
    note the first parameter is rotmat, the second is pos
    since we want to use default values for the second param
    :param npmat3: a 3x3 numpy ndarray
    :param npvec3: a 1x3 numpy ndarray
    :return: a LMatrix3f object, see panda3d
    author: weiwei
    date: 20170322
    """
    return Mat4(npmat3[0, 0], npmat3[1, 0], npmat3[2, 0], 0, \
                npmat3[0, 1], npmat3[1, 1], npmat3[2, 1], 0, \
                npmat3[0, 2], npmat3[1, 2], npmat3[2, 2], 0, \
                npvec3[0], npvec3[1], npvec3[2], 1)


def npmat4_to_pdmat4(npmat4: np.ndarray) -> Mat4:
    """
    # updated from cvtMat4
    convert numpy.2darray to LMatrix4 defined in Panda3d
    :param npmat3: a 3x3 numpy ndarray
    :param npvec3: a 1x3 numpy ndarray
    :return: a LMatrix3f object, see panda3d
    author: weiwei
    date: 20170322
    """
    return Mat4(npmat4[0, 0], npmat4[1, 0], npmat4[2, 0], 0, \
                npmat4[0, 1], npmat4[1, 1], npmat4[2, 1], 0, \
                npmat4[0, 2], npmat4[1, 2], npmat4[2, 2], 0, \
                npmat4[0, 3], npmat4[1, 3], npmat4[2, 3], 1)


def pdmat4_to_npmat4(pdmat4: Mat4) -> np.ndarray:
    """
    convert a mat4 matrix to a nparray
    :param pdmat4
    :return: numpy 2darray
    author: weiwei
    date: 20161216sapporo
    """
    return np.array(pdmat4.getRows()).T


def pdmat4_to_npv3mat3(pdmat4: np.ndarray):
    """
    :param pdmat4:
    :return: pos, rotmat: 1x3 and 3x3 nparray
    author: weiwei
    date: 20200206
    """
    homo_npmat = np.array(pdmat4.getRows()).T
    return [homo_npmat[:3, 3], homo_npmat[:3, :3]]


def npmat3_to_pdquat(npmat3: np.ndarray) -> LQuaternion:
    """
    :param npmat3: 3x3 nparray
    :return:
    author: weiwei
    date: 20210109
    """
    quat = LQuaternion()
    quat.setFromMatrix(npmat3_to_pdmat3(npmat3))
    return quat


def pdquat_to_npmat3(pdquat: LQuaternion) -> np.ndarray:
    """
    :param pdquat: panda.core.LQuaternion
    :return:
    author: weiwei
    date: 20210109
    """
    tmp_pdmat3 = Mat3()
    pdquat.extractToMatrix(tmp_pdmat3)
    return pdmat3_to_npmat3(tmp_pdmat3)


def npvec3_to_pdvec3(npvec3: np.ndarray) -> Vec3:
    """
    convert a numpy array to Panda3d Vec3...
    :param npvec3:
    :return: panda3d vec3
    author: weiwei
    date: 20170322
    """
    return Vec3(*npvec3)


def pdvec3_to_npvec3(pdvec3: Vec3) -> np.ndarray:
    """
    convert vbase3 to a nprray...
    :param pdmat3:
    :return: numpy 2darray
    author: weiwei
    date: 20161216sapporo
    """
    return np.array([pdvec3[0], pdvec3[1], pdvec3[2]])


def npvec4_to_pdvec4(npvec4: np.ndarray) -> Vec4:
    """
    convert a numpy array to Panda3d Vec4...
    :param npvec4:
    :return: panda3d vec3
    author: weiwei
    date: 20170322
    """
    return Vec4(*npvec4)


def pdvec4_to_npvec4(npvec4: Vec4) -> np.ndarray:
    """
    convert vbase3 to a nparray
    :param pdmat3:
    :return: numpy 2darray
    author: weiwei
    date: 20161216sapporo
    """
    return np.array([npvec4[0], npvec4[1], npvec4[2], npvec4[3]])


def trimesh_to_nodepath(trm_mesh: trm.Trimesh, name="auto") -> NodePath:
    """
    cvt mesh models to panda models
    :param trm_mesh:
    :return:
    author: weiwei
    date: 20180606
    """
    return pdgeomndp_from_vfnf(trm_mesh.vertices, trm_mesh.face_normals, trm_mesh.faces, name=name)


def o3dmesh_to_nodepath(o3d_mesh, name="auto") -> NodePath:
    """
    cvt open3d mesh models to panda models
    :param o3d_mesh:
    :return:
    author: weiwei
    date: 20191210
    """
    return pdgeomndp_from_vfnf(o3d_mesh.vertices, o3d_mesh.triangle_normals, o3d_mesh.triangles, name=name)


def pdgeom_from_vfnf(vertices: np.ndarray,
                     face_normals: np.ndarray,
                     triangles: np.ndarray,
                     name: str = 'auto') -> Geom:
    """
    :param vertices: nx3 nparray, each row is vertex
    :param face_normals: nx3 nparray, each row is the normal of a face
    :param triangles: nx3 nparray, each row is three idx to the vertices
    :param name:
    :return: a geom model that is ready to be used to define a pdndp
    author: weiwei
    date: 20160613, 20210109, 20230811
    """
    # expand vertices to let each triangle refer to a different vert+normal
    # vertices and normals
    vertex_format = GeomVertexFormat.getV3n3()
    vertex_data = GeomVertexData(name, vertex_format, Geom.UHStatic)
    vertex_ids = triangles.flatten()
    multiplied_vertices = np.empty((len(vertex_ids), 3), dtype=np.float32)
    multiplied_vertices[:] = vertices[vertex_ids]
    vertex_normals = np.repeat(face_normals.astype(np.float32), repeats=3, axis=0)
    np_bytes = np.hstack((multiplied_vertices, vertex_normals)).tobytes()
    vertex_data.modifyArrayHandle(0).setData(np_bytes)
    # triangles
    primitive = GeomTriangles(Geom.UHStatic)
    primitive.setIndexType(GeomEnums.NTUint32)
    multiplied_triangles = np.arange(len(vertex_ids), dtype=np.uint32).reshape(-1, 3)
    primitive.modifyVertices(-1).modifyHandle().setData(multiplied_triangles.tobytes())
    # make geom
    pedgeom = Geom(vertex_data)
    pedgeom.addPrimitive(primitive)
    return pedgeom


def pdgeomndp_from_vfnf(vertices: np.ndarray,
                        face_normals: np.ndarray,
                        triangles: np.ndarray,
                        name: str = 'auto') -> NodePath:
    """
    pack the given vertices and triangles into a panda3d geom, vf = vertices faces
    :param vertices: nx3 nparray, each row is a vertex
    :param face_normals: nx3 nparray, each row is the normal of a face
    :param triangles: nx3 nparray, each row is three idx to the vertices
    :param name:
    :return: pdndp (Panda3D NodePath), ndp is used to avoid confusion with np (numpy abbreviation)
    author: weiwei
    date: 20170221, 20210109, 20230811
    """
    pdgeom = pdgeom_from_vfnf(vertices, face_normals, triangles, name + '_pdgeom')
    pdgeom_nd = GeomNode(name + '_pdgeom_node')
    pdgeom_nd.addGeom(pdgeom)
    pdgeom_ndp = NodePath(pdgeom_nd)
    return pdgeom_ndp


def pdgeom_from_vvnf(vertices: np.ndarray,
                     vertex_normals: np.ndarray,
                     triangles: np.ndarray,
                     name: str = 'auto') -> Geom:
    """
    use environment.collisionmodel instead, vvnf = vertices, vertex normals, faces
    pack the vertices, vertice normals and triangles into a panda3d geom
    :param vertices: nx3 nparray, each row is a vertex
    :param vertex_normals: nx3 nparray, each row is the normal of a vertex
    :param triangles: nx3 nparray, each row is three idx to the vertices
    :param name:
    :return:
    author: weiwei
    date: 20171219, 20210901
    """
    vertex_format = GeomVertexFormat.getV3n3()
    vertex_data = GeomVertexData(name, vertex_format, Geom.UHStatic)
    vertex_data.modifyArrayHandle(0).setData(np.hstack((vertices, vertex_normals)).astype(np.float32).tobytes())
    primitive = GeomTriangles(Geom.UHStatic)
    primitive.setIndexType(GeomEnums.NTUint32)
    primitive.modifyVertices(-1).modifyHandle().setData(triangles.astype(np.uint32).tobytes())
    # make geom
    pdgeom = Geom(vertex_data)
    pdgeom.addPrimitive(primitive)
    return pdgeom


def pdgeomndp_from_vvnf(vertices: np.ndarray,
                        vertex_normals: np.ndarray,
                        triangles: np.ndarray,
                        name: str = 'auto') -> NodePath:
    """
    use environment.collisionmodel instead, vvnf = vertices, vertex normals, faces
    pack the vertices, vertex normals and triangles into a panda3d pdndp
    :param vertices: nx3 nparray, each row is a vertex
    :param vertex_normals: nx3 nparray, each row is the normal of a vertex
    :param triangles: nx3 nparray, each row is three idx to the vertices
    :param name: 
    :return:
    author: weiwei
    date: 20170221, 20210109, 20230811
    """
    pdgeom = pdgeom_from_vvnf(vertices, vertex_normals, triangles, name + 'geom')
    pdgeom_nd = GeomNode('GeomNode')
    pdgeom_nd.addGeom(pdgeom)
    pdgeom_ndp = NodePath(pdgeom_nd)
    return pdgeom_ndp


def pdgeom_from_v(vertices: np.ndarray,
                  rgba: np.ndarray = np.array([.78, 0.7, 0.44]),
                  name: str = 'auto'):
    """
    pack the vertices into a panda3d point cloud geom
    :param vertices:
    :param rgba: rgba color for each vertex, can be nparray (rgb is also acceptable)
    :param name:
    :return:
    author: weiwei
    date: 20170328, 20210116, 20220721, 20230811
    """
    if not isinstance(rgba, np.ndarray):
        raise ValueError('rgba must be an nparray!')
    if rgba.ndim == 1:
        vertex_rgbas = np.tile((rgba * 255).astype(np.uint8), (len(vertices), 1))
        n_color_bit = len(rgba)
    elif rgba.shape[0] == len(vertices):
        vertex_rgbas = (rgba * 255).astype(np.uint8)
        n_color_bit = rgba.shape[1]
    vertex_format = GeomVertexFormat()
    array_format = GeomVertexArrayFormat()
    array_format.addColumn(InternalName.getVertex(), 3, GeomEnums.NTFloat32, GeomEnums.CPoint)
    vertex_format.addArray(array_format)
    array_format = GeomVertexArrayFormat()
    array_format.addColumn(InternalName.getColor(), n_color_bit, GeomEnums.NTUint8, GeomEnums.CColor)
    vertex_format.addArray(array_format)
    vertex_format = GeomVertexFormat.registerFormat(vertex_format)
    vertex_data = GeomVertexData(name, vertex_format, Geom.UHStatic)
    vertex_data.modifyArrayHandle(0).copyDataFrom(np.ascontiguousarray(vertices, dtype=np.float32))
    vertex_data.modifyArrayHandle(1).copyDataFrom(vertex_rgbas)
    primitive = GeomPoints(Geom.UHStatic)
    primitive.setIndexType(GeomEnums.NTUint32)
    primitive.modifyVertices(-1).modifyHandle().copyDataFrom(np.arange(len(vertices), dtype=np.uint32))
    pdgeom = Geom(vertex_data)
    pdgeom.addPrimitive(primitive)
    return pdgeom


def pdgeomndp_from_v(vertices: np.ndarray,
                     rgba: np.ndarray = np.array([.78, 0.7, 0.44]),
                     name: str = 'auto'):
    """
    pack the vertices into a panda3d point cloud pdndp, designed for point clouds
    :param vertices:
    :param rgba: a list with a single 1x4 nparray or with len(vertices) 1x4 nparray
    :param name:
    :return:
    author: weiwei
    date: 20170328
    """
    pdgeom = pdgeom_from_v(vertices, rgba, name + 'geom')
    pdgeom_nd = GeomNode('GeomNode')
    pdgeom_nd.addGeom(pdgeom)
    pdgeom_ndp = NodePath(pdgeom_nd)
    pdgeom_ndp.setLightOff()
    return pdgeom_ndp


def loadfile_vf(file_path: str) -> NodePath:
    """
    load meshes objects into pandanp
    use face normals to pack
    :param file_path:
    :return:
    author: weiwei
    date: 20170221
    """
    trm_mesh = trm.load_mesh(file_path)
    pdgeom_ndp = pdgeomndp_from_vfnf(trm_mesh.vertices, trm_mesh.face_normals, trm_mesh.faces)
    return pdgeom_ndp


def loadfile_vvnf(file_path: str) -> NodePath:
    """
    load meshes objects into panda pdndp
    use vertex normals to pack
    :param obj_path:
    :return:
    author: weiwei
    date: 20170221
    """
    trm_mesh = trm.load_mesh(file_path)
    pdgeom_ndp = pdgeomndp_from_vvnf(trm_mesh.vertices, trm_mesh.vertex_normals, trm_mesh.faces)
    return pdgeom_ndp



if __name__ == '__main__':
    import os
    import wrs.basis.trimesh as trm
    import wrs.visualization.panda.world as wd
    from panda3d.core import TransparencyAttrib

    base = wd.World(cam_pos=[1.0, 1, .0, 1.0], lookat_pos=[0, 0, 0])
    file_path = os.path.join(os.path.dirname(os.path.dirname(trm.__file__)), 'objects', 'bowl.stl')
    print(file_path)
    bt = trm.load(file_path)
    pdnp = pdgeomndp_from_vfnf(bt.vertices, bt.face_normals, bt.faces)
    pdnp.reparentTo(base.render)
    btch = bt.convex_hull
    pdnp_cvxh = pdgeomndp_from_vfnf(btch.vertices, btch.face_normals, btch.faces)
    pdnp_cvxh.setTransparency(TransparencyAttrib.MDual)
    pdnp_cvxh.setColor(0, 1, 0, .3)
    pdnp_cvxh.reparentTo(base.render)
    pdnp2 = pdgeomndp_from_vvnf(bt.vertices, bt.vertex_normals, bt.faces)
    pdnp2.setPos(0, 0, .1)
    pdnp2.reparentTo(base.render)
    base.run()
