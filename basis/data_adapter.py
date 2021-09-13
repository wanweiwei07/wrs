# An adapter file that converts data between panda3d and trimesh
import basis.trimesh as trm
import numpy as np
from panda3d.core import Geom, GeomNode, GeomPoints, GeomTriangles
from panda3d.core import GeomVertexData, GeomVertexFormat, GeomVertexArrayFormat, InternalName
from panda3d.core import GeomEnums
from panda3d.core import NodePath, Vec3, Mat3, Mat4, LQuaternion

# data manipulation
def randdom_colorarray(ncolors=1, alpha=1, nonrandcolor=None):
    """
    Generate an array of random colors
    if ncolor = 1, returns a 4-element list
    :param ncolors: the number of colors genrated
    :return: colorarray
    author: weiwei
    date: 20161130osaka
    """
    if ncolors == 1:
        if nonrandcolor:
            return [nonrandcolor[0], nonrandcolor[1], nonrandcolor[2], alpha]
        else:
            return [np.random.random(), np.random.random(), np.random.random(), alpha]
    colorarray = []
    for i in range(ncolors):
        if nonrandcolor:
            colorarray.append([nonrandcolor[0], nonrandcolor[1], nonrandcolor[2], alpha])
        else:
            colorarray.append([np.random.random(), np.random.random(), np.random.random(), alpha])
    return colorarray


def npmat3_to_pdmat3(npmat3):
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


def pdmat3_to_npmat3(pdmat3):
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

    return np.array([[row0[0], row1[0], row2[0]], [row0[1], row1[1], row2[1]], [row0[2], row1[2], row2[2]]])


def npv3mat3_to_pdmat4(npvec3=np.array([0, 0, 0]), npmat3=np.eye(3)):
    """
    convert numpy.2darray to LMatrix4 defined in Panda3d
    note the first parameter is rot, the second is pos
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


def npmat4_to_pdmat4(npmat4):
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


def pdmat4_to_npmat4(pdmat4):
    """
    convert a mat4 matrix to a nparray
    :param pdmat4
    :return: numpy 2darray
    author: weiwei
    date: 20161216sapporo
    """
    return np.array(pdmat4.getRows()).T


def pdmat4_to_npv3mat3(pdmat4):
    """
    :param pdmat4:
    :return: pos, rot: 1x3 and 3x3 nparray
    author: weiwei
    date: 20200206
    """
    homomat = np.array(pdmat4.getRows()).T
    return [homomat[:3, 3], homomat[:3, :3]]


def npmat3_to_pdquat(npmat3):
    """
    :param npmat3: 3x3 nparray
    :return:
    author: weiwei
    date: 20210109
    """
    quat = LQuaternion()
    quat.setFromMatrix(npmat3_to_pdmat3(npmat3))
    return quat


def pdquat_to_npmat3(pdquat):
    """
    :param pdquat: panda.core.LQuaternion
    :return:
    author: weiwei
    date: 20210109
    """
    tmp_pdmat3 = Mat3()
    pdquat.extractToMatrix(tmp_pdmat3)
    return pdmat3_to_npmat3(tmp_pdmat3)


def npv3_to_pdv3(npv3):
    """
    convert a numpy array to Panda3d Vec3...
    :param npv3:
    :return: panda3d vec3
    author: weiwei
    date: 20170322
    """
    return Vec3(npv3[0], npv3[1], npv3[2])


def pdv3_to_npv3(pdv3):
    """
    convert vbase3 to a nprray...
    :param pdmat3:
    :return: numpy 2darray
    author: weiwei
    date: 20161216sapporo
    """
    return np.array([pdv3[0], pdv3[1], pdv3[2]])


def npv4_to_pdv4(npv4):
    """
    convert a numpy array to Panda3d Vec4...
    :param npv4:
    :return: panda3d vec3
    author: weiwei
    date: 20170322
    """
    return Vec4(npv4[0], npv4[1], npv4[2], npv4[3])


def pdv4_to_npv4(pdv4):
    """
    convert vbase3 to a nparray
    :param pdmat3:
    :return: numpy 2darray
    author: weiwei
    date: 20161216sapporo
    """
    return np.array([pdv4[0], pdv4[1], pdv4[2], pdv4[3]])


def trimesh_to_nodepath(trimesh, name="auto"):
    """
    cvt trimesh models to panda models
    :param trimesh:
    :return:
    author: weiwei
    date: 20180606
    """
    return nodepath_from_vfnf(trimesh.vertices, trimesh.face_normals, trimesh.faces, name=name)


def o3dmesh_to_nodepath(o3dmesh, name="auto"):
    """
    cvt open3d mesh models to panda models
    :param trimesh:
    :return:
    author: weiwei
    date: 20191210
    """
    return nodepath_from_vfnf(o3dmesh.vertices, o3dmesh.triangle_normals, o3dmesh.triangles, name=name)


def pandageom_from_vfnf(vertices, face_normals, triangles, name='auto'):
    """
    :param vertices: nx3 nparray, each row is vertex
    :param face_normals: nx3 nparray, each row is the normal of a face
    :param triangles: nx3 nparray, each row is three idx to the vertices
    :param name:
    :return: a geom model that is ready to be used to define a nodepath
    author: weiwei
    date: 20160613, 20210109
    """
    # expand vertices to let each triangle refer to a different vert+normal
    # vertices and normals
    vertformat = GeomVertexFormat.getV3n3()
    vertexdata = GeomVertexData(name, vertformat, Geom.UHStatic)
    vertids = triangles.flatten()
    multiplied_verticies = np.empty((len(vertids), 3), dtype=np.float32)
    multiplied_verticies[:] = vertices[vertids]
    vertex_normals = np.repeat(face_normals.astype(np.float32), repeats=3, axis=0)
    npstr = np.hstack((multiplied_verticies, vertex_normals)).tobytes()
    vertexdata.modifyArrayHandle(0).setData(npstr)
    # triangles
    primitive = GeomTriangles(Geom.UHStatic)
    primitive.setIndexType(GeomEnums.NTUint32)
    multiplied_triangles = np.arange(len(vertids), dtype=np.uint32).reshape(-1,3)
    primitive.modifyVertices(-1).modifyHandle().setData(multiplied_triangles.tobytes())
    # make geom
    geom = Geom(vertexdata)
    geom.addPrimitive(primitive)
    return geom


def nodepath_from_vfnf(vertices, face_normals, triangles, name=''):
    """
    pack the given vertices and triangles into a panda3d geom, vf = vertices faces
    :param vertices: nx3 nparray, each row is a vertex
    :param face_normals: nx3 nparray, each row is the normal of a face
    :param triangles: nx3 nparray, each row is three idx to the vertices
    :param name:
    :return: panda3d nodepath
    author: weiwei
    date: 20170221, 20210109
    """
    objgeom = pandageom_from_vfnf(vertices, face_normals, triangles, name + 'geom')
    geomnodeobj = GeomNode(name + 'geomnode')
    geomnodeobj.addGeom(objgeom)
    pandanp = NodePath(name)
    pandanp.attachNewNode(geomnodeobj)
    return pandanp


def pandageom_from_vvnf(vertices, vertex_normals, triangles, name=''):
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
    vertformat = GeomVertexFormat.getV3n3()
    vertexdata = GeomVertexData(name, vertformat, Geom.UHStatic)
    vertexdata.modifyArrayHandle(0).setData(np.hstack((vertices, vertex_normals)).astype(np.float32).tobytes())
    primitive = GeomTriangles(Geom.UHStatic)
    primitive.setIndexType(GeomEnums.NTUint32)
    primitive.modifyVertices(-1).modifyHandle().setData(triangles.astype(np.uint32).tobytes())
    # make geom
    geom = Geom(vertexdata)
    geom.addPrimitive(primitive)
    return geom


def nodepath_from_vvnf(vertices, vertnormals, triangles, name=''):
    """
    use environment.collisionmodel instead, vvnf = vertices, vertex normals, faces
    pack the vertices, vertice normals and triangles into a panda3d nodepath
    :param vertices: nx3 nparray, each row is a vertex
    :param vertnormals: nx3 nparray, each row is the normal of a vertex
    :param triangles: nx3 nparray, each row is three idx to the vertices
    :param name: 
    :return:
    author: weiwei
    date: 20170221, 20210109
    """
    objgeom = pandageom_from_vvnf(vertices, vertnormals, triangles, name + 'geom')
    geomnodeobj = GeomNode('GeomNode')
    geomnodeobj.addGeom(objgeom)
    pandanp = NodePath(name + 'nodepath')
    pandanp.attachNewNode(geomnodeobj)
    return pandanp


def pandageom_from_points(vertices, rgba_list=None, name=''):
    """
    pack the vertices into a panda3d point cloud geom
    :param vertices:
    :param rgba_list: a list with a single 1x4 nparray or with len(vertices) 1x4 nparray
    :param name:
    :return:
    author: weiwei
    date: 20170328, 20210116
    """
    if rgba_list is None:
        # default
        vertex_rgbas = np.array([[0, 0, 0, 255]]*len(vertices), dtype=np.uint8)
    elif type(rgba_list) is not list:
            raise Exception('rgba\_list must be a list!')
    elif len(rgba_list) == 1:
        vertex_rgbas = np.tile((np.array(rgba_list[0])*255).astype(np.uint8), (len(vertices),1))
    elif len(rgba_list) == len(vertices):
        vertex_rgbas = (np.array(rgba_list)*255).astype(np.uint8)
    else:
        raise ValueError('rgba_list must be a list of one or len(vertices) 1x4 nparray!')
    vertformat = GeomVertexFormat()
    arrayformat = GeomVertexArrayFormat()
    arrayformat.addColumn(InternalName.getVertex(), 3, GeomEnums.NTFloat32, GeomEnums.CPoint)
    vertformat.addArray(arrayformat)
    arrayformat = GeomVertexArrayFormat()
    arrayformat.addColumn(InternalName.getColor(), 4, GeomEnums.NTUint8, GeomEnums.CColor)
    vertformat.addArray(arrayformat)
    vertformat = GeomVertexFormat.registerFormat(vertformat)
    vertexdata = GeomVertexData(name, vertformat, Geom.UHStatic)
    vertexdata.modifyArrayHandle(0).copyDataFrom(np.ascontiguousarray(vertices, dtype=np.float32))
    vertexdata.modifyArrayHandle(1).copyDataFrom(vertex_rgbas)
    primitive = GeomPoints(Geom.UHStatic)
    primitive.setIndexType(GeomEnums.NTUint32)
    primitive.modifyVertices(-1).modifyHandle().copyDataFrom(np.arange(len(vertices), dtype=np.uint32))
    geom = Geom(vertexdata)
    geom.addPrimitive(primitive)
    return geom


def nodepath_from_points(vertices, rgba_list=None, name=''):
    """
    pack the vertices into a panda3d point cloud nodepath
    :param vertices:
    :param rgba_list: a list with a single 1x4 nparray or with len(vertices) 1x4 nparray
    :param name:
    :return:
    author: weiwei
    date: 20170328
    """
    objgeom = pandageom_from_points(vertices, rgba_list, name + 'geom')
    geomnodeobj = GeomNode('GeomNode')
    geomnodeobj.addGeom(objgeom)
    pointcloud_nodepath = NodePath(name)
    pointcloud_nodepath.setLightOff()
    pointcloud_nodepath.attachNewNode(geomnodeobj)
    return pointcloud_nodepath

def loadfile_vf(objpath):
    """
    load meshes objects into pandanp
    use face normals to pack
    :param objpath:
    :return:
    author: weiwei
    date: 20170221
    """
    objtrm = trm.load_mesh(objpath)
    pdnp = nodepath_from_vfnf(objtrm.vertices, objtrm.face_normals, objtrm.faces)
    return pdnp


def loadfile_vvnf(objpath):
    """
    load meshes objects into panda nodepath
    use vertex normals to pack
    :param objpath:
    :return:
    author: weiwei
    date: 20170221
    """
    objtrm = trm.load_mesh(objpath)
    pdnp = nodepath_from_vvnf(objtrm.vertices, objtrm.vertex_normals, objtrm.faces)
    return pdnp


if __name__ == '__main__':
    import os, math, basis
    import basis.trimesh as trimesh
    import visualization.panda.world as wd
    from panda3d.core import TransparencyAttrib

    wd.World(cam_pos=[1.0, 1, .0, 1.0], lookat_pos=[0, 0, 0])
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    bt = trimesh.load(objpath)
    btch = bt.convex_hull
    pdnp = nodepath_from_vfnf(bt.vertices, bt.face_normals, bt.faces)
    pdnp.reparentTo(base.render)
    pdnp_cvxh = nodepath_from_vfnf(btch.vertices, btch.face_normals, btch.faces)
    pdnp_cvxh.setTransparency(TransparencyAttrib.MDual)
    pdnp_cvxh.setColor(0,1,0,.3)
    pdnp_cvxh.reparentTo(base.render)
    pdnp2 = nodepath_from_vvnf(bt.vertices, bt.vertex_normals, bt.faces)
    pdnp2.setPos(0, 0, .1)
    pdnp2.reparentTo(base.render)
    base.run()
