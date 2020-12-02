# A helper file that converts data between panda3d and trimesh
import trimesh
import numpy as np
from panda3d.core import Geom, GeomNode,GeomPoints, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter
from panda3d.core import NodePath, Vec3, Mat3, Mat4

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
    if ncolors is 1:
        if nonrandcolor:
            return [nonrandcolor[0], nonrandcolor[1], nonrandcolor[2]]
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
    return nodepath_from_vf(trimesh.vertices, trimesh.face_normals, trimesh.faces, name=name)


def o3dmesh_to_nodepath(o3dmesh, name="auto"):
    """
    cvt open3d mesh models to panda models
    :param trimesh:
    :return:
    author: weiwei
    date: 20191210
    """
    return nodepath_from_vf(o3dmesh.vertices, o3dmesh.triangle_normals, o3dmesh.triangles, name=name)


def pandageom_from_vf(vertices, facenormals, triangles, name='auto'):
    """
    pack the given vertices and triangles into a panda3d geom, vf = vertices faces, face normals will be used automatically
    # in 2017, the code was deprecated, by only using vertex normals, the shading was smoothed. sharp edges cannot be seem.
    # in 2018, I compared the number of vertices loaded by loader.loadModel and the one loaded by this function and packpandageom_vn
    # the code is as follows
            with open('loader.txt', 'wb') as fp:
            ur3base_filepath = Filename.fromOsSpecific(os.path.join(this_dir, "ur3egg", "base.egg"))
            ur3base_model2  = loader.loadModel(ur3base_filepath)
            geomNodeCollection = ur3base_model2.findAllMatches('**/+GeomNode')
            for nodePath in geomNodeCollection:
                geomNode = nodePath.node()
                for i in range(geomNode.getNumGeoms()):
                    geom = geomNode.getGeom(i)
                    vdata = geom.getVertexData()
                    vertex = GeomVertexReader(vdata, 'vertex')
                    normal = GeomVertexReader(vdata, 'normal')
                    while not vertex.isAtEnd():
                        v = vertex.getData3f()
                        t = normal.getData3f()
                        fp.write("v = %s, t = %s\n" % (repr(v), repr(t)))
                    break
                break
        with open('selfloader.txt', 'wb') as fp2:
            ur3base_model  = pg.loadstlaspandanp_fn(ur3base_filepath)
            ur3base_filepath = os.path.join(this_dir, "ur3stl", "base.stl")
            geomNodeCollection = ur3base_model.findAllMatches('**/+GeomNode')
            for nodePath in geomNodeCollection:
                geomNode = nodePath.node()
                for i in range(geomNode.getNumGeoms()):
                    geom = geomNode.getGeom(i)
                    vdata = geom.getVertexData()
                    vertex = GeomVertexReader(vdata, 'vertex')
                    normal = GeomVertexReader(vdata, 'normal')
                    while not vertex.isAtEnd():
                        v = vertex.getData3f()
                        t = normal.getData3f()
                        fp2.write("v = %s, t = %s\n" % (repr(v), repr(t)))
                    break
                break
    # the results showed that the loader.loadmodel also repeated the vertices to
    # let each vertex have multiple normals and get better rendering effects.
    # Thus, the function is reused
    # The negative point is the method costs much memory
    # I think it could be solved by developing an independent procedure
    # to analyze normals and faces beforehand
    # in 2019, the duplication is further confirm by reading
    # https://discourse.panda3d.org/t/geom-with-flat-shading/12175/3
    # the shared vertices method is not implemented since they require to make sure the last vertex is not shared
    # a remainin question is does the duplication make the computation load three times larger
    :param vertices: nx3 nparray, each row is vertex
    :param facenormals: nx3 nparray, each row is the normal of aface
    :param triangles: nx3 nparray, each row is three idx to the vertices
    :param name:
    :return: a geom model that is ready to be used to define a nodepath
    author: weiwei
    date: 20160613
    """
    vertformat = GeomVertexFormat.getV3n3()
    vertexdata = GeomVertexData(name, vertformat, Geom.UHStatic)
    vertwritter = GeomVertexWriter(vertexdata, 'vertex')
    normalwritter = GeomVertexWriter(vertexdata, 'normal')
    primitive = GeomTriangles(Geom.UHStatic)
    for i, fvidx in enumerate(triangles):
        vert0 = vertices[fvidx[0], :]
        vert1 = vertices[fvidx[1], :]
        vert2 = vertices[fvidx[2], :]
        vertwritter.addData3f(vert0[0], vert0[1], vert0[2])
        normalwritter.addData3f(facenormals[i, 0], facenormals[i, 1], facenormals[i, 2])
        vertwritter.addData3f(vert1[0], vert1[1], vert1[2])
        normalwritter.addData3f(facenormals[i, 0], facenormals[i, 1], facenormals[i, 2])
        vertwritter.addData3f(vert2[0], vert2[1], vert2[2])
        normalwritter.addData3f(facenormals[i, 0], facenormals[i, 1], facenormals[i, 2])
        primitive.addVertices(i * 3, i * 3 + 1, i * 3 + 2)
    geom = Geom(vertexdata)
    geom.addPrimitive(primitive)
    return geom


def nodepath_from_vf(vertices, facenormals, triangles, name=''):
    """
    pack the given vertices and triangles into a panda3d geom, vf = vertices faces
    :param vertices: nx3 nparray, each row is a vertex
    :param facenormals: nx3 nparray, each row is the normal of a face
    :param triangles: nx3 nparray, each row is three idx to the vertices
    :param name:
    :return: panda3d nodepath
    author: weiwei
    date: 20170221
    """
    objgeom = pandageom_from_vf(vertices, facenormals, triangles, name + 'geom')
    geomnodeobj = GeomNode(name + 'geomnode')
    geomnodeobj.addGeom(objgeom)
    pandanp = NodePath(name + 'nodepath')
    pandanp.attachNewNode(geomnodeobj)

    return pandanp


def pandageom_from_vvnf(vertices, vertnormals, triangles, name=''):
    """
    use environment.collisionmodel instead, vvnf = vertices, vertice normals, faces
    pack the vertices, vertice normals and triangles into a panda3d geom
    :param vertices: nx3 nparray, each row is a vertex
    :param vertnormals: nx3 nparray, each row is the normal of a vertex
    :param triangles: nx3 nparray, each row is three idx to the vertices
    :param name:
    :return:
    author: weiwei
    date: 20171219
    """
    vertformat = GeomVertexFormat.getV3n3()
    vertexdata = GeomVertexData(name, vertformat, Geom.UHStatic)
    vertwritter = GeomVertexWriter(vertexdata, 'vertex')
    normalwritter = GeomVertexWriter(vertexdata, 'normal')
    primitive = GeomTriangles(Geom.UHStatic)
    for i, vert in enumerate(vertices):
        vertwritter.addData3f(vert[0], vert[1], vert[2])
        normalwritter.addData3f(vertnormals[i, 0], vertnormals[i, 1], vertnormals[i, 2])
    for triangle in triangles:
        primitive.addVertices(triangle[0], triangle[1], triangle[2])
    primitive.setShadeModel(GeomEnums.SM_uniform)
    geom = Geom(vertexdata)
    geom.addPrimitive(primitive)
    return geom


def nodepath_from_vvnf(vertices, vertnormals, triangles, name=''):
    """
    use environment.collisionmodel instead, vvnf = vertices, vertice normals, faces
    pack the vertices, vertice normals and triangles into a panda3d nodepath
    :param vertices: nx3 nparray, each row is a vertex
    :param vertnormals: nx3 nparray, each row is the normal of a vertex
    :param triangles: nx3 nparray, each row is three idx to the vertices
    :param name: 
    :return:
    author: weiwei
    date: 20170221
    """
    objgeom = pandageom_from_vvnf(vertices, vertnormals, triangles, name + 'geom')
    geomnodeobj = GeomNode('GeomNode')
    geomnodeobj.addGeom(objgeom)
    pandanp = NodePath(name + 'nodepath')
    pandanp.attachNewNode(geomnodeobj)
    return pandanp


def pandageom_from_points(vertices, rgbas=None, name=''):
    """
    pack the vertices into a panda3d point cloud geom
    :param vertices:
    :param rgbas: 1x4 nparray for all points, or nx4 nparray for each point
    :param name:
    :return:
    author: weiwei
    date: 20170328
    """
    vertformat = GeomVertexFormat.getV3c4()
    vertexdata = GeomVertexData(name, vertformat, Geom.UHStatic)
    vertwritter = GeomVertexWriter(vertexdata, 'vertex')
    colorwritter = GeomVertexWriter(vertexdata, 'color')
    primitive = GeomPoints(Geom.UHStatic)
    for i, vert in enumerate(vertices):
        vertwritter.addData3f(vert[0], vert[1], vert[2])
        if rgbas is None:
            # default
            colorwritter.addData4f(.2, .2, .2, 1)
        elif rgbas.shape == (4,):
            colorwritter.addData4f(rgbas[0], rgbas[1], rgbas[2], rgbas[3])
        else:
            colorwritter.addData4f(rgbas[i][0], rgbas[i][1], rgbas[i][2], rgbas[i][3])
        primitive.addVertex(i)
    geom = Geom(vertexdata)
    geom.addPrimitive(primitive)
    return geom


def nodepath_from_points(vertices, rgbas=None, name=''):
    """
    pack the vertices into a panda3d point cloud nodepath
    :param vertices:
    :param rgbas: 1x4 nparray for all points, or nx4 nparray for each point
    :param name:
    :return:
    author: weiwei
    date: 20170328
    """
    vertformat = GeomVertexFormat.getV3c4()
    vertexdata = GeomVertexData(name, vertformat, Geom.UHStatic)
    vertwritter = GeomVertexWriter(vertexdata, 'vertex')
    colorwritter = GeomVertexWriter(vertexdata, 'color')
    primitive = GeomPoints(Geom.UHStatic)
    for i, vert in enumerate(vertices):
        vertwritter.addData3f(vert[0], vert[1], vert[2])
        if rgbas is None:
            # default
            colorwritter.addData4f(.2, .2, .2, 1)
        elif rgbas.shape == (4,):
            colorwritter.addData4f(rgbas[0], rgbas[1], rgbas[2], rgbas[3])
        else:
            colorwritter.addData4f(rgbas[i][0], rgbas[i][1], rgbas[i][2], rgbas[i][3])
        primitive.addVertex(i)
    geom = Geom(vertexdata)
    geom.addPrimitive(primitive)
    geom_node = GeomNode(name+'geom')
    geom_node.addGeom(geom)
    pointcloud_nodepath = NodePath(name)
    pointcloud_nodepath.attachNewNode(geom_node)
    return pointcloud_nodepath


def loadfile_vf(objpath):
    """
    load stl objects into pandanp
    use face normals to pack
    :param objpath:
    :return:
    author: weiwei
    date: 20170221
    """
    objtrimesh = trimesh.load_mesh(objpath)
    objnp = nodepath_from_vf(objtrimesh.vertices, objtrimesh.face_normals, objtrimesh.faces)
    return objnp


def loadfile_vvnf(objpath):
    """
    load stl objects into panda nodepath
    use vertex normals to pack
    :param objpath:
    :return:
    author: weiwei
    date: 20170221
    """
    objtrimesh = trimesh.load_mesh(objpath)
    objnp = nodepath_from_vvnf(objtrimesh.vertices, objtrimesh.vertex_normals, objtrimesh.faces)
    return objnp
