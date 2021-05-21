import numpy as np
import open3d as o3d
import copy
import sklearn.cluster as skc
import basis.trimesh as trimesh

# abbreviations
# pnp panda nodepath
# o3d open3d
# pnppcd - a point cloud in the panda nodepath format

def nparray_to_o3dpcd(nx3nparray_pnts, nx3nparray_nrmls=None, estimate_normals = False):
    """
    :param nx3nparray_pnts: (n,3) nparray
    :param nx3nparray_nrmls, estimate_normals: if nx3nparray_nrmls is None, check estimate_normals, or else do not work on normals
    :return:
    author: ruishuang, weiwei
    date: 20191210
    """

    o3dpcd = o3d.geometry.PointCloud()
    o3dpcd.points = o3d.utility.Vector3dVector(nx3nparray_pnts[:, :3])
    if nx3nparray_nrmls is not None:
        o3dpcd.normals = o3d.utility.Vector3dVector(nx3nparray_nrmls[:,:3])
    elif estimate_normals:
        o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    return o3dpcd

def o3dpcd_to_parray(o3dpcd, return_normals = False):
    """
    :param o3dpcd: open3d point cloud
    :param estimate_normals
    :return:
    author:  weiwei
    date: 20191229, 20200316
    """
    if return_normals:
        if o3dpcd.has_normals():
            return [np.asarray(o3dpcd.points), np.asarray(o3dpcd.normals)]
        else:
            o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
            return [np.asarray(o3dpcd.points), np.asarray(o3dpcd.normals)]
    else:
        return np.asarray(o3dpcd.points)

def o3dmesh_to_trimesh(o3dmesh):
    """
    :param o3dmesh:
    :return:
    author: weiwei
    date: 20191210
    """
    vertices = np.asarray(o3dmesh.vertices)
    faces = np.asarray(o3dmesh.triangles)
    face_normals = np.asarray(o3dmesh.triangle_normals)
    cvterd_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_normals=face_normals)
    return cvterd_trimesh

def crop_o3dpcd(o3dpcd, xrng, yrng, zrng):
    """
    crop a o3dpcd
    :param o3dpcd:
    :param xrng, yrng, zrng: [min, max]
    :return:
    author: weiwei
    date: 20191210
    """
    o3dpcdarray = np.asarray(o3dpcd.points)
    xmask = np.logical_and(o3dpcdarray[:,0]>xrng[0], o3dpcdarray[:,0]<xrng[1])
    ymask = np.logical_and(o3dpcdarray[:,1]>yrng[0], o3dpcdarray[:,1]<yrng[1])
    zmask = np.logical_and(o3dpcdarray[:,2]>zrng[0], o3dpcdarray[:,2]<zrng[1])
    mask = xmask*ymask*zmask
    return nparray_to_o3dpcd(o3dpcdarray[mask])

def crop_nx3_nparray(nx3nparray, xrng, yrng, zrng):
    """
    crop a n-by-3 nparray
    :param nx3nparray:
    :param xrng, yrng, zrng: [min, max]
    :return:
    author: weiwei
    date: 20191210
    """
    xmask = np.logical_and(nx3nparray[:,0]>xrng[0], nx3nparray[:,0]<xrng[1])
    ymask = np.logical_and(nx3nparray[:,1]>yrng[0], nx3nparray[:,1]<yrng[1])
    zmask = np.logical_and(nx3nparray[:,2]>zrng[0], nx3nparray[:,2]<zrng[1])
    mask = xmask*ymask*zmask
    return nx3nparray[mask]