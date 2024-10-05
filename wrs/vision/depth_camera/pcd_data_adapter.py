import numpy as np
import open3d as o3d
import wrs.basis.trimesh as trm


# abbreviations
# pnp panda pdndp
# o3d open3d
# pnppcd - a point cloud in the panda pdndp format

def nparray_to_o3dpcd(nx3nparray_pnts, nx3nparray_nrmls=None, estimate_normals=False):
    """
    :param nx3nparray_pnts: (n,3) nparray
    :param nx3nparray_nrmls, estimate_normals: if nx3nparray_nrmls is None, check estimate_normals, or else do not work on normals
    :return:
    author: ruishuang, weiwei
    date: 20191210
    """

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(nx3nparray_pnts[:, :3])
    if nx3nparray_nrmls is not None:
        o3d_pcd.normals = o3d.utility.Vector3dVector(nx3nparray_nrmls[:, :3])
    elif estimate_normals:
        o3d_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    return o3d_pcd


def o3dpcd_to_parray(o3d_pcd, return_normals=False):
    """
    :param o3d_pcd: open3d point cloud
    :param estimate_normals
    :return:
    author:  weiwei
    date: 20191229, 20200316
    """
    if return_normals:
        if o3d_pcd.has_normals():
            return [np.asarray(o3d_pcd.points), np.asarray(o3d_pcd.normals)]
        else:
            o3d_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
            return [np.asarray(o3d_pcd.points), np.asarray(o3d_pcd.normals)]
    else:
        return np.asarray(o3d_pcd.points)


def o3dmesh_to_trimesh(o3d_mesh):
    """
    :param o3d_mesh:
    :return:
    author: weiwei
    date: 20191210
    """
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    face_normals = np.asarray(o3d_mesh.triangle_normals)
    cvted_trimesh = trm.Trimesh(vertices=vertices, faces=faces, face_normals=face_normals)
    return cvted_trimesh


def crop_o3dpcd(o3d_pcd, x_rng, y_rng, z_rng):
    """
    crop a o3d_pcd
    :param o3d_pcd:
    :param x_rng, y_rng, z_rng: [min, max]
    :return:
    author: weiwei
    date: 20191210
    """
    o3d_pcd_array = np.asarray(o3d_pcd.points)
    x_mask = np.logical_and(o3d_pcd_array[:, 0] > x_rng[0], o3d_pcd_array[:, 0] < x_rng[1])
    y_mask = np.logical_and(o3d_pcd_array[:, 1] > y_rng[0], o3d_pcd_array[:, 1] < y_rng[1])
    z_mask = np.logical_and(o3d_pcd_array[:, 2] > z_rng[0], o3d_pcd_array[:, 2] < z_rng[1])
    mask = x_mask * y_mask * z_mask
    return nparray_to_o3dpcd(o3d_pcd_array[mask])


def crop_nx3_nparray(nx3nparray, x_rng, y_rng, z_rng):
    """
    crop a n-by-3 nparray
    :param nx3nparray:
    :param x_rng, y_rng, z_rng: [min, max]
    :return:
    author: weiwei
    date: 20191210
    """
    x_mask = np.logical_and(nx3nparray[:, 0] > x_rng[0], nx3nparray[:, 0] < x_rng[1])
    y_mask = np.logical_and(nx3nparray[:, 1] > y_rng[0], nx3nparray[:, 1] < y_rng[1])
    z_mask = np.logical_and(nx3nparray[:, 2] > z_rng[0], nx3nparray[:, 2] < z_rng[1])
    mask = x_mask * y_mask * z_mask
    return nx3nparray[mask]
