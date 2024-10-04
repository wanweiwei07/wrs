import numpy as np
import copy
import open3d as o3d
import sklearn.cluster as skc
from wrs import vision as pda


def __draw_registration_result(source_o3d, target_o3d, transformation):
    source_temp = copy.deepcopy(source_o3d)
    target_temp = copy.deepcopy(target_o3d)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def __preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    down_radius_normal = voxel_size * 3
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=down_radius_normal, max_nn=30))
    radius_feature = voxel_size * 3
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def registration_ptpt(src, tgt, downsampling_voxelsize=.003, toggledebug = False):
    """
    registrate two point clouds using global registration + local icp
    the correspondence checker for icp is point to point
    :param src: nparray
    :param tgt: nparray
    :param downsampling_voxelsize:
    :param icp_distancethreshold:
    :param debug:
    :return: quality, pos: quality is measured as RMSE of all inlier correspondences
    author: ruishuang, revised by weiwei
    date: 20191210
    """

    src_o3d = pda.nparray_to_o3dpcd(src)
    tgt_o3d = pda.nparray_to_o3dpcd(tgt)
    if toggledebug:
        __draw_registration_result(src_o3d, tgt_o3d, np.identity(4))
    source_down, source_fpfh = __preprocess_point_cloud(src_o3d, downsampling_voxelsize)
    target_down, target_fpfh = __preprocess_point_cloud(tgt_o3d, downsampling_voxelsize)
    distance_threshold = downsampling_voxelsize * 1.5
    if toggledebug:
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % downsampling_voxelsize)
        print("   we use a liberal linear_distance threshold %.3f." % distance_threshold)
    # result_global = o3d.pipelines.registration.registration_fass_based_on_feature_matching(
    #     source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
    #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
    #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
    #     o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    result_global = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    if toggledebug:
        __draw_registration_result(source_down, target_down, result_global.transformation)
    distance_threshold = downsampling_voxelsize * 0.4
    if toggledebug:
        print(":: Point-to-point ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   linear_distance threshold %.3f." % distance_threshold)
    return _registration_icp_ptpt_o3d(src_o3d, tgt_o3d, result_global.transformation, toggledebug=toggledebug)

    # def _registration_icp_ptpt_o3d(src, tgt, inithomomat=np.eye(4), maxcorrdist=2, toggledebug=False):
    #     """
    #
    #     :param src:
    #     :param tgt:
    #     :param maxcorrdist:
    #     :param toggledebug:
    #     :return:
    #
    #     author: weiwei
    #     date: 20191229
    #     """
    #
    #     criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
    #                                                        # converge if fitnesss smaller than this
    #                                                        relative_rmse=1e-6,  # converge if rmse smaller than this
    #                                                        max_iteration=2000)
    #     result_icp = o3d.pipelines.registration.registration_icp(src, tgt, maxcorrdist, inithomomat, criteria=criteria)
    #     if toggledebug:
    #         __draw_registration_result(src, tgt, result_icp.transformation)
    #     return [result_icp.inlier_rmse, result_icp.transformation]
    #
    # result_icp = o3d.pipelines.registration.registration_icp(
    #     src_o3d, tgt_o3d, distance_threshold, result_global.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False))
    # if toggledebug:
    #     __draw_registration_result(src_o3d, tgt_o3d, result_icp.transformation)
    # return [result_icp.inlier_rmse, result_icp.transformation]

def registration_ptpln(src, tgt, downsampling_voxelsize=2, toggledebug = False):
    """
    registrate two point clouds using global registration + local icp
    the correspondence checker for icp is point to plane
    :param src:
    :param tgt:
    :param downsampling_voxelsize:
    :param icp_distancethreshold:
    :param debug:
    :return: quality, pos: quality is measured as RMSE of all inlier correspondences
    author: ruishuang, revised by weiwei
    date: 20191210
    """
    src_o3d = pda.nparray_to_o3dpcd(src)
    tgt_o3d = pda.nparray_to_o3dpcd(tgt)
    def __preprocess_point_cloud(pcd, voxel_size):
        original_radius_normal = voxel_size
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=original_radius_normal, max_nn=30))
        pcd_down = pcd.voxel_down_sample(voxel_size)
        down_radius_normal = voxel_size * 2
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=down_radius_normal, max_nn=30))
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh
    if toggledebug:
        __draw_registration_result(src_o3d, tgt_o3d, np.identity(4))
    source_down, source_fpfh = __preprocess_point_cloud(src_o3d, downsampling_voxelsize)
    target_down, target_fpfh = __preprocess_point_cloud(tgt_o3d, downsampling_voxelsize)
    distance_threshold = downsampling_voxelsize * 1.5
    if toggledebug:
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % downsampling_voxelsize)
        print("   we use a liberal linear_distance threshold %.3f." % distance_threshold)
    result_global = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    if toggledebug:
        __draw_registration_result(source_down, target_down, result_global.transformation)
    distance_threshold = downsampling_voxelsize * 0.4
    if toggledebug:
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   linear_distance threshold %.3f." % distance_threshold)
    result_icp = o3d.pipelines.registration.registration_icp(
        src_o3d, tgt_o3d, distance_threshold, result_global.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    if toggledebug:
        __draw_registration_result(src_o3d, tgt_o3d, result_icp.transformation)
    return [result_icp.inlier_rmse, result_icp.transformation]

def registration_icp_ptpt(src, tgt, inithomomat=np.eye(4), maxcorrdist=2, toggledebug=False):
    """
    :param src:
    :param tgt:
    :param maxcorrdist:
    :param toggledebug:
    :return:
    author: weiwei
    date: 20191229
    """
    src_o3d = pda.nparray_to_o3dpcd(src)
    tgt_o3d = pda.nparray_to_o3dpcd(tgt)
    return _registration_icp_ptpt_o3d(src_o3d, tgt_o3d, inithomomat, maxcorrdist, toggledebug)

def remove_outlier(src_nparray, downsampling_voxelsize=2, nb_points=7, radius=3, estimate_normals = False, toggledebug=False):
    """
    downsample and remove outliers statistically
    :param src:
    :return: cleared o3d point cloud and their normals [pcd_nparray, nrmls_nparray]
    author: weiwei
    date: 20191229
    """
    src_o3d = pda.nparray_to_o3dpcd(src_nparray, estimate_normals = estimate_normals)
    cl = _removeoutlier_o3d(src_o3d, downsampling_voxelsize, nb_points, radius, toggledebug)
    return pda.o3dpcd_to_nparray(cl, return_normals=estimate_normals)

def _removeoutlier_o3d(src_o3d, downsampling_voxelsize=2, nb_points=7, radius=3, toggledebug=False):
    """
    downsample and remove outliers statistically
    :param src:
    :return: cleared o3d point cloud
    author: weiwei
    date: 20200316
    """
    if downsampling_voxelsize is None:
        src_o3d_down = src_o3d
    else:
        src_o3d_down = src_o3d.voxel_down_sample(downsampling_voxelsize)
    cl, _ = src_o3d_down.remove_radius_outlier(nb_points=nb_points, radius=radius)
    if toggledebug:
        src_o3d_down.paint_uniform_color([1, 0, 0])
        src_o3d.paint_uniform_color([0, 1, 0])
        cl.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([src_o3d_down, src_o3d, cl])
    return cl

def _registration_icp_ptpt_o3d(src, tgt, inithomomat=np.eye(4), maxcorrdist=2, toggledebug=False):
    """
    :param src:
    :param tgt:
    :param maxcorrdist:
    :param toggledebug:
    :return: [rmse of matched points, size of matched area, pos]
    author: weiwei
    date: 20191229
    """
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,  # converge if fitnesss smaller than this
                                                       relative_rmse=1e-6, # converge if rmse smaller than this
                                                       max_iteration=2000)
    result_icp = o3d.pipelines.registration.registration_icp(src, tgt, maxcorrdist, inithomomat, criteria=criteria)
    if toggledebug:
        __draw_registration_result(src, tgt, result_icp.transformation)
    return [result_icp.inlier_rmse, result_icp.fitness, result_icp.transformation]

def cluster_pcd(pcd_nparray, pcd_nparray_nrmls = None):
    """
    segment mph into clusters using the DBSCAN method
    :param pcd_nparray:
    :return:
    author: weiwei
    date: 20200316
    """
    db = skc.DBSCAN(eps=10, min_samples=50, n_jobs=-1).fit(pcd_nparray)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    pcd_nparray_list = []
    pcdnrmls_nparray_list = []
    if pcd_nparray_nrmls is None:
        pcd_nparray_nrmls = np.array([[0, 0, 1]] * pcd_nparray.shape[0])
    for k in unique_labels:
        if k == -1:
            continue
        else:
            class_member_mask = (labels == k)
            temppartialpcd = pcd_nparray[class_member_mask & core_samples_mask]
            pcd_nparray_list.append(temppartialpcd)
            temppartialpcdnrmls = pcd_nparray_nrmls[class_member_mask & core_samples_mask]
            pcdnrmls_nparray_list.append(temppartialpcdnrmls)
    return [pcd_nparray_list, pcdnrmls_nparray_list]

def merge_pcd(pnppcd1, pnppcd2, rotmat2, posmat2):
    """
    merge nppcd2 and nppcd1 by rotating and moving nppcd2 using rotmat2 and posmat2
    :param pnppcd1:
    :param pnppcd2:
    :param rotmat2:
    :param posmat2:
    :return:
    author: weiwei
    date: 20200221
    """
    transformednppcd2 = np.dot(rotmat2, pnppcd2.T).T+posmat2
    mergednppcd = np.zeros((len(transformednppcd2)+len(pnppcd1),3))
    mergednppcd[:len(pnppcd1), :] = pnppcd1
    mergednppcd[len(pnppcd1):, :] = transformednppcd2
    return mergednppcd

def reconstruct_surfaces_bp(nppcd, nppcdnrmls = None, radii = [5], doseparation=True):
    """
    :param nppcd:
    :param radii:
    :param doseparation: separate the reconstructed meshes or not when they are disconnected
    :return:
    """
    if doseparation:
        # db = skc.DBSCAN(eps=10, min_samples=50).fit(nppcd)
        # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # core_samples_mask[db.core_sample_indices_] = True
        # labels = db.labels_
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # unique_labels = set(labels)
        # nppcdlist = []
        # nppcdnrmlslist = []
        # if nppcdnrmls is None:
        #     nppcdnrmls = np.array([[0,0,1]]*nppcd.shape[0])
        # else:
        #     nppcdnrmls = nppcdnrmls
        # for k in unique_labels:
        #     if k == -1:
        #         continue
        #     else:
        #         class_member_mask = (labels == k)
        #         temppartialpcd = nppcd[class_member_mask & core_samples_mask]
        #         nppcdlist.append(temppartialpcd)
        #         temppartialpcdnrmls = nppcdnrmls[class_member_mask & core_samples_mask]
        #         nppcdnrmlslist.append(temppartialpcdnrmls)
        nppcdlist, nppcdnrmlslist = cluster_pcd(nppcd, nppcdnrmls)

        tmmeshlist = []
        for i, thisnppcd in enumerate(nppcdlist):
            o3dpcd = pda.nparray_to_o3dpcd(thisnppcd, nppcdnrmlslist[i])
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(o3dpcd, o3d.utility.DoubleVector(radii))
            # mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            tmmesh = pda.o3dmesh_to_trimesh(mesh)
            tmmeshlist.append(tmmesh)
        return tmmeshlist, nppcdlist
    else:
        if nppcdnrmls is None:
            npnrmls = np.array([[0,0,1]]*nppcd.shape[0])
        else:
            npnrmls = nppcdnrmls
        o3dpcd = pda.nparray_to_o3dpcd(nppcd, npnrmls)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(o3dpcd, o3d.utility.DoubleVector(radii))
        # mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        tmmesh = pda.o3dmesh_to_trimesh(mesh)
        return tmmesh

def get_obb(pnppcd):
    """
    :param pnppcd:
    :return: [center_3x1nparray, corners_8x3nparray]
    author:
    date:
    """
    # TODO get the object oriented bounding box of a point cloud using PoindCloud.get_oriented_bounding_box() and OrientedBoundinBox
    pass

