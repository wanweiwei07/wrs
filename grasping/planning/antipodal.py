import os
import math
import numpy as np
import basis.robot_math as rm
import grasping.annotation.utils as gau
from scipy.spatial import cKDTree
import modeling.geometric_model as mgm


def plan_contact_pairs(obj_cmodel,
                       angle_between_contact_normals=math.radians(160),
                       max_samples=100,
                       min_dist_between_sampled_contact_points=.005,
                       toggle_sampled_points=False):
    """
    find the contact pairs using rayshooting
    the finally returned number of contact pairs may be smaller than the given max_samples due to the min_dist constraint
    :param angle_between_contact_normals:
    :param toggle_sampled_points
    :return: [[contact_p0, contact_p1], ...]
    author: weiwei
    date: 20190805, 20210504
    """
    contact_points, contact_normals = obj_cmodel.sample_surface(n_samples=max_samples,
                                                                radius=min_dist_between_sampled_contact_points / 2,
                                                                toggle_option='normals')
    contact_pairs = []
    tree = cKDTree(contact_points)
    near_history = np.array([0] * len(contact_points), dtype=bool)
    dot_thresh = -math.cos(angle_between_contact_normals)
    for i, contact_p0 in enumerate(contact_points):
        if near_history[i]:  # if the point was previous near to some points, ignore
            continue
        contact_n0 = contact_normals[i]
        hit_points, hit_normals = obj_cmodel.ray_hit(contact_p0 - contact_n0 * .001, contact_p0 - contact_n0 * 100)
        if len(hit_points) > 0:
            for contact_p1, contact_n1 in zip(hit_points, hit_normals):
                if np.dot(contact_n0, contact_n1) < dot_thresh:
                    near_points_indices = tree.query_ball_point(contact_p1, min_dist_between_sampled_contact_points)
                    if len(near_points_indices):
                        for npi in near_points_indices:
                            if np.dot(contact_n0, contact_normals[npi]) < dot_thresh:
                                near_history[npi] = True
                    contact_pairs.append([[contact_p0, contact_n0], [contact_p1, contact_n1]])
    if toggle_sampled_points:
        return contact_pairs, contact_points
    return contact_pairs


def plan_gripper_grasps(gripper,
                        obj_cmodel,
                        angle_between_contact_normals=math.radians(160),
                        rotation_interval=math.radians(22.5),
                        max_samples=100,
                        min_dist_between_sampled_contact_points=.005,
                        contact_offset=.002,
                        toggle_dbg=False):
    """
    :param gripper:
    :param obj_cmodel:
    :param angle_between_contact_normals:
    :param rotation_granularity:
    :param max_samples:
    :param min_dist_between_sampled_contact_points:
    :param contact_offset: offset at the cotnact to avoid being closely in touch with object surfaces
    :return: a list [[jaw_width, jaw_center_pos, gripper_root_pos, gripper_root_rotmat], ...]
    """
    contact_pairs = plan_contact_pairs(obj_cmodel,
                                       max_samples=max_samples,
                                       min_dist_between_sampled_contact_points=min_dist_between_sampled_contact_points,
                                       angle_between_contact_normals=angle_between_contact_normals)
    if toggle_dbg:
        print(len(contact_pairs))
        for i, cp in enumerate(contact_pairs):
            contact_p0, contact_n0 = cp[0]
            contact_p1, contact_n1 = cp[1]
            mgm.gen_sphere(pos=contact_p0, rgba=np.array([1, 0, 0, 1])).attach_to(base)
            mgm.gen_arrow(spos=contact_p0, epos=contact_p0 + contact_n0 * .01, stick_radius=.00057,
                          rgba=np.array([1, 0, 0, 1])).attach_to(base)
            mgm.gen_sphere(pos=contact_p1, rgba=np.array([0, 0, 1, 1])).attach_to(base)
            mgm.gen_arrow(spos=contact_p1, epos=contact_p1 + contact_n1 * .01, stick_radius=.00057,
                          rgba=np.array([0, 0, 1, 1])).attach_to(base)
    grasp_info_list = []
    for i, cp in enumerate(contact_pairs):
        print(f"{i} of {len(contact_pairs)} done!")
        contact_p0, contact_n0 = cp[0]
        contact_p1, contact_n1 = cp[1]
        contact_center = (contact_p0 + contact_p1) / 2
        jaw_width = np.linalg.norm(contact_p0 - contact_p1) + contact_offset * 2
        if jaw_width > gripper.jaw_range[1]:
            continue
        grasp_info_list += gau.define_gripper_grasps_with_rotation(gripper, obj_cmodel, gl_jaw_center_pos=contact_center,
                                                                   gl_approaching_vec=rm.orthogonal_vector(contact_n0),
                                                                   gl_fgr0_opening_vec=contact_n0, jaw_width=jaw_width,
                                                                   rotation_interval=rotation_interval,
                                                                   toggle_flip=True, toggle_dbg=toggle_dbg)
    return grasp_info_list


def write_pickle_file(obj_name, grasp_info_list, path=None, file_name='preannotated_grasps.pickle', append=False):
    if path is None:
        path = os.getcwd()
    gau.write_pickle_file(obj_name, grasp_info_list, path=path, file_name=file_name, append=append)


def load_pickle_file(obj_name, path=None, file_name='preannotated_grasps.pickle'):
    if path is None:
        path = os.getcwd()
    return gau.load_pickle_file(obj_name, path=path, file_name=file_name)


if __name__ == '__main__':
    import os
    import basis
    import robot_sim.end_effectors.gripper.xarm_gripper.xarm_gripper as xag
    import modeling.collision_model as cm
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
    gripper_s = xag.XArmGripper(enable_cc=True)
    objpath = os.path.join(basis.__path__[0], 'objects', 'block.stl')
    objcm = cm.CollisionModel(objpath)
    objcm.attach_to(base)
    objcm.show_local_frame()
    grasp_info_list = plan_gripper_grasps(gripper_s, objcm, min_dist_between_sampled_contact_points=.02)
    for grasp_info in grasp_info_list:
        jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        gic = gripper_s.copy()
        gic.fix_to(hnd_pos, hnd_rotmat)
        gic.change_jaw_width(jaw_width)
        print(hnd_pos, hnd_rotmat)
        gic.gen_mesh_model().attach_to(base)
    base.run()
