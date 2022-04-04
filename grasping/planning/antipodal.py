import math
import numpy as np
import basis.robot_math as rm
import grasping.annotation.utils as gu
from scipy.spatial import cKDTree


def plan_contact_pairs(objcm,
                       max_samples=100,
                       min_dist_between_sampled_contact_points=.005,
                       angle_between_contact_normals=math.radians(160),
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
    contact_points, contact_normals = objcm.sample_surface(nsample=max_samples,
                                                           radius=min_dist_between_sampled_contact_points / 2,
                                                           toggle_option='normals')
    contact_pairs = []
    tree = cKDTree(contact_points)
    near_history = np.array([0] * len(contact_points), dtype=bool)
    for i, contact_p0 in enumerate(contact_points):
        if near_history[i]:  # if the point was previous near to some points, ignore
            continue
        contact_n0 = contact_normals[i]
        hit_points, hit_normals = objcm.ray_hit(contact_p0 - contact_n0 * .001, contact_p0 - contact_n0 * 100)
        if len(hit_points) > 0:
            for contact_p1, contact_n1 in zip(hit_points, hit_normals):
                if np.dot(contact_n0, contact_n1) < -math.cos(angle_between_contact_normals):
                    near_points_indices = tree.query_ball_point(contact_p1, min_dist_between_sampled_contact_points)
                    if len(near_points_indices):
                        for npi in near_points_indices:
                            if np.dot(contact_normals[npi], contact_n1) > math.cos(angle_between_contact_normals):
                                near_history[npi] = True
                    contact_pairs.append([[contact_p0, contact_n0], [contact_p1, contact_n1]])
    if toggle_sampled_points:
        return contact_pairs, contact_points
    return contact_pairs


def plan_grasps(hnd_s,
                objcm,
                angle_between_contact_normals=math.radians(160),
                openning_direction = 'loc_x',
                rotation_interval=math.radians(22.5),
                max_samples=100,
                min_dist_between_sampled_contact_points=.005,
                contact_offset=.002):
    """

    :param objcm:
    :param hnd_s:
    :param angle_between_contact_normals:
    :param openning_direction: 'loc_x' or 'loc_y' depending on gripper types
    :param rotation_granularity:
    :param max_samples:
    :param min_dist_between_sampled_contact_points:
    :param contact_offset: offset at the cotnact to avoid being closely in touch with object surfaces
    :return: a list [[jaw_width, gl_jaw_center_pos, pos, rotmat], ...]
    """
    contact_pairs = plan_contact_pairs(objcm,
                                       max_samples=max_samples,
                                       min_dist_between_sampled_contact_points=min_dist_between_sampled_contact_points,
                                       angle_between_contact_normals=angle_between_contact_normals)
    grasp_info_list = []
    import modeling.geometric_model as gm
    for i, cp in enumerate(contact_pairs):
        print(f"{i} of {len(contact_pairs)} done!")
        contact_p0, contact_n0 = cp[0]
        contact_p1, contact_n1 = cp[1]
        contact_center = (contact_p0 + contact_p1) / 2
        jaw_width = np.linalg.norm(contact_p0 - contact_p1) + contact_offset * 2
        if jaw_width > hnd_s.jawwidth_rng[1]:
            continue
        if openning_direction == 'loc_x':
            jaw_center_x = contact_n0
            jaw_center_z = rm.orthogonal_vector(contact_n0)
            jaw_center_y = np.cross(jaw_center_z, jaw_center_x)
        elif openning_direction == 'loc_y':
            jaw_center_y = contact_n0
            jaw_center_z = rm.orthogonal_vector(contact_n0)
        else:
            raise ValueError("Openning direction must be loc_x or loc_y!")
        grasp_info_list += gu.define_grasp_with_rotation(hnd_s,
                                                         objcm,
                                                         gl_jaw_center_pos=contact_center,
                                                         gl_jaw_center_z=jaw_center_z,
                                                         gl_jaw_center_y=jaw_center_y,
                                                         jaw_width=jaw_width,
                                                         gl_rotation_ax=contact_n0,
                                                         rotation_interval=rotation_interval,
                                                         toggle_flip=True)
    return grasp_info_list


def write_pickle_file(objcm_name, grasp_info_list, root=None, file_name='preannotated_grasps.pickle', append=False):
    if root is None:
        root = './'
    gu.write_pickle_file(objcm_name, grasp_info_list, root=root, file_name=file_name, append=append)


def load_pickle_file(objcm_name, root=None, file_name='preannotated_grasps.pickle'):
    if root is None:
        root = './'
    return gu.load_pickle_file(objcm_name, root=root, file_name=file_name)


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
    objcm.show_localframe()
    grasp_info_list = plan_grasps(gripper_s, objcm, min_dist_between_sampled_contact_points=.02)
    for grasp_info in grasp_info_list:
        jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        gic = gripper_s.copy()
        gic.fix_to(hnd_pos, hnd_rotmat)
        gic.jaw_to(jaw_width)
        print(hnd_pos, hnd_rotmat)
        gic.gen_meshmodel().attach_to(base)
    base.run()
