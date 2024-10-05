import math
import numpy as np
import wrs.basis.robot_math as rm
from wrs.grasping.grasp import GraspCollection


def define_gripper_grasps(gripper,
                          obj_cmodel,
                          jaw_center_pos,
                          approaching_direction,
                          thumb_opening_direction,
                          jaw_width,
                          toggle_flip=True,
                          toggle_dbg=False):
    """
    :param gripper:
    :param cmodel:
    :param jaw_center_pos:
    :param approaching_direction: hand approaching motion_vec
    :param thumb_opening_direction: normal motion_vec of thumb's contact surface
    :param jaw_width:
    :param toggle_flip:
    :param toggle_dbg:
    :return: a grasping.grasp.GraspCollection object
    author: haochen, weiwei
    date: 20200104, 2040319
    """
    grasp_collection = GraspCollection(end_effector=gripper)
    collided_grasp_collection = GraspCollection(end_effector=gripper)
    grasp = gripper.grip_at_by_twovecs(jaw_center_pos=jaw_center_pos,
                                       approaching_direction=approaching_direction,
                                       thumb_opening_direction=thumb_opening_direction,
                                       jaw_width=jaw_width)
    if not gripper.is_mesh_collided([obj_cmodel]):
        grasp_collection.append(grasp)
    else:
        collided_grasp_collection.append(grasp)
    if toggle_flip:
        grasp = gripper.grip_at_by_twovecs(jaw_center_pos=jaw_center_pos,
                                           approaching_direction=approaching_direction,
                                           thumb_opening_direction=-thumb_opening_direction,
                                           jaw_width=jaw_width)
        if not gripper.is_mesh_collided([obj_cmodel]):
            grasp_collection.append(grasp)
        else:
            collided_grasp_collection.append(grasp)
    if toggle_dbg:
        for grasp in collided_grasp_collection:
            gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos,
                                    jaw_center_rotmat=grasp.ac_rotmat,
                                    jaw_width=grasp.ee_values)
            gripper.gen_mesh_model(rgba=[1, 0, 0, .3]).attach_to(base)
        for grasp in grasp_collection:
            gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos,
                                    jaw_center_rotmat=grasp.ac_rotmat,
                                    jaw_width=grasp.ee_values)
            gripper.gen_mesh_model(rgba=[0, 1, 0, .3]).attach_to(base)
    return grasp_collection


def define_gripper_grasps_with_rotation(gripper,
                                        obj_cmodel,
                                        jaw_center_pos,
                                        approaching_direction,
                                        thumb_opening_direction,
                                        jaw_width,
                                        rotation_interval=math.radians(60),
                                        rotation_range=(math.radians(-180), math.radians(180)),
                                        toggle_flip=True,
                                        toggle_dbg=False):
    """
    :param gripper:
    :param obj_cmodel:
    :param jaw_center_pos:
    :param approaching_direction: hand approaching motion_vec
    :param thumb_opening_direction: normal motion_vec of thumb's contact surface
    :param jaw_width: 
    :param rotation_interval: 
    :param rotation_range: 
    :param toggle_flip:
    :param toggle_dbg:
    :return: a grasping.grasp.GraspCollection object
    author: haochen, weiwei
    date: 20200104, 20240319
    """
    grasp_collection = GraspCollection(end_effector=gripper)
    collided_grasp_collection = GraspCollection(end_effector=gripper)
    for rotate_angle in np.arange(rotation_range[0], rotation_range[1], rotation_interval):
        rotated_rotmat = rm.rotmat_from_axangle(thumb_opening_direction, rotate_angle)
        rotated_approaching_direction = np.dot(rotated_rotmat, approaching_direction)
        rotated_thumb_opening_direction = np.dot(rotated_rotmat, thumb_opening_direction)
        grasp = gripper.grip_at_by_twovecs(jaw_center_pos=jaw_center_pos,
                                           approaching_direction=rotated_approaching_direction,
                                           thumb_opening_direction=rotated_thumb_opening_direction,
                                           jaw_width=jaw_width)
        if toggle_dbg:
            gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos,
                                    jaw_center_rotmat=grasp.ac_rotmat,
                                    jaw_width=grasp.ee_values)
            gripper.gen_meshmodel(alpha=.3).attach_to(base)
        if not gripper.is_mesh_collided([obj_cmodel], toggle_dbg=toggle_dbg):
            grasp_collection.append(grasp)
        else:
            collided_grasp_collection.append(grasp)
    if toggle_flip:
        for rotate_angle in np.arange(rotation_range[0], rotation_range[1], rotation_interval):
            rotated_rotmat = rm.rotmat_from_axangle(thumb_opening_direction, rotate_angle)
            rotated_approaching_direction = np.dot(rotated_rotmat, approaching_direction)
            rotated_thumb_opening_direction = np.dot(rotated_rotmat, -thumb_opening_direction)
            grasp = gripper.grip_at_by_twovecs(jaw_center_pos=jaw_center_pos,
                                               approaching_direction=rotated_approaching_direction,
                                               thumb_opening_direction=rotated_thumb_opening_direction,
                                               jaw_width=jaw_width)
            if not gripper.is_mesh_collided([obj_cmodel]):
                grasp_collection.append(grasp)
            else:
                collided_grasp_collection.append(grasp)
    if toggle_dbg:
        for grasp in collided_grasp_collection:
            gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos,
                                    jaw_center_rotmat=grasp.ac_rotmat,
                                    jaw_width=grasp.ee_values)
            gripper.gen_meshmodel(rgba=[1, 0, 0, .3]).attach_to(base)
        for grasp in grasp_collection:
            gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos,
                                    jaw_center_rotmat=grasp.ac_rotmat,
                                    jaw_width=grasp.ee_values)
            gripper.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
    return grasp_collection


if __name__ == '__main__':
    import os
    import wrs.visualization.panda.world as wd
    import wrs.modeling.collision_model as mcm
    import wrs.robot_sim.end_effectors.grippers.xarm_gripper.xarm_gripper as end_effector

    base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
    gripper = end_effector.XArmGripper()
    obj_path = os.path.join(os.path.dirname(rm.__file__), 'objects', 'block.stl')
    obj_cmodel = mcm.CollisionModel(obj_path)
    obj_cmodel.attach_to(base)
    obj_cmodel.show_local_frame()
    grasp_collection = define_gripper_grasps_with_rotation(gripper,
                                                           obj_cmodel,
                                                           jaw_center_pos=np.array([0, 0, 0]),
                                                           approaching_direction=np.array([1, 0, 0]),
                                                           thumb_opening_direction=np.array([0, 1, 0]),
                                                           jaw_width=.04)
    print(grasp_collection)
    for grasp in grasp_collection:
        gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat,
                                jaw_width=grasp.ee_values)
        gripper.gen_meshmodel(alpha=.3).attach_to(base)
    base.run()
