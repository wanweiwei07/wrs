def get_collisionfree_graspids(hnd, grasp_info_list, goal_and_obstacle_info):
    """
    :param hnd:
    :param grasp_info_list:
    :param goal_and_obstacle_info: [obj_pos, obj_rotmat, obstacle_list]
    :return:
    """
    obj_pos, obj_rotmat, obstacle_list = goal_and_obstacle_info
    available_graspids = []
    for graspid, grasp_info in enumerate(grasp_info_list):
        jaw_width, _, loc_hnd_pos, loc_hnd_rotmat = grasp_info
        gl_hnd_pos = obj_rotmat.dot(loc_hnd_pos) + obj_pos
        gl_hnd_rotmat = obj_rotmat.dot(loc_hnd_rotmat)
        hnd.fix_to(gl_hnd_pos, gl_hnd_rotmat)
        hnd.jaw_to(jaw_width)  # TODO detect a range?
        if not hnd.is_mesh_collided(obstacle_list):
            available_graspids.append(graspid)
    return available_graspids


def get_common_collisionfree_graspids(hnd, grasp_info_list, goal_and_obstacle_info_list):
    """
    get the common collisionfree graspids from a list of [obj_pos, obj_rotmat, and obstacle_list]
    :param hnd:
    :param grasp_info_list:
    :param goal_and_obstacle_info_list: [[obj_pos, obj_rotmat, obstacle_list], ...]
    :return:
    """
    previously_available_graspids = range(len(grasp_info_list))
    intermediate_available_graspids = []
    for goal_and_obstacle_info in goal_and_obstacle_info_list:
        obj_pos, obj_rotmat, obstacle_list = goal_and_obstacle_info
        graspid_and_graspinfo_list = zip(previously_available_graspids,  # need .copy()?
                                         [grasp_info_list[i] for i in previously_available_graspids])
        previously_available_graspids = []
        for graspid, grasp_info in graspid_and_graspinfo_list:
            jaw_width, _, loc_hnd_pos, loc_hnd_rotmat = grasp_info
            gl_hnd_pos = obj_rotmat.dot(loc_hnd_pos) + obj_pos
            gl_hnd_rotmat = obj_rotmat.dot(loc_hnd_rotmat)
            hnd.fix_to(gl_hnd_pos, gl_hnd_rotmat)
            hnd.jaw_to(jaw_width)  # TODO detect a range?
            if not hnd.is_mesh_collided(obstacle_list):
                previously_available_graspids.append(graspid)
        intermediate_available_graspids.append(previously_available_graspids.copy())
    return previously_available_graspids, intermediate_available_graspids

if __name__ == '__main__':
    pass