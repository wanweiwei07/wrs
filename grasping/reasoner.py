def get_collisionfree_graspids(hnd, grasp_info_list, goal_info, obstacle_list):
    """
    :param hnd:
    :param grasp_info_list:
    :param goal_info: [goal_pos, goal_rotmat]
    :param obstacle_list
    :return:
    """
    goal_pos, goal_rotmat = goal_info
    available_graspids = []
    for graspid, grasp_info in enumerate(grasp_info_list):
        jaw_width, _, loc_hnd_pos, loc_hnd_rotmat = grasp_info
        gl_hnd_pos = goal_rotmat.dot(loc_hnd_pos) + goal_pos
        gl_hnd_rotmat = goal_rotmat.dot(loc_hnd_rotmat)
        hnd.fix_to(gl_hnd_pos, gl_hnd_rotmat)
        hnd.jaw_to(jaw_width)  # TODO detect a range?
        if not hnd.is_mesh_collided(obstacle_list):
            available_graspids.append(graspid)
    return available_graspids


def get_common_collisionfree_graspids(hnd, grasp_info_list, goal_info_list, obstacle_list):
    """
    get the common collisionfree graspids from a list of [goal_pos, goal_rotmat] and obstacle_list
    :param hnd:
    :param grasp_info_list:
    :param goal_info_list: [[goal_pos, goal_rotmat], ...]
    :param obstacle_list
    :return:
    """
    previously_available_graspids = range(len(grasp_info_list))
    intermediate_available_graspids = []
    for goal_info in goal_info_list:
        goal_pos, goal_rotmat = goal_info
        graspid_and_graspinfo_list = zip(previously_available_graspids,  # need .copy()?
                                         [grasp_info_list[i] for i in previously_available_graspids])
        previously_available_graspids = []
        for graspid, grasp_info in graspid_and_graspinfo_list:
            jaw_width, _, loc_hnd_pos, loc_hnd_rotmat = grasp_info
            gl_hnd_pos = goal_rotmat.dot(loc_hnd_pos) + goal_pos
            gl_hnd_rotmat = goal_rotmat.dot(loc_hnd_rotmat)
            hnd.fix_to(gl_hnd_pos, gl_hnd_rotmat)
            hnd.jaw_to(jaw_width)  # TODO detect a range?
            if not hnd.is_mesh_collided(obstacle_list):
                previously_available_graspids.append(graspid)
        intermediate_available_graspids.append(previously_available_graspids.copy())
    final_avilable_graspids = previously_available_graspids
    return final_avilable_graspids, intermediate_available_graspids

if __name__ == '__main__':
    pass