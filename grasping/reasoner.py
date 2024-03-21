
class GraspReasonser(object):
    def __init__(self, robot):
        self.robot = robot

    def reason_common_gids(self,
                           grasp_collection,
                           goal_pose_list,
                           obstacle_list=None,
                           toggle_dbg=False):
        """
        find the common collision free and IK feasible gids
        :param eef: an end effector instance
        :param grasp_collection: grasping.grasp.GraspCollection
        :param goal_pose_list: [[pos0, rotmat0]], [pos1, rotmat1], ...]
        :param obstacle_list
        :return: common grasp poses
        author: weiwei
        date: 20210113, 20210125
        """
        # start reasoning
        previous_available_gids = range(len(grasp_collection))
        intermediate_available_gids = []
        eef_collided_grasps_num = 0
        ik_failed_grasps_num = 0
        rbt_collided_grasps_num = 0
        for goal_id, goal_pose in enumerate(goal_pose_list):
            goal_pos = goal_pose[0]
            goal_rotmat = goal_pose[1]
            grasp_with_gid = zip(previous_available_gids,  # need .copy()?
                                 [grasp_collection[i] for i in previous_available_gids])
            previous_available_gids = []
            for gid, grasp in grasp_with_gid:
                goal_jaw_center_pos = goal_pos + goal_rotmat.dot(grasp.ac_pos)
                goal_jaw_center_rotmat = goal_rotmat.dot(grasp.ac_rotmat)
                jnt_values = self.robot.ik(tgt_pos=goal_jaw_center_pos, tgt_rotmat=goal_jaw_center_rotmat)
                if jnt_values is not None:
                    self.robot.goto_given_conf(jnt_values=jnt_values)
                    if not self.robot.is_collided(obstacle_list=obstacle_list):
                        if not self.robot.end_effector.is_mesh_collided(cmodel_list=obstacle_list):
                            previous_available_gids.append(gid)
                            if toggle_dbg:
                                self.robot.end_effector.gen_meshmodel(rgb=rm.bc.green, alpha=1).attach_to(base)
                        else:  # ee collided
                            eef_collided_grasps_num += 1
                            if toggle_dbg:
                                self.robot.end_effector.gen_meshmodel(rgb=rm.bc.yellow, alpha=.3).attach_to(base)
                    else:  # robot collided
                        rbt_collided_grasps_num += 1
                        if toggle_dbg:
                            self.robot.end_effector.gen_meshmodel(rgb=rm.bc.orange, alpha=.3).attach_to(base)
                else:  # ik failure
                    ik_failed_grasps_num += 1
                    if toggle_dbg:
                        self.robot.end_effector.grip_at_by_pose(jaw_center_pos=goal_jaw_center_pos,
                                                                jaw_center_rotmat=goal_jaw_center_rotmat,
                                                                jaw_width=grasp.ee_values)
                        self.robot.end_effector.gen_meshmodel(rgb=rm.bc.magenta, alpha=.3).attach_to(base)
            intermediate_available_gids.append(previous_available_gids.copy())
            if toggle_dbg:
                print('-----start-----')
                print(f"Number of available grasps at goal-{str(goal_id)}: ", len(previous_available_gids))
                print("Number of collided grasps at goal-{str(goal_id)}: ", eef_collided_grasps_num)
                print("Number of failed IK at goal-{str(goal_id)}: ", ik_failed_grasps_num)
                print("Number of collided robots at goal-{str(goal_id)}: ", rbt_collided_grasps_num)
                print("------end_type------")
                base.run()
        return previous_available_gids

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
        hnd.change_jaw_width(jaw_width)  # TODO detect a range?
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
            hnd.change_jaw_width(jaw_width)  # TODO detect a range?
            if not hnd.is_mesh_collided(obstacle_list):
                previously_available_graspids.append(graspid)
        intermediate_available_graspids.append(previously_available_graspids.copy())
    final_avilable_graspids = previously_available_graspids
    return final_avilable_graspids, intermediate_available_graspids

# TODO incremental

if __name__ == '__main__':
    pass