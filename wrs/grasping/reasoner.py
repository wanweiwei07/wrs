import wrs.basis.robot_math as rm
import wrs.grasping.grasp as gg


class GraspReasoner(object):
    def __init__(self, robot):
        self.robot = robot

    @staticmethod
    def keep_states_decorator(method):
        """
        decorator function for save and restore robot states
        :return:
        author: weiwei
        date: 20220404
        """

        def wrapper(self, *args, **kwargs):
            if getattr(self, "toggle_keep", True):
                self.robot.backup_state()
                result = method(self, *args, **kwargs)
                self.robot.restore_state()
                return result
            else:
                result = method(self, *args, **kwargs)
                return result

        return wrapper

    def transform_grasp(self, goal_pos, goal_rotmat, grasp):
        """
        transform a grasp to the goal pose
        :param goal_pos:
        :param goal_rotmat:
        :param grasp:
        :return:
        """
        return gg.Grasp(ee_values=grasp.ee_values,
                        ac_pos=goal_pos + goal_rotmat.dot(grasp.ac_pos),
                        ac_rotmat=goal_rotmat.dot(grasp.ac_rotmat))

    ## The following code examines robot IK first. In contrast, the code blow this one examines EE collisions first.
    ## The code below this more is slight more efficient and is thus adopted.
    ## This following code is commented out for future reference.
    ## author: weiwei
    ## date 20240322
    # @keep_states_decorator
    # def reason_incremental_common_gids(self,
    #                                    previous_available_gids,
    #                                    reference_grasp_collection,
    #                                    goal_pose_list,
    #                                    obstacle_list=None,
    #                                    consider_robot=True,
    #                                    toggle_keep=True,
    #                                    toggle_dbg=False):
    #     # start reasoning
    #     intermediate_available_gids = []
    #     eef_collided_grasps_num = 0
    #     ik_failed_grasps_num = 0
    #     rbt_collided_grasps_num = 0
    #     for goal_id, goal_pose in enumerate(goal_pose_list):
    #         goal_pos = goal_pose[0]
    #         goal_rotmat = goal_pose[1]
    #         grasp_with_gid = zip(previous_available_gids,  # need .copy()?
    #                              [reference_grasp_collection[i] for i in previous_available_gids])
    #         previous_available_gids = []
    #         for gid, grasp in grasp_with_gid:
    #             goal_jaw_center_pos = goal_pos + goal_rotmat.dot(grasp.ac_pos)
    #             goal_jaw_center_rotmat = goal_rotmat.dot(grasp.ac_rotmat)
    #             if consider_robot:
    #                 jnt_values = self.robot.ik(tgt_pos=goal_jaw_center_pos, tgt_rotmat=goal_jaw_center_rotmat)
    #                 if jnt_values is not None:
    #                     self.robot.goto_given_conf(jnt_values=jnt_values)
    #                     if not self.robot.is_collided(obstacle_list=obstacle_list):
    #                         if not self.robot.end_effector.is_mesh_collided(cmodel_list=obstacle_list):
    #                             previous_available_gids.append(gid)
    #                             if toggle_dbg:
    #                                 self.robot.end_effector.gen_meshmodel(rgb=rm.const.green, alpha=1).attach_to(base)
    #                         else:  # ee collided
    #                             eef_collided_grasps_num += 1
    #                             if toggle_dbg:
    #                                 self.robot.end_effector.gen_meshmodel(rgb=rm.const.yellow, alpha=1).attach_to(base)
    #                     else:  # robot collided
    #                         rbt_collided_grasps_num += 1
    #                         if toggle_dbg:
    #                             self.robot.gen_meshmodel(rgb=rm.const.orange, toggle_cdprim=True, alpha=1).attach_to(base)
    #                 else:  # ik failure
    #                     ik_failed_grasps_num += 1
    #                     if toggle_dbg:
    #                         self.robot.end_effector.grip_at_by_pose(jaw_center_pos=goal_jaw_center_pos,
    #                                                                 jaw_center_rotmat=goal_jaw_center_rotmat,
    #                                                                 jaw_width=grasp.ee_values)
    #                         self.robot.end_effector.gen_meshmodel(rgb=rm.const.magenta, alpha=1).attach_to(base)
    #             else:
    #                 ik_failed_grasps_num = '-'
    #                 rbt_collided_grasps_num = '-'
    #                 self.robot.end_effector.grip_at_by_pose(jaw_center_pos=goal_jaw_center_pos,
    #                                                         jaw_center_rotmat=goal_jaw_center_rotmat,
    #                                                         jaw_width=grasp.ee_values)
    #                 if not self.robot.end_effector.is_mesh_collided(cmodel_list=obstacle_list):
    #                     previous_available_gids.append(gid)
    #                     if toggle_dbg:
    #                         self.robot.end_effector.gen_meshmodel(rgb=rm.const.green, alpha=1).attach_to(base)
    #                 else:
    #                     eef_collided_grasps_num += 1
    #                     if toggle_dbg:
    #                         self.robot.end_effector.gen_meshmodel(rgb=rm.const.yellow, alpha=1).attach_to(base)
    #         intermediate_available_gids.append(previous_available_gids.copy())
    #         if toggle_dbg:
    #             for obstacle in obstacle_list:
    #                 obstacle.attach_to(base)
    #                 obstacle.show_cdprim()
    #             print('-----start-----')
    #             print(f"Number of available grasps at goal-{str(goal_id)}: {len(previous_available_gids)}")
    #             print(f"Number of collided grasps at goal-{str(goal_id)}: {eef_collided_grasps_num}")
    #             print(f"Number of failed IK at goal-{str(goal_id)}: {ik_failed_grasps_num}")
    #             print(f"Number of collided robots at goal-{str(goal_id)}: {rbt_collided_grasps_num}")
    #             print("------end_type------")
    #             base.run()
    #     return previous_available_gids

    @keep_states_decorator
    def reason_incremental_common_gids(self,
                                       previous_available_gids,
                                       reference_grasp_collection,
                                       goal_pose_list,
                                       obstacle_list=None,
                                       consider_robot=True,
                                       toggle_keep=True,
                                       toggle_dbg=False):
        # start reasoning
        intermediate_available_gids = []
        intermediate_available_grasps = []
        intermediate_available_jv_list = []
        eef_collided_grasps_num = 0
        ik_failed_grasps_num = 0
        rbt_collided_grasps_num = 0
        if toggle_dbg:
            for obstacle in obstacle_list:
                obstacle.attach_to(base)
                obstacle.show_cdprim()
        for goal_id, goal_pose in enumerate(goal_pose_list):
            goal_pos = goal_pose[0]
            goal_rotmat = goal_pose[1]
            grasp_with_gid = zip(previous_available_gids,  # need .copy()?
                                 [reference_grasp_collection[i] for i in previous_available_gids])
            previous_available_gids = []
            previous_availalbe_grasps = []
            previous_available_jv_list = []
            for gid, grasp in grasp_with_gid:
                goal_grasp = self.transform_grasp(goal_pos, goal_rotmat, grasp)
                self.robot.end_effector.grip_at_by_pose(jaw_center_pos=goal_grasp.ac_pos,
                                                        jaw_center_rotmat=goal_grasp.ac_rotmat,
                                                        jaw_width=goal_grasp.ee_values)
                if self.robot.end_effector.is_mesh_collided(cmodel_list=obstacle_list):
                    # ee collided
                    eef_collided_grasps_num += 1
                    if toggle_dbg:
                        self.robot.end_effector.gen_meshmodel(rgb=rm.const.white, alpha=.3).attach_to(base)
                else:
                    if consider_robot:
                        jnt_values = self.robot.ik(tgt_pos=goal_grasp.ac_pos, tgt_rotmat=goal_grasp.ac_rotmat)
                        if jnt_values is None:
                            # ik failure
                            ik_failed_grasps_num += 1
                            if toggle_dbg:
                                self.robot.end_effector.grip_at_by_pose(jaw_center_pos=goal_grasp.ac_pos,
                                                                        jaw_center_rotmat=goal_grasp.ac_rotmat,
                                                                        jaw_width=goal_grasp.ee_values)
                                self.robot.end_effector.gen_meshmodel(rgb=rm.const.magenta, alpha=.3).attach_to(base)
                        else:
                            self.robot.goto_given_conf(jnt_values=jnt_values)
                            if not self.robot.is_collided(obstacle_list=obstacle_list):
                                previous_available_gids.append(gid)
                                previous_availalbe_grasps.append(goal_grasp)
                                previous_available_jv_list.append(jnt_values)
                                if toggle_dbg:
                                    self.robot.end_effector.gen_meshmodel(rgb=rm.const.green, alpha=1).attach_to(base)
                            else:  # robot collided
                                rbt_collided_grasps_num += 1
                                if toggle_dbg:
                                    self.robot.end_effector.gen_meshmodel(rgb=rm.const.orange, alpha=.3).attach_to(base)
                                    self.robot.gen_meshmodel(rgb=rm.const.orange, alpha=.3).attach_to(base)
                                    # _, points = self.robot.is_collided(obstacle_list=obstacle_list, toggle_contacts=True)
                                    # from wrs import mgm
                                    # for point in points:
                                    #     mgm.gen_sphere(pos=point, radius=.001).attach_to(base)
                                    # base.run()
                    else:
                        ik_failed_grasps_num = '-'
                        rbt_collided_grasps_num = '-'
                        previous_available_gids.append(gid)
                        previous_availalbe_grasps.append(goal_grasp)
                        if toggle_dbg:
                            self.robot.end_effector.gen_meshmodel(rgb=rm.const.green, alpha=1).attach_to(base)
            intermediate_available_gids.append(previous_available_gids.copy())
            intermediate_available_grasps.append(previous_availalbe_grasps.copy())
            intermediate_available_jv_list.append(previous_available_jv_list.copy())
            if toggle_dbg:
                for obstacle in obstacle_list:
                    obstacle.attach_to(base)
                    obstacle.show_cdprim()
                print('-----start-----')
                print(f"Number of available grasps at goal-{str(goal_id)}: {len(previous_available_gids)}")
                print(f"Number of collided grasps at goal-{str(goal_id)}: {eef_collided_grasps_num}")
                print(f"Number of failed IK at goal-{str(goal_id)}: {ik_failed_grasps_num}")
                print(f"Number of collided robots at goal-{str(goal_id)}: {rbt_collided_grasps_num}")
                print("------end_type------")
                base.run()
        if consider_robot:
            if len(previous_available_gids) == 0:
                return None, None, None
            return previous_available_gids, previous_availalbe_grasps, previous_available_jv_list
        else:
            if len(previous_available_gids) == 0:
                return None, None
            return previous_available_gids, previous_availalbe_grasps

    # @keep_states_decorator
    def reason_common_gids(self,
                           reference_grasp_collection,
                           goal_pose_list,
                           obstacle_list=None,
                           consider_robot=True,
                           toggle_keep=True,
                           toggle_dbg=False):
        """
        find the common collision free and IK feasible gids
        :param reference_grasp_collection grasping.grasp.GraspCollection
        :param goal_pose_list[[pos0, rotmat0]], [pos1, rotmat1], ...]
        :param obstacle_list
        :param consider_robot whether to consider robot ik and collision
        :param toggle_keep: keep robot states or not
        :param toggle_dbg
        :return: common grasp poses
        author: weiwei
        date: 20210113, 20210125
        """
        # start reasoning
        previous_available_gids = range(len(reference_grasp_collection))
        return self.reason_incremental_common_gids(previous_available_gids=previous_available_gids,
                                                   reference_grasp_collection=reference_grasp_collection,
                                                   goal_pose_list=goal_pose_list,
                                                   obstacle_list=obstacle_list,
                                                   consider_robot=consider_robot,
                                                   toggle_keep=toggle_keep,
                                                   toggle_dbg=toggle_dbg)

    def find_feasible_gids(self,
                           reference_grasp_collection,
                           goal_pose,
                           obstacle_list=None,
                           consider_robot=True,
                           toggle_keep=True,
                           toggle_dbg=False):
        """
        :param reference_grasp_collection:
        :param goal_pose:
        :param obstacle_list:
        :param consider_robot:
        :param toggle_keep:
        :param toggle_dbg:
        :return:
        """
        return self.reason_common_gids(reference_grasp_collection=reference_grasp_collection,
                                       goal_pose_list=[goal_pose],
                                       obstacle_list=obstacle_list,
                                       consider_robot=consider_robot,
                                       toggle_keep=toggle_keep,
                                       toggle_dbg=toggle_dbg)


# TODO incremental

if __name__ == '__main__':
    pass
