import wrs.basis.robot_math as rm
import wrs.motion.primitives.approach_depart_planner as adp
import wrs.robot_sim.robots.yumi.yumi as ym


class PickPlacePlanner(adp.ADPlanner):

    def __init__(self, robot):
        """
        :param object:
        :param robot: must be an instance of SglArmRobotInterface
        author: weiwei, hao
        date: 20191122, 20210113, 20240316
        """
        super().__init__(robot)

    @adp.mpi.InterplatedMotion.keep_states_decorator
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
            grasp_and_gid = zip(previous_available_gids,  # need .copy()?
                                [grasp_collection[i] for i in previous_available_gids])
            previous_available_gids = []
            for gid, grasp in grasp_and_gid:
                goal_jaw_center_pos = goal_pos + goal_rotmat.dot(grasp.ac_pos)
                goal_jaw_center_rotmat = goal_rotmat.dot(grasp.ac_rotmat)
                jnt_values = self.robot.ik(tgt_pos=goal_jaw_center_pos, tgt_rotmat=goal_jaw_center_rotmat)
                if jnt_values is not None:
                    self.robot.goto_given_conf(jnt_values=jnt_values, ee_values=grasp.ee_values)
                    if not self.robot.is_collided(obstacle_list=obstacle_list):
                        if not self.robot.end_effector.is_mesh_collided(cmodel_list=obstacle_list):
                            previous_available_gids.append(gid)
                            if toggle_dbg:
                                self.robot.end_effector.gen_meshmodel(rgb=rm.const.green, alpha=1).attach_to(base)
                                # self.robot.gen_meshmodel(rgb=rm.const.green, alpha=.3).attach_to(base)
                        else:  # ee collided
                            eef_collided_grasps_num += 1
                            if toggle_dbg:
                                self.robot.end_effector.gen_meshmodel(rgb=rm.const.yellow, alpha=.3).attach_to(base)
                                # self.robot.gen_meshmodel(rgb=rm.const.yellow, alpha=.3).attach_to(base)
                    else:  # robot collided
                        rbt_collided_grasps_num += 1
                        if toggle_dbg:
                            self.robot.end_effector.gen_meshmodel(rgb=rm.const.orange, alpha=.3).attach_to(base)
                            # self.robot.gen_meshmodel(rgb=rm.const.orange, alpha=.3).attach_to(base)
                else:  # ik failure
                    ik_failed_grasps_num += 1
                    if toggle_dbg:
                        self.robot.end_effector.grip_at_by_pose(jaw_center_pos=goal_jaw_center_pos,
                                                                jaw_center_rotmat=goal_jaw_center_rotmat,
                                                                jaw_width=grasp.ee_values)
                        self.robot.end_effector.gen_meshmodel(rgb=rm.const.magenta, alpha=.3).attach_to(base)
            intermediate_available_gids.append(previous_available_gids.copy())
            if toggle_dbg:
                print('-----start-----')
                print(f"Number of available grasps at goal-{str(goal_id)}: ", len(previous_available_gids))
                print("Number of collided grasps at goal-{str(goal_id)}: ", eef_collided_grasps_num)
                print("Number of failed IK at goal-{str(goal_id)}: ", ik_failed_grasps_num)
                print("Number of collided robots at goal-{str(goal_id)}: ", rbt_collided_grasps_num)
                print("------end_type------")
        if toggle_dbg:
            base.run()
        return previous_available_gids

    @adp.mpi.InterplatedMotion.keep_states_decorator
    def gen_pick_and_moveto(self,
                            obj_cmodel,
                            grasp,
                            goal_pose_list,
                            approach_direction_list,
                            approach_distance_list,
                            depart_direction_list,
                            depart_distance_list,
                            pick_jaw_width=None,
                            pick_approach_direction=None,
                            pick_approach_distance=.07,
                            pick_depart_direction=None,
                            pick_depart_distance=.07,
                            linear_granularity=.02,
                            obstacle_list=None,
                            use_rrt=True):
        """
        pick and move an object to multiple poses
        :param obj_cmodel:
        :param grasp:
        :param goal_pose_list: [[pos, rotmat], [pos, rotmat], ...]
        :param approach_direction_list: the first element will be ignored
        :param approach_distance_list: the first element will be ignored
        :param depart_direction_list: the last element will be ignored
        :param depart_distance_list: the last element will be ignored
        :param pick_approach_direction
        :param pick_approach_distance
        :param pick_depart_direction
        :param pick_depart_distance
        :param linear_granularity:
        :param obstacle_list:
        :param seed_jnt_values:
        :return:
        """
        # pick up object
        pick_tcp_pos = obj_cmodel.rotmat.dot(grasp.ac_pos) + obj_cmodel.pos
        pick_tcp_rotmat = obj_cmodel.rotmat.dot(grasp.ac_rotmat)
        if pick_jaw_width is None:
            pick_jaw_width = self.robot.end_effector.jaw_range[1]
        pick_motion = self.gen_approach(goal_tcp_pos=pick_tcp_pos,
                                        goal_tcp_rotmat=pick_tcp_rotmat,
                                        start_jnt_values=self.robot.get_jnt_values(),
                                        linear_direction=pick_approach_direction,
                                        linear_distance=pick_approach_distance,
                                        ee_values=pick_jaw_width,
                                        granularity=linear_granularity,
                                        obstacle_list=obstacle_list,
                                        object_list=[obj_cmodel],
                                        use_rrt=use_rrt)
        if pick_motion is None:
            print("PPPlanner: Error encountered when generating pick approach motion!")
            return None
        else:
            obj_cmodel_copy = obj_cmodel.copy()
            self.robot.goto_given_conf(pick_motion.jv_list[-1])
            # self.robot.gen_meshmodel().attach_to(base)
            self.robot.hold(obj_cmodel=obj_cmodel_copy, jaw_width=grasp.ee_values)
            # self.robot.gen_meshmodel().attach_to(base)
            # print("before back up")
            pick_motion.extend([pick_motion.jv_list[-1]], [grasp.ee_values], [self.robot.gen_meshmodel()])
            pick_depart = self.gen_linear_depart_from_given_conf(start_jnt_values=pick_motion.jv_list[-1],
                                                                 direction=pick_depart_direction,
                                                                 distance=pick_depart_distance,
                                                                 ee_values=None,
                                                                 granularity=linear_granularity,
                                                                 obstacle_list=obstacle_list)
            # self.robot.gen_meshmodel().attach_to(base)
            # base.run()
            if pick_depart is None:
                print("PPPlanner: Error encountered when generating pick depart motion!")
                return None
            else:
                moveto_motion = adp.mpi.motd.MotionData(robot=self.robot)
                # move to goals
                moveto_start_jnt_values = pick_depart.jv_list[-1]
                for i, goal_pose in enumerate(goal_pose_list):
                    goal_tcp_pos = goal_pose[1].dot(grasp.ac_pos) + goal_pose[0]
                    goal_tcp_rotmat = goal_pose[1].dot(grasp.ac_rotmat)
                    moveto_ap = self.gen_approach_depart(goal_tcp_pos=goal_tcp_pos,
                                                         goal_tcp_rotmat=goal_tcp_rotmat,
                                                         start_jnt_values=moveto_start_jnt_values,
                                                         approach_direction=approach_direction_list[i],
                                                         approach_distance=approach_distance_list[i],
                                                         approach_ee_values=None,
                                                         depart_direction=depart_direction_list[i],
                                                         depart_distance=depart_distance_list[i],
                                                         depart_ee_values=None,  # do not change jaw width
                                                         granularity=linear_granularity,
                                                         obstacle_list=obstacle_list,
                                                         use_rrt=use_rrt)
                    if moveto_ap is None:
                        print(f"Error encountered when generating motion to the {i}th goal!")
                        return None
                    else:
                        moveto_motion += moveto_ap
                        moveto_start_jnt_values = moveto_motion.jv_list[-1]
                return pick_motion + pick_depart + moveto_motion

    @adp.mpi.InterplatedMotion.keep_states_decorator
    def gen_pick_and_place(self,
                           obj_cmodel,
                           end_jnt_values,
                           grasp_collection,
                           goal_pose_list,
                           approach_direction_list=None,
                           approach_distance_list=None,
                           depart_direction_list=None,
                           depart_distance_list=None,
                           depart_jaw_width=None,
                           pick_jaw_width=None,
                           pick_approach_direction=None,  # handz
                           pick_approach_distance=None,
                           pick_depart_direction=None,  # handz
                           pick_depart_distance=None,
                           linear_granularity=.02,
                           use_rrt=True,
                           obstacle_list=None,
                           reason_grasps=True):
        """
        :param obj_cmodel:
        :param end_jnt_values:
        :param grasp_collection: grasping.grasp.GraspCollection
        :param goal_pose_list:
        :param approach_direction_list:
        :param approach_distance_list:
        :param depart_direction_list:
        :param depart_distance_list:
        :param approach_jaw_width:
        :param depart_jaw_width:
        :param ad_granularity:
        :param use_rrt:
        :param obstacle_list:
        :param reason_grasps: examine grasps sequentially in case of False
        :return:
        author: weiwei
        date: 20191122, 20200105, 20240317
        """
        ## picking parameters
        if pick_jaw_width is None:
            pick_jaw_width = self.robot.end_effector.jaw_range[1]
        if pick_approach_distance is None:
            pick_approach_distance = .07
        if pick_depart_distance is None:
            pick_depart_distance = .07
        ## approach depart parameters
        if depart_jaw_width is None:
            depart_jaw_width = self.robot.end_effector.jaw_range[1]
        if approach_direction_list is None:
            approach_direction_list = [-rm.const.z_ax] * len(goal_pose_list)
        if approach_distance_list is None:
            approach_distance_list = [.07] * len(goal_pose_list)
        if depart_direction_list is None:
            depart_direction_list = [rm.const.z_ax] * len(goal_pose_list)
        if depart_distance_list is None:
            depart_distance_list = [.07] * len(goal_pose_list)
        if reason_grasps:
            common_gid_list = self.reason_common_gids(grasp_collection=grasp_collection,
                                                      goal_pose_list=[obj_cmodel.pose] + goal_pose_list,
                                                      obstacle_list=obstacle_list,
                                                      toggle_dbg=False)
        else:
            common_gid_list = range(len(grasp_collection))
        if len(common_gid_list) == 0:
            print("No common grasp id at the given goal poses!")
            return None
        for gid in common_gid_list:
            obj_cmodel_copy = obj_cmodel.copy()
            pm_mot = self.gen_pick_and_moveto(obj_cmodel=obj_cmodel_copy,
                                              grasp=grasp_collection[gid],
                                              goal_pose_list=goal_pose_list,
                                              approach_direction_list=approach_direction_list,
                                              approach_distance_list=approach_distance_list,
                                              depart_direction_list=depart_direction_list,
                                              depart_distance_list=depart_distance_list[:-1] + [0],
                                              pick_jaw_width=pick_jaw_width,
                                              pick_approach_direction=pick_approach_direction,
                                              pick_approach_distance=pick_approach_distance,
                                              pick_depart_direction=pick_depart_direction,
                                              pick_depart_distance=pick_depart_distance,
                                              linear_granularity=linear_granularity,
                                              obstacle_list=obstacle_list,
                                              use_rrt=use_rrt)
            if pm_mot is None:
                print("Cannot generate the pick and moveto motion!")
                continue
            # place
            last_goal_pos = goal_pose_list[-1][0]
            last_goal_rotmat = goal_pose_list[-1][1]
            obj_cmodel_copy.pose = (last_goal_pos, last_goal_rotmat)
            dep_mot = self.gen_depart_from_given_conf(start_jnt_values=pm_mot.jv_list[-1],
                                                      end_jnt_values=end_jnt_values,
                                                      linear_direction=depart_direction_list[-1],
                                                      linear_distance=depart_distance_list[-1],
                                                      ee_values=depart_jaw_width,
                                                      granularity=linear_granularity,
                                                      obstacle_list=obstacle_list,
                                                      object_list=[obj_cmodel_copy],
                                                      use_rrt=use_rrt)
            if dep_mot is None:
                print("Cannot generate the release motion!")
                continue
            return pm_mot + dep_mot
        print("None of the reasoned common grasps are valid.")
        return None


if __name__ == '__main__':
    import time
    import numpy as np
    import wrs.basis.robot_math as rm
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm
    import wrs.modeling.collision_model as mcm
    import wrs.grasping.annotation.gripping as gutil

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    mgm.gen_frame().attach_to(base)
    obj_cmodel = cm.CollisionModel(initor='tubebig.stl')
    obj_cmodel.pos = np.array([.45, -.2, .2])
    obj_cmodel.rotmat = np.eye(3)
    obj_cmodel_copy = obj_cmodel.copy()
    obj_cmodel_copy.rgb = rm.const.orange
    obj_cmodel_copy.attach_to(base)
    robot = ym.Yumi(enable_cc=True)
    robot.use_rgt()
    start_conf = robot.get_jnt_values()
    n_goal = 2
    goal_pose_list = []
    for i in range(n_goal):
        goal_pos = np.array([.4, -.2, .1]) - np.array([0, i * .1, .0])
        goal_rotmat = np.eye(3)
        goal_pose_list.append((goal_pos, goal_rotmat))
        tmp_objcm = obj_cmodel.copy()
        tmp_objcm.rgba = np.array([1, 0, 0, .3])
        tmp_objcm.homomat = rm.homomat_from_posrot(goal_pos, goal_rotmat)
        tmp_objcm.attach_to(base)
    grasp_info_list = gutil.load_pickle_file(file_name='yumi_gripper_tube_big.pickle')
    grasp_info = grasp_info_list[0]
    pp_planner = PickPlacePlanner(robot=robot)
    # robot.gen_meshmodel().attach_to(base)
    # print(obj_cmodel.pose)
    # robot.hold(obj_cmodel)
    # print(obj_cmodel.pose)
    # robot.gen_meshmodel(rgb=rm.const.tab20_list[3], alpha=1).attach_to(base)
    # robot.goto_given_conf(robot.get_jnt_values())
    # print(obj_cmodel.pose)
    # robot.gen_meshmodel(rgb=rm.const.tab20_list[6], alpha=.5).attach_to(base)
    # base.run()
    # for grasp_info in grasp_info_list:
    #     mot_data = pp_planner.gen_pick_and_moveto(obj_cmodel=obj_cmodel,
    #                                               grasp_info=grasp_info,
    #                                               goal_pose_list=goal_pose_list,
    #                                               approach_direction_list=[np.array([0, 0, -1])] * n_goal,
    #                                               approach_distance_list=[.07] * n_goal,
    #                                               depart_direction_list=[np.array([0, 0, 1])] * n_goal,
    #                                               depart_distance_list=[.07] * n_goal,
    #                                               pick_approach_direction=None,
    #                                               pick_approach_distance=.07,
    #                                               pick_depart_direction=None,
    #                                               pick_depart_distance=.07,
    #                                               linear_granularity=.02,
    #                                               obstacle_list=[],
    #                                               use_rrt=True)
    #     if mot_data is not None:
    #         break
    # base.toggle_mesh=False
    tic = time.time()
    mot_data = pp_planner.gen_pick_and_place(obj_cmodel=obj_cmodel,
                                             end_jnt_values=robot.get_jnt_values(),
                                             grasp_info_list=grasp_info_list,
                                             goal_pose_list=goal_pose_list,
                                             approach_direction_list=[np.array([0, 0, -1])] * n_goal,
                                             approach_distance_list=[.07] * n_goal,
                                             depart_direction_list=[np.array([0, 0, 1])] * n_goal,
                                             depart_distance_list=[.07] * n_goal,
                                             pick_approach_direction=None,
                                             pick_approach_distance=.07,
                                             pick_depart_direction=None,
                                             pick_depart_distance=.07,
                                             linear_granularity=.02,
                                             obstacle_list=[],
                                             use_rrt=True)
    toc = time.time()
    print(toc - tic)
    print(mot_data)


    class Data(object):
        def __init__(self, mot_data):
            self.counter = 0
            self.mot_data = mot_data


    anime_data = Data(mot_data)


    def update(anime_data, task):
        if anime_data.counter > 0:
            anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
        if anime_data.counter >= len(anime_data.mot_data):
            # for mesh_model in anime_data.mot_data.mesh_list:
            #     mesh_model.detach()
            anime_data.counter = 0
        mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
        mesh_model.attach_to(base)
        if base.inputmgr.keymap['space']:
            anime_data.counter += 1
        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[anime_data],
                          appendTask=True)

    base.run()


    class Data(object):
        def __init__(self, robot, arm, motion_data):
            self.robot_attached_list = []
            self.counter = 0
            # self.motion_data = approach_motion_data + depart_motion_data
            self.motion_data = motion_data
            self.robot = robot
            self.arm = arm


    anime_data = Data(robot, arm, motion_data)


    def update(anime_data, task):
        if anime_data.counter == 0:
            anime_data.arm.backup_state()
        if anime_data.counter >= len(anime_data.motion_data):
            anime_data.arm.restore_state()
            if len(anime_data.robot_attached_list) != 0:
                for robot_attached in anime_data.robot_attached_list:
                    robot_attached.detach()
            anime_data.robot_attached_list.clear()
            anime_data.counter = 0
            anime_data.arm.backup_state()
        if len(anime_data.robot_attached_list) > 1:
            for robot_attached in anime_data.robot_attached_list:
                robot_attached.detach()
        conf = anime_data.motion_data.conf_list[anime_data.counter]
        jaw_width = anime_data.motion_data.jaw_width_list[anime_data.counter]
        anime_data.arm.goto_given_conf(jnt_values=conf)
        if jaw_width is not None:
            if anime_data.motion_data.hold_list[anime_data.counter] is not None:
                obj_cmodel = anime_data.motion_data.hold_list[anime_data.counter]
                anime_data.arm.hold(obj_cmodel, jaw_width=jaw_width)
            elif anime_data.motion_data.release_list[anime_data.counter] is not None:
                obj_cmodel = anime_data.motion_data.release_list[anime_data.counter]
                anime_data.arm.release(obj_cmodel, jaw_width=jaw_width)
            else:
                anime_data.arm.change_jaw_width(jaw_width=jaw_width)
        robot_meshmodel = anime_data.robot.gen_meshmodel(toggle_cdprim=False, alpha=1)
        robot_meshmodel.attach_to(base)
        anime_data.robot_attached_list.append(robot_meshmodel)
        if base.inputmgr.keymap['space']:
            anime_data.counter += 1

        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[anime_data],
                          appendTask=True)
    base.run()

    robot.rgt_arm.goto_given_conf(jnt_values=motion_data.conf_list[0])
    robot.rgt_arm.hold(obj_cmodel)
    for i, conf in enumerate(conf_list[1:]):
        robot.rgt_arm.goto_given_conf(jnt_values=conf)
        robot.rgt_arm.gen_meshmodel().attach_to(base)
        base.run()

    # # animation
    # robot_attached_list = []
    # object_attached_list = []
    # counter = [0]
    #
    # def update(robot_s,
    #            hand_name,
    #            obj_cmodel,
    #            robot_path,
    #            jawwidth_path,
    #            obj_path,
    #            robot_attached_list,
    #            object_attached_list,
    #            counter,
    #            task):
    #     if counter[0] >= len(robot_path):
    #         counter[0] = 0
    #     if len(robot_attached_list) != 0:
    #         for robot_attached in robot_attached_list:
    #             robot_attached.detach()
    #         for object_attached in object_attached_list:
    #             object_attached.detach()
    #         robot_attached_list.clear()
    #         object_attached_list.clear()
    #     pose = robot_path[counter[0]]
    #     robot_s.fk(hand_name, pose)
    #     robot_s.change_jaw_width(hand_name, jawwidth_path[counter[0]])
    #     robot_meshmodel = robot_s.gen_mesh_model()
    #     robot_meshmodel.attach_to(base)
    #     robot_attached_list.append(robot_meshmodel)
    #     obj_pose = obj_path[counter[0]]
    #     objb_copy = obj_cmodel.copy()
    #     objb_copy.set_homomat(obj_pose)
    #     objb_copy.attach_to(base)
    #     object_attached_list.append(objb_copy)
    #     counter[0] += 1
    #     return task.again
    #
    #
    # taskMgr.doMethodLater(0.01, update, "update",
    #                       extraArgs=[robot_s,
    #                                  hand_name,
    #                                  obj_cmodel,
    #                                  conf_list,
    #                                  jawwidth_list,
    #                                  objpose_list,
    #                                  robot_attached_list,
    #                                  object_attached_list,
    #                                  counter],
    #                       appendTask=True)
    base.run()
