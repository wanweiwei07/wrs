import motion.primitives.approach_depart_planner as adp


class PickPlacePlanner(adp.ADPlanner):

    def __init__(self, robot):
        """
        :param object:
        :param robot: must be an instance of SglArmRobotInterface
        author: weiwei, hao
        date: 20191122, 20210113, 20240316
        """
        super().__init__(robot)

    @staticmethod
    def keep_obj_decorator(method):
        """
        decorator function for save and restore objects
        applicable to both single or multi-arm sgl_arm_robots
        :return:
        author: weiwei
        date: 20220404
        """

        def wrapper(self, obj_cmodel, **kwargs):
            obj_pose_bk = obj_cmodel.pose
            result = method(self, obj_cmodel, **kwargs)
            obj_cmodel.pose = obj_pose_bk
            return result

        return wrapper

    @adp.mpi.InterplatedMotion.keep_states_decorator
    def find_common_gids(self,
                         grasp_info_list,
                         goal_homomat_list,
                         obstacle_list=[],
                         toggle_dbg=False):
        """
        find the common collision free and IK feasible gids
        :param eef: an end effector instance
        :param grasp_info_list: a list like [[ee_values, jaw_center_pos, jaw_center_rotmat, pos, rotmat], ...]
        :param goal_homomat_list: [homomat0, homomat1,, ...]
        :param obstacle_list
        :return: [final_available_graspids, intermediate_available_graspids]
        author: weiwei
        date: 20210113, 20210125
        """
        # start reasoning
        previously_available_gids = range(len(grasp_info_list))
        intermediate_available_gids = []
        eef_collided_grasps_num = 0
        ik_failed_grasps_num = 0
        rbt_collided_grasps_num = 0
        for goal_id, goal_homomat in enumerate(goal_homomat_list):
            goal_pos = goal_homomat[:3, 3]
            goal_rotmat = goal_homomat[:3, :3]
            gidinfo_list = zip(previously_available_gids,  # need .copy()?
                               [grasp_info_list[i] for i in previously_available_gids])
            previously_available_gids = []
            for gid, ginfo in gidinfo_list:
                jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = ginfo
                goal_jaw_center_pos = goal_pos + goal_rotmat.dot(jaw_center_pos)
                goal_jaw_center_rotmat = goal_rotmat.dot(jaw_center_rotmat)
                self.sgl_arm_robot.end_effector.grip_at_by_pose(goal_jaw_center_pos, goal_jaw_center_rotmat, jaw_width)
                if not self.sgl_arm_robot.end_effector.is_mesh_collided(obstacle_list):  # gripper cd
                    jnt_values = self.sgl_arm_robot.ik(goal_jaw_center_pos, goal_jaw_center_rotmat)
                    if jnt_values is not None:  # common graspid with robot_s ik
                        # if toggle_dbg:
                        #     robot.end_effector.gen_meshmodel(rgb=rm.bc.green, alhpa=.3).attach_to(base)
                        self.sgl_arm_robot.goto_given_conf(jnt_values=jnt_values)
                        is_rbt_collided = self.sgl_arm_robot.is_collided(obstacle_list=obstacle_list)  # robot cd
                        # TODO is_obj_collided
                        is_obj_collided = False  # obj cd
                        if (not is_rbt_collided) and (not is_obj_collided):  # gripper cdf, rbt ikf/cdf, obj cdf
                            if toggle_dbg:
                                self.sgl_arm_robot.end_effector.gen_mesh_model(rgb=rm.bc.green, alpha=.3).attach_to(
                                    base)
                            previously_available_gids.append(gid)
                        elif (not is_obj_collided):  # gripper cdf, rbt ikf, rbt collided
                            rbt_collided_grasps_num += 1
                            if toggle_dbg:
                                self.sgl_arm_robot.end_effector.gen_mesh_model(rgb=rm.bc.yellow, alpha=.3).attach_to(
                                    base)
                    else:  # gripper cdf, ik infeasible
                        ik_failed_grasps_num += 1
                        if toggle_dbg:
                            self.sgl_arm_robot.end_effector.gen_mesh_model(rgba=rm.bc.orange, alpha=.3).attach_to(base)
                else:  # gripper collided
                    eef_collided_grasps_num += 1
                    if toggle_dbg:
                        self.sgl_arm_robot.end_effector.gen_mesh_model(rgba=rm.bc.magenta, alpha=.3).attach_to(base)
            intermediate_available_gids.append(previously_available_gids.copy())
            print('-----start-----')
            print('Number of collided grasps at goal-' + str(goal_id) + ': ', eef_collided_grasps_num)
            print('Number of failed IK at goal-' + str(goal_id) + ': ', ik_failed_grasps_num)
            print('Number of collided robots at goal-' + str(goal_id) + ': ', rbt_collided_grasps_num)
            print('------end_type------')
        final_available_gids = previously_available_gids
        return final_available_gids, intermediate_available_gids

    @keep_obj_decorator
    @adp.mpi.InterplatedMotion.keep_states_decorator
    def gen_pick_and_moveto(self,
                            obj_cmodel,
                            grasp_info,
                            goal_pose_list,
                            approach_direction_list,
                            approach_distance_list,
                            depart_direction_list,
                            depart_distance_list,
                            pick_approach_direction=None,
                            pick_approach_distance=.07,
                            pick_depart_direction=None,
                            pick_depart_distance=.07,
                            linear_granularity=.02,
                            obstacle_list=[],
                            use_rrt=True):
        """
        pick and move an object to multiple poses
        :param obj_cmodel:
        :param grasp_info:
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
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        pick_tcp_pos = obj_cmodel.rotmat.dot(jaw_center_pos) + obj_cmodel.pos
        pick_tcp_rotmat = obj_cmodel.rotmat.dot(jaw_center_rotmat)
        pick_motion = self.gen_approach(goal_tcp_pos=pick_tcp_pos,
                                        goal_tcp_rotmat=pick_tcp_rotmat,
                                        start_jnt_values=self.robot.get_jnt_values(),
                                        linear_direction=pick_approach_direction,
                                        linear_distance=pick_approach_distance,
                                        ee_values=self.robot.end_effector.jaw_range[1],
                                        granularity=linear_granularity,
                                        obstacle_list=obstacle_list,
                                        object_list=[obj_cmodel],
                                        use_rrt=use_rrt)
        if pick_motion is None:
            print("PPPlanner: Error encountered when generating pick approach motion!")
            return None
        else:
            self.robot.goto_given_conf(pick_motion.jv_list[-1])
            # self.robot.gen_meshmodel().attach_to(base)
            self.robot.hold(obj_cmodel=obj_cmodel, jaw_width=jaw_width)
            # self.robot.gen_meshmodel().attach_to(base)
            # print("before back up")
            pick_motion.extend([pick_motion.jv_list[-1]], [jaw_width], [self.robot.gen_meshmodel()])
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
                moveto_motion = adp.mpi.motu.MotionData(robot=self.robot)
                # move to goals
                moveto_start_jnt_values = pick_depart.jv_list[-1]
                for i, goal_pose in enumerate(goal_pose_list):
                    goal_tcp_pos = goal_pose[1].dot(jaw_center_pos) + goal_pose[0]
                    goal_tcp_rotmat = goal_pose[1].dot(jaw_center_rotmat)
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
                                                         object_list=[],
                                                         use_rrt=use_rrt)
                    if moveto_ap is None:
                        print(f"Error encountered when generating motion to the {i}th goal!")
                        return None
                    else:
                        moveto_motion += moveto_ap
                        moveto_start_jnt_values = moveto_motion.jv_list[-1]
                return pick_motion + pick_depart + moveto_motion

    @keep_obj_decorator
    @adp.mpi.InterplatedMotion.keep_states_decorator
    def gen_pick_and_place_motion(self,
                                  obj_cmodel,
                                  start_conf,
                                  end_conf,
                                  grasp_info_list,
                                  goal_homomat_list,
                                  approach_direction_list,
                                  approach_distance_list,
                                  depart_direction_list,
                                  depart_distance_list,
                                  approach_jawwidth=None,
                                  depart_jawwidth=None,
                                  ad_granularity=.007,
                                  use_rrt=True,
                                  obstacle_list=[],
                                  use_incremental=False):
        """
        :param obj_cmodel:
        :param grasp_info_list:
        :param goal_homomat_list:
        :param start_conf: RRT motion between start_state and pre_approach; No RRT motion if None
        :param end_conf: RRT motion between post_depart and end_conf; Noe RRT motion if None
        :param approach_direction_list: the first element will be the pick approach motion_vec
        :param approach_distance_list: the first element will be the pick approach motion_vec
        :param depart_direction_list: the last element will be the release depart motion_vec
        :param depart_distance_list: the last element will be the release depart motion_vec
        :param approach_jawwidth:
        :param depart_jawwidth:
        :param ad_granularity:
        :param use_rrt:
        :param obstacle_list:
        :param use_incremental:
        :return:
        author: weiwei
        date: 20191122, 20200105
        """
        if approach_jawwidth is None:
            approach_jawwidth = self.robot_s.hnd_dict[hnd_name].jaw_range[1]
        if depart_jawwidth is None:
            depart_jawwidth = self.robot_s.hnd_dict[hnd_name].jaw_range[1]
        first_goal_pos = goal_homomat_list[0][:3, 3]
        first_goal_rotmat = goal_homomat_list[0][:3, :3]
        last_goal_pos = goal_homomat_list[-1][:3, 3]
        last_goal_rotmat = goal_homomat_list[-1][:3, :3]
        if use_incremental:
            common_grasp_id_list = range(len(grasp_info_list))
        else:
            common_grasp_id_list, _ = self.find_common_graspids(hnd_name,
                                                                grasp_info_list,
                                                                goal_homomat_list)
        if len(common_grasp_id_list) == 0:
            print("No common grasp id at the given goal homomats!")
            return None, None, None
        for grasp_id in common_grasp_id_list:
            grasp_info = grasp_info_list[grasp_id]
            jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
            # approach
            first_jaw_center_pos = first_goal_rotmat.dot(jaw_center_pos) + first_goal_pos
            first_jaw_center_rotmat = first_goal_rotmat.dot(jaw_center_rotmat)
            # obj_cmodel as an obstacle
            objcm_copy = objcm.copy()
            objcm_copy.set_pos(first_goal_pos)
            objcm_copy.set_rotmat(first_goal_rotmat)
            conf_list_approach, jawwidthlist_approach = \
                self.gen_approach(component_name=hnd_name,
                                  goal_tcp_pos=first_jaw_center_pos,
                                  goal_tcp_rotmat=first_jaw_center_rotmat,
                                  start_conf=start_conf,
                                  linear_direction=approach_direction_list[0],
                                  linear_distance=approach_distance_list[0],
                                  approach_jawwidth=approach_jawwidth,
                                  granularity=ad_granularity,
                                  obstacle_list=obstacle_list,
                                  object_list=[objcm_copy],
                                  seed_jnt_values=start_conf)
            if conf_list_approach is None:
                print("Cannot generate the pick motion!")
                continue
            # middle
            conf_list_middle, jawwidthlist_middle, objpose_list_middle = \
                self.gen_pick_and_moveto(hand_name=hnd_name,
                                         objcm=objcm,
                                         grasp_info=grasp_info,
                                         goal_pose_list=goal_homomat_list,
                                         depart_direction_list=depart_direction_list,
                                         approach_direction_list=approach_direction_list,
                                         depart_distance_list=depart_distance_list,
                                         approach_distance_list=approach_distance_list,
                                         ad_granularity=.003,
                                         use_rrt=use_rrt,
                                         obstacle_list=obstacle_list,
                                         seed_jnt_values=conf_list_approach[-1])
            if conf_list_middle is None:
                continue
            # departure
            last_jaw_center_pos = last_goal_rotmat.dot(jaw_center_pos) + last_goal_pos
            last_jaw_center_rotmat = last_goal_rotmat.dot(jaw_center_rotmat)
            # obj_cmodel as an obstacle
            objcm_copy.set_pos(last_goal_pos)
            objcm_copy.set_rotmat(last_goal_rotmat)
            conf_list_depart, jawwidthlist_depart = \
                self.gen_depart(component_name=hnd_name,
                                start_tcp_pos=last_jaw_center_pos,
                                start_tcp_rotmat=last_jaw_center_rotmat,
                                end_conf=end_conf,
                                linear_direction=depart_direction_list[0],
                                linear_distance=depart_distance_list[0],
                                ee_values=depart_jawwidth,
                                granularity=ad_granularity,
                                obstacle_list=obstacle_list,
                                object_list=[objcm_copy],
                                seed_jnt_values=conf_list_middle[-1])
            if conf_list_depart is None:
                print("Cannot generate the release motion!")
                continue
            objpose_list_approach = self.gen_object_motion(component_name=hnd_name,
                                                           conf_list=jawwidthlist_approach,
                                                           obj_pos=first_goal_pos,
                                                           obj_rotmat=first_goal_rotmat,
                                                           type='absolute')
            objpose_list_depart = self.gen_object_motion(component_name=hnd_name,
                                                         conf_list=conf_list_depart,
                                                         obj_pos=last_goal_pos,
                                                         obj_rotmat=last_goal_rotmat,
                                                         type='absolute')
            return conf_list_approach + conf_list_middle + conf_list_depart, \
                   jawwidthlist_approach + jawwidthlist_middle + jawwidthlist_depart, \
                   objpose_list_approach + objpose_list_middle + objpose_list_depart
        return None, None, None


if __name__ == '__main__':
    import robot_sim.robots.yumi.yumi as ym
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import modeling.collision_model as cm
    import grasping.annotation.utils as gutil
    import numpy as np
    import basis.robot_math as rm

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)
    obj_cmodel = cm.CollisionModel(initor='tubebig.stl')
    obj_cmodel.pos = np.array([.55, -.15, .2])
    obj_cmodel.rotmat = np.eye(3)
    obj_cmodel.copy().attach_to(base)
    robot = ym.Yumi(enable_cc=True)
    robot.use_rgt()
    start_conf = robot.get_jnt_values()
    n_goal = 2
    goal_pose_list = []
    for i in range(n_goal):
        goal_pos = np.array([.45, -.2, 0]) - np.array([0, i * .1, 0])
        goal_rotmat = np.eye(3)
        goal_pose_list.append((goal_pos, goal_rotmat))
        tmp_objcm = obj_cmodel.copy()
        tmp_objcm.rgba = np.array([1, 0, 0, .3])
        tmp_objcm.homomat = rm.homomat_from_posrot(goal_pos, goal_rotmat)
        tmp_objcm.attach_to(base)
    grasp_info_list = gutil.load_pickle_file(cmodel_name='tubebig', file_name='yumi_gripper_tube_big.pickle')
    grasp_info = grasp_info_list[0]
    pp_planner = PickPlacePlanner(robot=robot)
    # robot.gen_meshmodel().attach_to(base)
    # print(obj_cmodel.pose)
    # robot.hold(obj_cmodel)
    # print(obj_cmodel.pose)
    # robot.gen_meshmodel(rgb=rm.bc.tab20_list[3], alpha=1).attach_to(base)
    # robot.goto_given_conf(robot.get_jnt_values())
    # print(obj_cmodel.pose)
    # robot.gen_meshmodel(rgb=rm.bc.tab20_list[6], alpha=.5).attach_to(base)
    # base.run()
    for grasp_info in grasp_info_list:
        mot_data = pp_planner.gen_pick_and_moveto(obj_cmodel=obj_cmodel,
                                                  grasp_info=grasp_info,
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
        if mot_data is not None:
            break

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
    #            objcm,
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
    #     objb_copy = objcm.copy()
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
