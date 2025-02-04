import wrs.basis.robot_math as rm
import wrs.motion.primitives.approach_depart_planner as adp
import wrs.robot_sim.robots.yumi.yumi as ym
import wrs.motion.motion_data as motd


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
                            moveto_pose_list,
                            moveto_approach_direction_list,
                            moveto_approach_distance_list,
                            moveto_depart_direction_list,
                            moveto_depart_distance_list,
                            start_jnt_values=None,
                            pick_approach_jaw_width=None,
                            pick_approach_direction=None,
                            pick_approach_distance=.07,
                            pick_depart_direction=None,
                            pick_depart_distance=.07,
                            linear_granularity=.02,
                            obstacle_list=None,
                            use_rrt=True,
                            toggle_dbg=False):
        """
        pick and move an object to multiple poses
        :param obj_cmodel:
        :param grasp:
        :param moveto_pose_list: [[pos, rotmat], [pos, rotmat], ...]
        :param moveto_approach_direction_list:
        :param moveto_approach_distance_list:
        :param moveto_depart_direction_list:
        :param moveto_depart_distance_list:
        :param start_jnt_values: None means starting from the linear end of the pick motion
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
        if pick_approach_jaw_width is None:
            pick_approach_jaw_width = self.robot.end_effector.jaw_range[1]
        pick = self.gen_approach(goal_tcp_pos=pick_tcp_pos,
                                     goal_tcp_rotmat=pick_tcp_rotmat,
                                     start_jnt_values=start_jnt_values,
                                     linear_direction=pick_approach_direction,
                                     linear_distance=pick_approach_distance,
                                     ee_values=pick_approach_jaw_width,
                                     linear_granularity=linear_granularity,
                                     obstacle_list=obstacle_list,
                                     object_list=[obj_cmodel],
                                     use_rrt=use_rrt,
                                     toggle_dbg=toggle_dbg)
        if pick is None:
            print("PPPlanner: Error encountered when generating pick approach motion!")
            return None
        obj_cmodel_copy = obj_cmodel.copy()
        for robot_mesh in pick.mesh_list:
            obj_cmodel_copy.attach_to(robot_mesh)
        self.robot.goto_given_conf(pick.jv_list[-1])
        self.robot.hold(obj_cmodel=obj_cmodel_copy, jaw_width=grasp.ee_values)
        pick.extend(jv_list=[pick.jv_list[-1]])
        pick_depart = self.gen_linear_depart_from_given_conf(start_jnt_values=pick.jv_list[-1],
                                                                 direction=pick_depart_direction,
                                                                 distance=pick_depart_distance,
                                                                 ee_values=None,
                                                                 granularity=linear_granularity,
                                                                 obstacle_list=obstacle_list,
                                                                 toggle_dbg=toggle_dbg)
        if pick_depart is None:
            print("PPPlanner: Error encountered when generating pick depart motion!")
            return None
        else:
            moveto = adp.mpi.motd.MotionData(robot=self.robot)
            # move to goals
            moveto_start_jnt_values = pick_depart.jv_list[-1]
            for i, goal_pose in enumerate(moveto_pose_list):
                goal_tcp_pos = goal_pose[1].dot(grasp.ac_pos) + goal_pose[0]
                goal_tcp_rotmat = goal_pose[1].dot(grasp.ac_rotmat)
                moveto_ap = self.gen_approach_depart(goal_tcp_pos=goal_tcp_pos,
                                                     goal_tcp_rotmat=goal_tcp_rotmat,
                                                     start_jnt_values=moveto_start_jnt_values,
                                                     approach_direction=moveto_approach_direction_list[i],
                                                     approach_distance=moveto_approach_distance_list[i],
                                                     approach_ee_values=None,
                                                     depart_direction=moveto_depart_direction_list[i],
                                                     depart_distance=moveto_depart_distance_list[i],
                                                     depart_ee_values=None,  # do not change jaw width
                                                     linear_granularity=linear_granularity,
                                                     obstacle_list=obstacle_list,
                                                     use_rrt=use_rrt,
                                                     toggle_dbg=toggle_dbg)
                if moveto_ap is None:
                    print(f"Error encountered when generating motion to the {i}th goal!")
                    return None
                else:
                    moveto_ap.obj_cmodel = obj_cmodel.copy()
                    moveto_ap.obj_pose_list = [obj_cmodel.pose] * len(moveto_ap.jv_list)
                    moveto += moveto_ap
                    moveto_start_jnt_values = moveto.jv_list[-1]
            return pick + pick_depart + moveto

    @adp.mpi.InterplatedMotion.keep_states_decorator
    def gen_pick_and_place(self,
                           obj_cmodel,
                           grasp_collection,
                           goal_pose_list,
                           start_jnt_values=None,
                           end_jnt_values=None,
                           pick_approach_jaw_width=None,
                           pick_approach_direction=None,  # handz
                           pick_approach_distance=None,
                           pick_depart_direction=None,  # handz
                           pick_depart_distance=None,
                           place_approach_direction_list=None,
                           place_approach_distance_list=None,
                           place_depart_direction_list=None,
                           place_depart_distance_list=None,
                           place_depart_jaw_width=None,
                           linear_granularity=.02,
                           use_rrt=True,
                           obstacle_list=None,
                           reason_grasps=True,
                           toggle_dbg=False):
        """
        :param obj_cmodel:
        :param grasp_collection: grasping.grasp.GraspCollection
        :param goal_pose_list:
        :param start_jnt_values: start from the start of pick approach if None
        :param end_jnt_values: end at the end of place depart if None
        :param pick_approach_jaw_width: default value if None
        :param pick_approach_direction: handz if None
        :param pick_approach_distance:
        :param pick_depart_direction: handz if None
        :param pick_depart_distance:
        :param place_approach_direction_list:
        :param place_approach_distance_list:
        :param place_depart_direction_list:
        :param place_depart_distance_list:
        :param place_depart_jaw_width:
        :param linear_granularity:
        :param use_rrt:
        :param obstacle_list:
        :param reason_grasps: examine grasps sequentially in case of False
        :return:
        author: weiwei
        date: 20191122, 20200105, 20240317
        """
        ## picking parameters
        if pick_approach_jaw_width is None:
            pick_approach_jaw_width = self.robot.end_effector.jaw_range[1]
        if pick_approach_distance is None:
            pick_approach_distance = .07
        if pick_depart_distance is None:
            pick_depart_distance = .07
        ## approach depart parameters
        if place_depart_jaw_width is None:
            place_depart_jaw_width = self.robot.end_effector.jaw_range[1]
        if place_approach_direction_list is None:
            place_approach_direction_list = [-rm.const.z_ax] * len(goal_pose_list)
        if place_approach_distance_list is None:
            place_approach_distance_list = [.07] * len(goal_pose_list)
        if place_depart_direction_list is None:
            place_depart_direction_list = [rm.const.z_ax] * len(goal_pose_list)
        if place_depart_distance_list is None:
            place_depart_distance_list = [.07] * len(goal_pose_list)
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
            pick_and_moveto = self.gen_pick_and_moveto(obj_cmodel=obj_cmodel_copy,
                                                       grasp=grasp_collection[gid],
                                                       moveto_pose_list=goal_pose_list,
                                                       moveto_approach_direction_list=place_approach_direction_list,
                                                       moveto_approach_distance_list=place_approach_distance_list,
                                                       moveto_depart_direction_list=place_depart_direction_list,
                                                       moveto_depart_distance_list=place_depart_distance_list[:-1] + [
                                                           0],
                                                       start_jnt_values=start_jnt_values,
                                                       pick_approach_jaw_width=pick_approach_jaw_width,
                                                       pick_approach_direction=pick_approach_direction,
                                                       pick_approach_distance=pick_approach_distance,
                                                       pick_depart_direction=pick_depart_direction,
                                                       pick_depart_distance=pick_depart_distance,
                                                       linear_granularity=linear_granularity,
                                                       obstacle_list=obstacle_list,
                                                       use_rrt=use_rrt,
                                                       toggle_dbg=toggle_dbg)
            if pick_and_moveto is None:
                print("Cannot generate the pick and moveto motion!")
                continue
            # place
            last_goal_pos = goal_pose_list[-1][0]
            last_goal_rotmat = goal_pose_list[-1][1]
            obj_cmodel_copy.pose = (last_goal_pos, last_goal_rotmat)
            depart = self.gen_depart_from_given_conf(start_jnt_values=pick_and_moveto.jv_list[-1],
                                                     end_jnt_values=end_jnt_values,
                                                     linear_direction=place_depart_direction_list[-1],
                                                     linear_distance=place_depart_distance_list[-1],
                                                     ee_values=place_depart_jaw_width,
                                                     linear_granularity=linear_granularity,
                                                     obstacle_list=obstacle_list,
                                                     object_list=[obj_cmodel_copy],
                                                     use_rrt=use_rrt,
                                                     toggle_dbg=toggle_dbg)
            if depart is None:
                print("Cannot generate the release motion!")
                continue
            for robot_mesh in depart.mesh_list:
                obj_cmodel_copy.attach_to(robot_mesh)
            return pick_and_moveto + depart
        print("None of the reasoned common grasps are valid.")
        return None

    @adp.mpi.InterplatedMotion.keep_states_decorator
    def gen_pick_and_moveto_with_given_conf(self,
                                            obj_cmodel,
                                            pick_jnt_values,
                                            moveto_jnt_values,
                                            start_jnt_values=None,
                                            pick_approach_jaw_width=None,
                                            pick_approach_direction=None,
                                            pick_approach_distance=.07,
                                            pick_depart_direction=None,
                                            pick_depart_distance=.07,
                                            grasp_jaw_width=.0,
                                            moveto_approach_direction=None,
                                            moveto_approach_distance=.07,
                                            linear_granularity=.02,
                                            obstacle_list=None,
                                            use_rrt=True,
                                            toggle_dbg=False):
        """
        pick and move an object to multiple poses
        :param obj_cmodel:
        :param pick_jnt_values:
        :param moveto_jnt_values:
        :param start_jnt_values: None means starting from the linear end of the pick motion
        :param pick_approach_direction
        :param pick_approach_distance
        :param pick_depart_direction
        :param pick_depart_distance
        :param moveto_approach_direction
        :param moveto_approach_distance
        :param linear_granularity:
        :param obstacle_list:
        :param seed_jnt_values:
        :return:
        """
        # pick up object
        pick = self.gen_approach_to_given_conf(goal_jnt_values=pick_jnt_values,
                                                      start_jnt_values=start_jnt_values,
                                                      linear_direction=pick_approach_direction,
                                                      linear_distance=pick_approach_distance,
                                                      ee_values=pick_approach_jaw_width,
                                                      linear_granularity=linear_granularity,
                                                      obstacle_list=obstacle_list,
                                                      object_list=[obj_cmodel],
                                                      use_rrt=use_rrt,
                                                      toggle_dbg=toggle_dbg)
        if pick is None:
            print("PPPlanner: Error encountered when generating pick approach motion!")
            return None
        obj_cmodel_copy = obj_cmodel.copy()
        for robot_mesh in pick.mesh_list:
            obj_cmodel_copy.attach_to(robot_mesh)
        self.robot.goto_given_conf(pick.jv_list[-1])
        self.robot.hold(obj_cmodel=obj_cmodel_copy, jaw_width=grasp_jaw_width)
        pick.extend(jv_list = [pick.jv_list[-1]], ev_list = [grasp_jaw_width])
        moveto = self.gen_depart_approach_with_given_conf(start_jnt_values=pick.jv_list[-1],
                                                          end_jnt_values=moveto_jnt_values,
                                                          depart_direction=pick_depart_direction,
                                                          depart_distance=pick_depart_distance,
                                                          depart_ee_values=grasp_jaw_width,
                                                          approach_direction=moveto_approach_direction,
                                                          approach_distance=moveto_approach_distance,
                                                          approach_ee_values=grasp_jaw_width,
                                                          linear_granularity=linear_granularity,
                                                          obstacle_list=obstacle_list,
                                                          use_rrt=use_rrt,
                                                          toggle_dbg=toggle_dbg)
        if moveto is None:
            print("PPPlanner: Error encountered when generating depart approach motion with given conf!")
            return None
        else:
            return pick + moveto


if __name__ == '__main__':
    import time
    from wrs import wd, rm, mgm, mcm, gg

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    mgm.gen_frame().attach_to(base)
    obj_cmodel = mcm.CollisionModel(initor='tubebig.stl')
    obj_cmodel.pos = rm.np.array([.55, 0, .2])
    obj_cmodel.rotmat = rm.eye(3)
    # obj_cmodel_copy = obj_cmodel.copy()
    # obj_cmodel_copy.rgb = rm.const.orange
    # obj_cmodel_copy.attach_to(base)
    robot = ym.Yumi(enable_cc=True)
    robot.use_rgt()
    start_conf = robot.get_jnt_values()
    n_goal = 2
    goal_pose_list = []
    for i in range(n_goal):
        goal_pos = rm.np.array([.6, -.2, .1]) - rm.np.array([0, i * .1, .0])
        goal_rotmat = rm.eye(3)
        goal_pose_list.append((goal_pos, goal_rotmat))
        tmp_objcm = obj_cmodel.copy()
        tmp_objcm.rgba = rm.np.array([1, 0, 0, .3])
        tmp_objcm.homomat = rm.homomat_from_posrot(goal_pos, goal_rotmat)
        tmp_objcm.attach_to(base)
    grasp_collection = gg.GraspCollection.load_from_disk(file_name='yumi_gripper_tube_big.pickle')
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
    #                                               moveto_pose_list=moveto_pose_list,
    #                                               moveto_approach_direction_list=[rm.np.array([0, 0, -1])] * n_goal,
    #                                               moveto_approach_distance_list=[.07] * n_goal,
    #                                               moveto_depart_direction_list=[rm.np.array([0, 0, 1])] * n_goal,
    #                                               moveto_depart_distance_list=[.07] * n_goal,
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
                                             grasp_collection=grasp_collection,
                                             goal_pose_list=goal_pose_list,
                                             place_approach_direction_list=[rm.np.array([0, 0, -1])] * n_goal,
                                             place_approach_distance_list=[.07] * n_goal,
                                             place_depart_direction_list=[rm.np.array([0, 0, 1])] * n_goal,
                                             place_depart_distance_list=[.07] * n_goal,
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
