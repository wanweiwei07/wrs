import math
import numpy as np
import basis.robot_math as rm
import basis.data_adapter as da
import motion.optimization_based.incremental_nik as inik
import motion.probabilistic.rrt_connect as rrtc
import manipulation.approach_depart_planner as adp
import motion.utils as mu


class PickPlacePlanner(adp.ADPlanner):

    def __init__(self, sgl_arm_robot):
        """
        :param object:
        :param robot_helper:
        author: weiwei, hao
        date: 20191122, 20210113
        """
        super().__init__(sgl_arm_robot)

    def gen_object_motion(self, conf_list, obj_pos, obj_rotmat, type='absolute'):
        """
        :param conf_list:
        :param obj_pos:
        :param obj_rotmat:
        :param type: 'absolute' or 'relative'
        :return:
        author: weiwei
        date: 20210125
        """
        objpose_list = []
        if type == 'absolute':
            for _ in conf_list:
                objpose_list.append(rm.homomat_from_posrot(obj_pos, obj_rotmat))
        elif type == 'relative':
            jnt_values_bk = self.robot_s.get_jnt_values(component_name)
            for conf in conf_list:
                self.robot_s.fk(component_name, conf)
                gl_obj_pos, gl_obj_rotmat = self.robot_s.cvt_loc_tcp_to_gl(component_name, obj_pos, obj_rotmat)
                objpose_list.append(rm.homomat_from_posrot(gl_obj_pos, gl_obj_rotmat))
            self.robot_s.fk(component_name, jnt_values_bk)
        else:
            raise ValueError('Type must be absolute or relative!')
        return objpose_list

    @mu.keep_jnt_values_decorator
    def find_common_gids(self,
                         grasp_info_list,
                         goal_homomat_list,
                         obstacle_list=[],
                         toggle_dbg=False):
        """
        find the common collision free and IK feasible gids
        :param eef: an end effector instance
        :param grasp_info_list: a list like [[jaw_width, jaw_center_pos, jaw_center_rotmat, pos, rotmat], ...]
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
                self.robot.end_effector.grip_at_by_pose(goal_jaw_center_pos, goal_jaw_center_rotmat, jaw_width)
                if not self.robot.end_effector.is_mesh_collided(obstacle_list):  # gripper cd
                    jnt_values = self.robot.ik(goal_jaw_center_pos, goal_jaw_center_rotmat)
                    if jnt_values is not None:  # common graspid with robot_s ik
                        # if toggle_dbg:
                        #     sgl_arm_robot.end_effector.gen_meshmodel(rgb=rm.bc.green, alhpa=.3).attach_to(base)
                        self.robot.goto_given_conf(jnt_values=jnt_values)
                        is_rbt_collided = self.robot.is_collided(obstacle_list=obstacle_list)  # robot cd
                        # TODO is_obj_collided
                        is_obj_collided = False  # obj cd
                        if (not is_rbt_collided) and (not is_obj_collided):  # gripper cdf, rbt ikf/cdf, obj cdf
                            if toggle_dbg:
                                self.robot.end_effector.gen_mesh_model(rgb=rm.bc.green, alpha=.3).attach_to(base)
                            previously_available_gids.append(gid)
                        elif (not is_obj_collided):  # gripper cdf, rbt ikf, rbt collided
                            rbt_collided_grasps_num += 1
                            if toggle_dbg:
                                self.robot.end_effector.gen_mesh_model(rgb=rm.bc.yellow, alpha=.3).attach_to(base)
                    else:  # gripper cdf, ik infeasible
                        ik_failed_grasps_num += 1
                        if toggle_dbg:
                            self.robot.end_effector.gen_mesh_model(rgba=rm.bc.orange, alpha=.3).attach_to(base)
                else:  # gripper collided
                    eef_collided_grasps_num += 1
                    if toggle_dbg:
                        self.robot.end_effector.gen_mesh_model(rgba=rm.bc.magenta, alpha=.3).attach_to(base)
            intermediate_available_gids.append(previously_available_gids.copy())
            print('-----start-----')
            print('Number of collided grasps at goal-' + str(goal_id) + ': ', eef_collided_grasps_num)
            print('Number of failed IK at goal-' + str(goal_id) + ': ', ik_failed_grasps_num)
            print('Number of collided robots at goal-' + str(goal_id) + ': ', rbt_collided_grasps_num)
            print('------end_type------')
        final_available_gids = previously_available_gids
        return final_available_gids, intermediate_available_gids

    @mu.keep_jnt_jaw_objpose_values_decorator
    def gen_pick_and_move_motion(self,
                                 obj_cmodel,
                                 grasp_info,
                                 obj_pose_list,
                                 depart_vec_list,
                                 depart_dist_list,
                                 approach_vec_list,
                                 approach_dist_list,
                                 ad_granularity=.007,
                                 use_rrt=True,
                                 obstacle_list=[]):
        """
        pick and move an object to multiple poses
        :param obj_cmodel:
        :param grasp_info:
        :param obj_pose_list:
        :param depart_vec_list: the last element will be ignored
        :param depart_dist_list: the last element will be ignored
        :param approach_vec_list: the first element will be ignored
        :param approach_dist_list: the first element will be ignored
        :param ad_granularity:
        :param obstacle_list:
        :param seed_jnt_values:
        :return:
        """
        # final
        conf_list = []
        jaw_width_list = []
        # hold object
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        first_obj_pos = obj_pose_list[0][:3, 3]
        first_obj_rotmat = obj_pose_list[0][:3, :3]
        first_jaw_center_pos = first_obj_rotmat.dot(jaw_center_pos) + first_obj_pos
        first_jaw_center_rotmat = first_obj_rotmat.dot(jaw_center_rotmat)
        first_conf = self.robot.ik(first_jaw_center_pos,
                                   first_jaw_center_rotmat)
        if first_conf is None:
            print("Cannot solve the ik at the first grasping pose!")
            return None, None
        self.robot.goto_given_conf(jnt_values=first_conf)
        self.robot.hold(obj_cmodel, jaw_width=jaw_width)
        # # set a copy of the object to the start pose, hold the object, and move it to goal object pose
        # objcm_copy = obj_cmodel.copy()
        # objcm_copy.set_pos(first_obj_pos)
        # objcm_copy.set_rotmat(first_obj_rotmat)
        # rel_obj_pos, rel_obj_rotmat = self.robot_s.hold(hand_name, objcm_copy, jaw_width)
        seed_conf = first_conf
        for i in range(len(obj_pose_list) - 1):
            # get start and goal object poses
            start_obj_pos = obj_pose_list[i][:3, 3]
            start_obj_rotmat = obj_pose_list[i][:3, :3]
            goal_obj_pos = obj_pose_list[i + 1][:3, 3]
            goal_obj_rotmat = obj_pose_list[i + 1][:3, :3]
            # transform grasps
            start_jaw_center_pos = start_obj_rotmat.dot(jaw_center_pos) + start_obj_pos
            start_jaw_center_rotmat = start_obj_rotmat.dot(jaw_center_rotmat)
            goal_jaw_center_pos = goal_obj_rotmat.dot(jaw_center_pos) + goal_obj_pos
            goal_jaw_center_rotmat = goal_obj_rotmat.dot(jaw_center_rotmat)
            depart_vec = depart_vec_list[i]
            if depart_vec is None:
                depart_vec = -start_jaw_center_rotmat[:, 2]
            depart_dist = depart_dist_list[i]
            if depart_dist is None:
                depart_dist = 0
            approach_vec = approach_vec_list[i + 1]
            if approach_vec is None:
                approach_vec = goal_jaw_center_rotmat[:, 2]
            approach_dist = approach_dist_list[i + 1]
            if approach_dist is None:
                approach_dist = 0
            # depart linear
            depart_conf_list = self.im_planner.gen_rel_linear_motion(goal_tcp_pos=start_jaw_center_pos,
                                                                     goal_tcp_rotmat=start_jaw_center_rotmat,
                                                                     motion_vec=depart_vec,
                                                                     motion_dist=depart_dist,
                                                                     obstacle_list=obstacle_list,
                                                                     granularity=ad_granularity,
                                                                     seed_jnt_values=seed_conf,
                                                                     type='source')
            if depart_conf_list is None:
                print(f"Cannot generate the linear part of the {i}th holding depart motion!")
                self.robot.release(obj_cmodel)
                return None, None
            depart_jaw_width_list = self.gen_jaw_width_list(depart_conf_list, jaw_width)
            if use_rrt:  # if use rrt, we shall find start and goal conf first and then perform rrt
                # approach linear
                seed_conf = depart_conf_list[-1]
                approach_conf_list = self.im_planner.gen_rel_linear_motion(goal_tcp_pos=goal_jaw_center_pos,
                                                                           goal_tcp_rotmat=goal_jaw_center_rotmat,
                                                                           motion_vec=approach_vec,
                                                                           motion_dist=approach_dist,
                                                                           obstacle_list=obstacle_list,
                                                                           granularity=ad_granularity,
                                                                           seed_jnt_values=seed_conf,
                                                                           type='sink')
                if approach_conf_list is None:
                    print(f"Cannot generate the linear part of the {i}th holding approach motion!")
                    self.robot.release(obj_cmodel)
                    return None, None
                mid_conf_list = self.rrtc_planner.plan(start_conf=depart_conf_list[-1],
                                                       goal_conf=approach_conf_list[0],
                                                       obstacle_list=obstacle_list,
                                                       other_robot_list=[],
                                                       ext_dist=.07,
                                                       max_n_iter=300)
                if mid_conf_list is None:
                    print(f"Cannot generate the rrtc part of the {i}th holding approach motion!")
                    self.robot.release(obj_cmodel)
                    return None, None
            else:
                # if do not use rrt, we start from depart end to approach start, and then approach to the goal
                mid_start_tcp_pos, mid_start_tcp_rotmat = self.robot.fk(jnt_values=depart_conf_list[-1])
                mid_goal_tcp_pos = goal_jaw_center_pos - approach_vec * approach_dist
                mid_goal_tcp_rotmat = goal_jaw_center_rotmat
                mid_conf_list = self.im_planner.gen_linear_motion(start_tcp_pos=mid_start_tcp_pos,
                                                                  start_tcp_rotmat=mid_start_tcp_rotmat,
                                                                  goal_tcp_pos=mid_goal_tcp_pos,
                                                                  goal_tcp_rotmat=mid_goal_tcp_rotmat,
                                                                  obstacle_list=obstacle_list,
                                                                  granularity=ad_granularity,
                                                                  seed_jnt_values=seed_conf)
                if mid_conf_list is None:
                    print(f"Cannot generate the {i}th holding switching motion!")
                    self.robot.release(obj_cmodel)
                    return None, None
                # approach linear
                seed_conf = mid_conf_list[-1]
                approach_conf_list = self.im_planner.gen_rel_linear_motion(goal_tcp_pos=goal_jaw_center_pos,
                                                                           goal_tcp_rotmat=goal_jaw_center_rotmat,
                                                                           motion_vec=approach_vec,
                                                                           motion_dist=approach_dist,
                                                                           obstacle_list=obstacle_list,
                                                                           granularity=ad_granularity,
                                                                           seed_jnt_values=seed_conf,
                                                                           type='sink')
                if approach_conf_list is None:
                    print(f"Cannot generate the linear part of the {i}th holding approach motion!")
                    self.robot.release(obj_cmodel)
                    return None, None
            approach_jaw_width_list = self.gen_jaw_width_list(approach_conf_list, jaw_width)
            mid_jaw_width_list = self.gen_jaw_width_list(mid_conf_list, jaw_width)
            conf_list = conf_list + depart_conf_list + mid_conf_list + approach_conf_list
            jaw_width_list = jaw_width_list + depart_jaw_width_list + mid_jaw_width_list + approach_jaw_width_list
            seed_conf = conf_list[-1]
        return conf_list, jaw_width_list

    def gen_pick_and_place_motion(self,
                                  hnd_name,
                                  objcm,
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

        :param hnd_name:
        :param objcm:
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
                self.gen_approach_motion(component_name=hnd_name,
                                         goal_tcp_pos=first_jaw_center_pos,
                                         goal_tcp_rotmat=first_jaw_center_rotmat,
                                         start_conf=start_conf,
                                         approach_direction=approach_direction_list[0],
                                         approach_distance=approach_distance_list[0],
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
                self.gen_pick_and_move_motion(hand_name=hnd_name,
                                              objcm=objcm,
                                              grasp_info=grasp_info,
                                              obj_pose_list=goal_homomat_list,
                                              depart_vec_list=depart_direction_list,
                                              approach_vec_list=approach_direction_list,
                                              depart_dist_list=depart_distance_list,
                                              approach_dist_list=approach_distance_list,
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
                self.gen_depart_motion(component_name=hnd_name,
                                       start_tcp_pos=last_jaw_center_pos,
                                       start_tcp_rotmat=last_jaw_center_rotmat,
                                       end_conf=end_conf,
                                       depart_direction=depart_direction_list[0],
                                       depart_distance=depart_distance_list[0],
                                       depart_jaw_width=depart_jawwidth,
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
    import time
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
    robot = ym.Yumi(enable_cc=True)
    start_conf = robot.rgt_arm.get_jnt_values()
    goal_homomat_list = []
    for i in range(3):
        goal_pos = np.array([.55, -.1, .3]) - np.array([i * .1, i * .1, 0])
        # goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
        goal_rotmat = np.eye(3)
        goal_homomat_list.append(rm.homomat_from_posrot(goal_pos, goal_rotmat))
        tmp_objcm = obj_cmodel.copy()
        tmp_objcm.rgba = np.array([1, 0, 0, .3])
        tmp_objcm.homomat = rm.homomat_from_posrot(goal_pos, goal_rotmat)
        tmp_objcm.attach_to(base)
    grasp_info_list = gutil.load_pickle_file(cmodel_name='tubebig', file_name='yumi_tube_big.pickle')
    grasp_info = grasp_info_list[0]
    pp_planner = PickPlacePlanner(sgl_arm_robot=robot.rgt_arm)
    # conf_list, jawwidth_list, objpose_list = \
    #     pp_planner.gen_pick_and_place_motion(hnd_name=hand_name,
    #                                          obj_cmodel=obj_cmodel,
    #                                          grasp_info_list=grasp_info_list,
    #                                          goal_homomat_list=goal_homomat_list,
    #                                          start_conf=robot_s.get_jnt_values(hand_name),
    #                                          end_conf=robot_s.get_jnt_values(hand_name),
    #                                          depart_direction_list=[np.array([0, 0, 1])] * len(goal_homomat_list),
    #                                          approach_direction_list=[np.array([0, 0, -1])] * len(goal_homomat_list),
    #                                          # depart_dist_list=[None] * len(goal_homomat_list),
    #                                          # approach_dist_list=[None] * len(goal_homomat_list),
    #                                          depart_distance_list=[.2] * len(goal_homomat_list),
    #                                          approach_distance_list=[.2] * len(goal_homomat_list),
    #                                          approach_jawwidth=None,
    #                                          depart_jawwidth=None,
    #                                          ad_granularity=.003,
    #                                          use_rrt=True,
    #                                          obstacle_list=[],
    #                                          use_incremental=False)
    for grasp_info in grasp_info_list:
        conf_list, jaw_width_list = \
            pp_planner.gen_pick_and_move_motion(obj_cmodel=obj_cmodel,
                                                grasp_info=grasp_info,
                                                obj_pose_list=goal_homomat_list,
                                                depart_vec_list=[np.array([0, 0, 1])] * len(goal_homomat_list),
                                                approach_vec_list=[np.array([0, 0, -1])] * len(goal_homomat_list),
                                                # depart_dist_list=[None] * len(goal_homomat_list),
                                                # approach_dist_list=[None] * len(goal_homomat_list),
                                                depart_dist_list=[.2] * len(goal_homomat_list),
                                                approach_dist_list=[.2] * len(goal_homomat_list),
                                                ad_granularity=.003,
                                                use_rrt=True,
                                                obstacle_list=[])
        if conf_list is not None:
            break

    robot.rgt_arm.goto_given_conf(jnt_values=conf_list[0])
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
