import math
import numpy as np
import basis.robot_math as rm
import basis.data_adapter as da
import motion.optimization_based.incremental_nik as inik
import motion.probabilistic.rrt_connect as rrtc
import manipulation.approach_depart_planner as adp


class PickPlacePlanner(adp.ADPlanner):

    def __init__(self, robot_s):
        """
        :param object:
        :param robot_helper:
        author: weiwei, hao
        date: 20191122, 20210113
        """
        super().__init__(robot_s)

    def gen_object_motion(self, component_name, conf_list, obj_pos, obj_rotmat, type='absolute'):
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

    def find_common_graspids(self,
                             hand_name,  # TODO hnd is on  a manipulator
                             grasp_info_list,
                             goal_homomat_list,
                             obstacle_list=[],
                             toggle_debug=False):
        """
        find the common collision free and IK feasible graspids
        :param hand_name: a component may have multiple hands
        :param grasp_info_list: a list like [[jaw_width, gl_jaw_center_pos, pos, rotmat], ...]
        :param goal_homomat_list: [homomat, ...]
        :param obstacle_list
        :return: [final_available_graspids, intermediate_available_graspids]
        author: weiwei
        date: 20210113, 20210125
        """
        hnd_instance = self.robot_s.hnd_dict[hand_name]
        # start reasoning
        previously_available_graspids = range(len(grasp_info_list))
        intermediate_available_graspids = []
        hndcollided_grasps_num = 0
        ikfailed_grasps_num = 0
        rbtcollided_grasps_num = 0
        jnt_values_bk = self.robot_s.get_jnt_values(hand_name)
        for goalid, goal_homomat in enumerate(goal_homomat_list):
            goal_pos = goal_homomat[:3, 3]
            goal_rotmat = goal_homomat[:3, :3]
            graspid_and_graspinfo_list = zip(previously_available_graspids,  # need .copy()?
                                             [grasp_info_list[i] for i in previously_available_graspids])
            previously_available_graspids = []
            for graspid, grasp_info in graspid_and_graspinfo_list:
                jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
                goal_jaw_center_pos = goal_pos + goal_rotmat.dot(jaw_center_pos)
                goal_jaw_center_rotmat = goal_rotmat.dot(jaw_center_rotmat)
                hnd_instance.grip_at_with_jcpose(goal_jaw_center_pos, goal_jaw_center_rotmat, jaw_width)
                if not hnd_instance.is_mesh_collided(obstacle_list):  # hnd_s cd
                    jnt_values = self.robot_s.ik(hand_name, goal_jaw_center_pos, goal_jaw_center_rotmat)
                    if jnt_values is not None:  # common graspid with robot_s ik
                        if toggle_debug:
                            hnd_tmp = hnd_instance.copy()
                            hnd_tmp.gen_meshmodel(rgba=[0, 1, 0, .2]).attach_to(base)
                        self.robot_s.fk(hand_name, jnt_values)
                        is_rbt_collided = self.robot_s.is_collided(obstacle_list)  # robot_s cd
                        # TODO is_obj_collided
                        is_obj_collided = False  # obj cd
                        if (not is_rbt_collided) and (not is_obj_collided):  # hnd cdfree, rbs ikf/cdfree, obj cdfree
                            if toggle_debug:
                                self.robot_s.gen_meshmodel(rgba=[0, 1, 0, .5]).attach_to(base)
                            previously_available_graspids.append(graspid)
                        elif (not is_obj_collided):  # hnd_s cdfree, robot_s ikfeasible, robot_s collided
                            rbtcollided_grasps_num += 1
                            if toggle_debug:
                                self.robot_s.gen_meshmodel(rgba=[1, 0, 1, .5]).attach_to(base)
                    else:  # hnd_s cdfree, robot_s ik infeasible
                        ikfailed_grasps_num += 1
                        if toggle_debug:
                            hnd_tmp = hnd_instance.copy()
                            hnd_tmp.gen_meshmodel(rgba=[1, .6, 0, .2]).attach_to(base)
                else:  # hnd_s collided
                    hndcollided_grasps_num += 1
                    if toggle_debug:
                        hnd_tmp = hnd_instance.copy()
                        hnd_tmp.gen_meshmodel(rgba=[1, 0, 1, .2]).attach_to(base)
            intermediate_available_graspids.append(previously_available_graspids.copy())
            print('-----start-----')
            print('Number of collided grasps at goal-' + str(goalid) + ': ', hndcollided_grasps_num)
            print('Number of failed IK at goal-' + str(goalid) + ': ', ikfailed_grasps_num)
            print('Number of collided robots at goal-' + str(goalid) + ': ', rbtcollided_grasps_num)
            print('------end------')
        final_available_graspids = previously_available_graspids
        self.robot_s.fk(hand_name, jnt_values_bk)
        return final_available_graspids, intermediate_available_graspids

    def gen_holding_rel_linear(self):
        pass

    def gen_holding_linear(self):
        pass

    def gen_holding_moveto(self,
                           hand_name,
                           objcm,
                           grasp_info,
                           obj_pose_list,
                           depart_direction_list,
                           depart_distance_list,
                           approach_direction_list,
                           approach_distance_list,
                           ad_granularity=.007,
                           use_rrt=True,
                           obstacle_list=[],
                           seed_jnt_values=None):
        """
        hold and move an object to multiple poses
        :param hand_name:
        :param grasp_info:
        :param obj_pose_list:
        :param depart_direction_list: the last element will be ignored
        :param depart_distance_list: the last element will be ignored
        :param approach_direction_list: the first element will be ignored
        :param approach_distance_list: the first element will be ignored
        :param ad_granularity:
        :param obstacle_list:
        :param seed_jnt_values:
        :return:
        """
        jnt_values_bk = self.robot_s.get_jnt_values(hand_name)
        jawwidth_bk = self.robot_s.get_jawwidth(hand_name)
        # final
        conf_list = []
        jawwidthlist = []
        objpose_list = []
        # hold object
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        first_obj_pos = obj_pose_list[0][:3, 3]
        first_obj_rotmat = obj_pose_list[0][:3, :3]
        first_jaw_center_pos = first_obj_rotmat.dot(jaw_center_pos) + first_obj_pos
        first_jaw_center_rotmat = first_obj_rotmat.dot(jaw_center_rotmat)
        first_conf = self.robot_s.ik(hand_name,
                                     first_jaw_center_pos,
                                     first_jaw_center_rotmat,
                                     seed_jnt_values=seed_jnt_values)
        if first_conf is None:
            print("Cannot solve the ik at the first grasping pose!")
            return None, None, None
        self.robot_s.fk(component_name=hand_name, jnt_values=first_conf)
        # set a copy of the object to the start pose, hold the object, and move it to goal object pose
        objcm_copy = objcm.copy()
        objcm_copy.set_pos(first_obj_pos)
        objcm_copy.set_rotmat(first_obj_rotmat)
        rel_obj_pos, rel_obj_rotmat = self.robot_s.hold(hand_name, objcm_copy, jaw_width)
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
            depart_direction = depart_direction_list[i]
            if depart_direction is None:
                depart_direction = -start_jaw_center_rotmat[:, 2]
            depart_distance = depart_distance_list[i]
            if depart_distance is None:
                depart_distance = 0
            approach_direction = approach_direction_list[i + 1]
            if approach_direction is None:
                approach_direction = goal_jaw_center_rotmat[:, 2]
            approach_distance = approach_distance_list[i + 1]
            if approach_distance is None:
                approach_distance = 0
            # depart linear
            conf_list_depart = self.inik_slvr.gen_rel_linear_motion(component_name=hand_name,
                                                                    goal_tcp_pos=start_jaw_center_pos,
                                                                    goal_tcp_rotmat=start_jaw_center_rotmat,
                                                                    direction=depart_direction,
                                                                    distance=depart_distance,
                                                                    obstacle_list=obstacle_list,
                                                                    granularity=ad_granularity,
                                                                    seed_jnt_values=seed_conf,
                                                                    type='source')
            if conf_list_depart is None:
                print(f"Cannot generate the linear part of the {i}th holding depart motion!")
                self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
                self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                return None, None, None
            jawwidthlist_depart = self.gen_jawwidth_motion(conf_list_depart, jaw_width)
            objpose_list_depart = self.gen_object_motion(component_name=hand_name,
                                                         conf_list=conf_list_depart,
                                                         obj_pos=rel_obj_pos,
                                                         obj_rotmat=rel_obj_rotmat,
                                                         type='relative')
            if use_rrt:  # if use rrt, we shall find start and goal conf first and then perform rrt
                # approach linear
                seed_conf = conf_list_depart[-1]
                conf_list_approach = self.inik_slvr.gen_rel_linear_motion(component_name=hand_name,
                                                                          goal_tcp_pos=goal_jaw_center_pos,
                                                                          goal_tcp_rotmat=goal_jaw_center_rotmat,
                                                                          direction=approach_direction,
                                                                          distance=approach_distance,
                                                                          obstacle_list=obstacle_list,
                                                                          granularity=ad_granularity,
                                                                          seed_jnt_values=seed_conf,
                                                                          type='sink')
                if conf_list_approach is None:
                    print(f"Cannot generate the linear part of the {i}th holding approach motion!")
                    self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
                    self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                    return None, None, None
                conf_list_middle = self.rrtc_planner.plan(component_name=hand_name,
                                                          start_conf=conf_list_depart[-1],
                                                          goal_conf=conf_list_approach[0],
                                                          obstacle_list=obstacle_list,
                                                          otherrobot_list=[],
                                                          ext_dist=.07,
                                                          max_iter=300)
                if conf_list_middle is None:
                    print(f"Cannot generate the rrtc part of the {i}th holding approach motion!")
                    self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
                    self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                    return None, None, None
            else:  # if do not use rrt, we start from depart end to mid end and then approach from mid end to goal
                seed_conf = conf_list_depart[-1]
                self.robot_s.fk(component_name=hand_name, jnt_values=seed_conf)
                mid_start_tcp_pos, mid_start_tcp_rotmat = self.robot_s.get_gl_tcp(hand_name)
                mid_goal_tcp_pos = goal_jaw_center_pos - approach_direction * approach_distance
                mid_goal_tcp_rotmat = goal_jaw_center_rotmat
                conf_list_middle = self.inik_slvr.gen_linear_motion(component_name=hand_name,
                                                                    start_tcp_pos=mid_start_tcp_pos,
                                                                    start_tcp_rotmat=mid_start_tcp_rotmat,
                                                                    goal_tcp_pos=mid_goal_tcp_pos,
                                                                    goal_tcp_rotmat=mid_goal_tcp_rotmat,
                                                                    obstacle_list=obstacle_list,
                                                                    granularity=ad_granularity,
                                                                    seed_jnt_values=seed_conf)
                if conf_list_middle is None:
                    print(f"Cannot generate the rrtc part of the {i}th holding approach motion!")
                    self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
                    self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                    return None, None, None
                # approach linear
                seed_conf = conf_list_middle[-1]
                conf_list_approach = self.inik_slvr.gen_rel_linear_motion(component_name=hand_name,
                                                                          goal_tcp_pos=goal_jaw_center_pos,
                                                                          goal_tcp_rotmat=goal_jaw_center_rotmat,
                                                                          direction=approach_direction,
                                                                          distance=approach_distance,
                                                                          obstacle_list=obstacle_list,
                                                                          granularity=ad_granularity,
                                                                          seed_jnt_values=seed_conf,
                                                                          type='sink')
                if conf_list_approach is None:
                    print(f"Cannot generate the linear part of the {i}th holding approach motion!")
                    self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
                    self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                    return None, None, None
            jawwidthlist_approach = self.gen_jawwidth_motion(conf_list_approach, jaw_width)
            objpose_list_approach = self.gen_object_motion(component_name=hand_name,
                                                           conf_list=conf_list_approach,
                                                           obj_pos=rel_obj_pos,
                                                           obj_rotmat=rel_obj_rotmat,
                                                           type='relative')
            jawwidthlist_middle = self.gen_jawwidth_motion(conf_list_middle, jaw_width)
            objpose_list_middle = self.gen_object_motion(component_name=hand_name,
                                                         conf_list=conf_list_middle,
                                                         obj_pos=rel_obj_pos,
                                                         obj_rotmat=rel_obj_rotmat,
                                                         type='relative')
            conf_list = conf_list + conf_list_depart + conf_list_middle + conf_list_approach
            jawwidthlist = jawwidthlist + jawwidthlist_depart + jawwidthlist_middle + jawwidthlist_approach
            objpose_list = objpose_list + objpose_list_depart + objpose_list_middle + objpose_list_approach
            seed_conf = conf_list[-1]
        self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
        self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
        return conf_list, jawwidthlist, objpose_list

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
        :param approach_direction_list: the first element will be the pick approach direction
        :param approach_distance_list: the first element will be the pick approach direction
        :param depart_direction_list: the last element will be the release depart direction
        :param depart_distance_list: the last element will be the release depart direction
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
            approach_jawwidth = self.robot_s.hnd_dict[hnd_name].jawwidth_rng[1]
        if depart_jawwidth is None:
            depart_jawwidth = self.robot_s.hnd_dict[hnd_name].jawwidth_rng[1]
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
            # objcm as an obstacle
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
                self.gen_holding_moveto(hand_name=hnd_name,
                                        objcm=objcm,
                                        grasp_info=grasp_info,
                                        obj_pose_list=goal_homomat_list,
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
            # objcm as an obstacle
            objcm_copy.set_pos(last_goal_pos)
            objcm_copy.set_rotmat(last_goal_rotmat)
            conf_list_depart, jawwidthlist_depart = \
                self.gen_depart_motion(component_name=hnd_name,
                                       start_tcp_pos=last_jaw_center_pos,
                                       start_tcp_rotmat=last_jaw_center_rotmat,
                                       end_conf=end_conf,
                                       depart_direction=depart_direction_list[0],
                                       depart_distance=depart_distance_list[0],
                                       depart_jawwidth=depart_jawwidth,
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
    objcm = cm.CollisionModel('tubebig.stl')
    robot_s = ym.Yumi(enable_cc=True)
    manipulator_name = 'rgt_arm'
    hand_name = 'rgt_hnd'
    start_conf = robot_s.get_jnt_values(manipulator_name)
    goal_homomat_list = []
    for i in range(6):
        goal_pos = np.array([.55, -.1, .3]) - np.array([i * .1, i * .1, 0])
        # goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
        goal_rotmat = np.eye(3)
        goal_homomat_list.append(rm.homomat_from_posrot(goal_pos, goal_rotmat))
        tmp_objcm = objcm.copy()
        tmp_objcm.set_rgba([1, 0, 0, .3])
        tmp_objcm.set_homomat(rm.homomat_from_posrot(goal_pos, goal_rotmat))
        tmp_objcm.attach_to(base)
    grasp_info_list = gutil.load_pickle_file(objcm_name='tubebig', file_name='yumi_tube_big.pickle')
    grasp_info = grasp_info_list[0]
    pp_planner = PickPlacePlanner(robot_s=robot_s)
    conf_list, jawwidth_list, objpose_list = \
        pp_planner.gen_pick_and_place_motion(hnd_name=hand_name,
                                             objcm=objcm,
                                             grasp_info_list=grasp_info_list,
                                             goal_homomat_list=goal_homomat_list,
                                             start_conf=robot_s.get_jnt_values(hand_name),
                                             end_conf=robot_s.get_jnt_values(hand_name),
                                             depart_direction_list=[np.array([0, 0, 1])] * len(goal_homomat_list),
                                             approach_direction_list=[np.array([0, 0, -1])] * len(goal_homomat_list),
                                             # depart_distance_list=[None] * len(goal_homomat_list),
                                             # approach_distance_list=[None] * len(goal_homomat_list),
                                             depart_distance_list=[.2] * len(goal_homomat_list),
                                             approach_distance_list=[.2] * len(goal_homomat_list),
                                             approach_jawwidth=None,
                                             depart_jawwidth=None,
                                             ad_granularity=.003,
                                             use_rrt=True,
                                             obstacle_list=[],
                                             use_incremental=False)
    # for grasp_info in grasp_info_list:
    #     conf_list, jawwidth_list, objpose_list = \
    #         pp_planner.gen_holding_moveto(hnd_name=hnd_name,
    #                                       objcm=objcm,
    #                                       grasp_info=grasp_info,
    #                                       obj_pose_list=goal_homomat_list,
    #                                       depart_direction_list=[np.array([0, 0, 1])] * len(goal_homomat_list),
    #                                       approach_direction_list=[np.array([0, 0, -1])] * len(goal_homomat_list),
    #                                       # depart_distance_list=[None] * len(goal_homomat_list),
    #                                       # approach_distance_list=[None] * len(goal_homomat_list),
    #                                       depart_distance_list=[.2] * len(goal_homomat_list),
    #                                       approach_distance_list=[.2] * len(goal_homomat_list),
    #                                       ad_granularity=.003,
    #                                       use_rrt=True,
    #                                       obstacle_list=[],
    #                                       seed_jnt_values=start_conf)
    #     print(robot_s.rgt_oih_infos, robot_s.lft_oih_infos)
    #     if conf_list is not None:
    #         break

    # animation
    robot_attached_list = []
    object_attached_list = []
    counter = [0]


    def update(robot_s,
               hand_name,
               objcm,
               robot_path,
               jawwidth_path,
               obj_path,
               robot_attached_list,
               object_attached_list,
               counter,
               task):
        if counter[0] >= len(robot_path):
            counter[0] = 0
        if len(robot_attached_list) != 0:
            for robot_attached in robot_attached_list:
                robot_attached.detach()
            for object_attached in object_attached_list:
                object_attached.detach()
            robot_attached_list.clear()
            object_attached_list.clear()
        pose = robot_path[counter[0]]
        robot_s.fk(hand_name, pose)
        robot_s.jaw_to(hand_name, jawwidth_path[counter[0]])
        robot_meshmodel = robot_s.gen_meshmodel()
        robot_meshmodel.attach_to(base)
        robot_attached_list.append(robot_meshmodel)
        obj_pose = obj_path[counter[0]]
        objb_copy = objcm.copy()
        objb_copy.set_homomat(obj_pose)
        objb_copy.attach_to(base)
        object_attached_list.append(objb_copy)
        counter[0] += 1
        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[robot_s,
                                     hand_name,
                                     objcm,
                                     conf_list,
                                     jawwidth_list,
                                     objpose_list,
                                     robot_attached_list,
                                     object_attached_list,
                                     counter],
                          appendTask=True)
    base.run()
