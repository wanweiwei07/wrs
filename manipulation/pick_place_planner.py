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
                             component_name,
                             hand_name,  # TODO hnd is on  a manipulator
                             grasp_info_list,
                             goal_homomat_list,
                             obstacle_list=[],
                             toggle_debug=False):
        """
        find the common collision free and IK feasible graspids
        :param component_name:
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
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
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
                if not hnd_instance.is_mesh_collided(obstacle_list):  # common graspid without considering robots
                    jnt_values = self.robot_s.ik(component_name, goal_jaw_center_pos, goal_jaw_center_rotmat)
                    if jnt_values is not None:  # common graspid consdiering robot_s ik
                        if toggle_debug:
                            hnd_tmp = hnd_instance.copy()
                            hnd_tmp.gen_meshmodel(rgba=[0, 1, 0, .2]).attach_to(base)
                        self.robot_s.fk(component_name, jnt_values)
                        is_rbt_collided = self.robot_s.is_collided(
                            obstacle_list)  # common graspid consdiering robot_s cd
                        # TODO is_obj_collided
                        is_obj_collided = False  # common graspid consdiering obj cd
                        if (not is_rbt_collided) and (
                                not is_obj_collided):  # hnd_s cdfree, robot_s ikfeasible, robot_s cdfree
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
        self.robot_s.fk(component_name, jnt_values_bk)
        return final_available_graspids, intermediate_available_graspids

    def gen_holding_rel_linear(self,
                               hand_name,
                               objcm,
                               grasp_info,
                               obj_pos,
                               obj_rotmat,
                               direction,
                               distance
                               ):
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
                           ad_linear_granularity=.003,
                           use_rrt = True,
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
        :param ad_linear_granularity:
        :param obstacle_list:
        :param seed_jnt_values:
        :return:
        """
        jnt_values_bk = self.robot_s.get_jnt_values(hand_name)
        jaw_width_bk = self.robot_s.get_jaw_width(hand_name)
        # final
        conf_list = []
        jaw_width_list = []
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
            approach_direction = approach_direction_list[i+1]
            if approach_direction is None:
                approach_direction = goal_jaw_center_rotmat[:, 2]
            approach_distance = approach_distance_list[i+1]
            if approach_distance is None:
                approach_distance = 0
            # depart linear
            conf_list_depart_linear = self.inik_slvr.gen_rel_linear_motion(component_name=hand_name,
                                                                           goal_tcp_pos=start_jaw_center_pos,
                                                                           goal_tcp_rotmat=start_jaw_center_rotmat,
                                                                           direction=depart_direction,
                                                                           distance=depart_distance,
                                                                           obstacle_list=obstacle_list,
                                                                           granularity=ad_linear_granularity,
                                                                           seed_jnt_values=seed_conf,
                                                                           type='source')
            if conf_list_depart_linear is None:
                print(f"Cannot generate the linear part of the {i}th holding depart motion!")
                self.robot_s.release(hand_name, objcm_copy, jaw_width_bk)
                self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                return None, None, None
            jaw_width_list_depart_linear = self.gen_jawwidth_motion(conf_list_depart_linear, jaw_width)
            objpose_list_depart_linear = self.gen_object_motion(component_name=hand_name,
                                                                conf_list=conf_list_depart_linear,
                                                                obj_pos=rel_obj_pos,
                                                                obj_rotmat=rel_obj_rotmat,
                                                                type='relative')
            if use_rrt: # if use rrt, we shall find start and goal conf first and then perform rrt
                # approach linear
                seed_conf = conf_list_depart_linear[-1]
                conf_list_approach_linear = self.inik_slvr.gen_rel_linear_motion(component_name=hand_name,
                                                                                 goal_tcp_pos=goal_jaw_center_pos,
                                                                                 goal_tcp_rotmat=goal_jaw_center_rotmat,
                                                                                 direction=approach_direction,
                                                                                 distance=approach_distance,
                                                                                 obstacle_list=obstacle_list,
                                                                                 granularity=ad_linear_granularity,
                                                                                 seed_jnt_values=seed_conf,
                                                                                 type='sink')
                if conf_list_approach_linear is None:
                    print(f"Cannot generate the linear part of the {i}th holding approach motion!")
                    self.robot_s.release(hand_name, objcm_copy, jaw_width_bk)
                    self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                    return None, None, None
                conf_list_middle = self.rrtc_planner.plan(component_name=hand_name,
                                                        start_conf=conf_list_depart_linear[-1],
                                                        goal_conf=conf_list_approach_linear[0],
                                                        obstacle_list=obstacle_list,
                                                        otherrobot_list=[],
                                                        ext_dist=.07,
                                                        max_iter=300)
                if conf_list_middle is None:
                    print(f"Cannot generate the rrtc part of the {i}th holding approach motion!")
                    self.robot_s.release(hand_name, objcm_copy, jaw_width_bk)
                    self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                    return None, None, None
            else: # if do not use rrt, we start from depart end to mid end and then approach from mid end to goal
                seed_conf = conf_list_depart_linear[-1]
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
                                                                    granularity=ad_linear_granularity,
                                                                    seed_jnt_values=seed_conf)
                if conf_list_middle is None:
                    print(f"Cannot generate the rrtc part of the {i}th holding approach motion!")
                    self.robot_s.release(hand_name, objcm_copy, jaw_width_bk)
                    self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                    return None, None, None
                # approach linear
                seed_conf = conf_list_middle[-1]
                conf_list_approach_linear = self.inik_slvr.gen_rel_linear_motion(component_name=hand_name,
                                                                                 goal_tcp_pos=goal_jaw_center_pos,
                                                                                 goal_tcp_rotmat=goal_jaw_center_rotmat,
                                                                                 direction=approach_direction,
                                                                                 distance=approach_distance,
                                                                                 obstacle_list=obstacle_list,
                                                                                 granularity=ad_linear_granularity,
                                                                                 seed_jnt_values=seed_conf,
                                                                                 type='sink')
                if conf_list_approach_linear is None:
                    print(f"Cannot generate the linear part of the {i}th holding approach motion!")
                    self.robot_s.release(hand_name, objcm_copy, jaw_width_bk)
                    self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                    return None, None, None
            jaw_width_list_approach_linear = self.gen_jawwidth_motion(conf_list_approach_linear, jaw_width)
            objpose_list_approach_linear = self.gen_object_motion(component_name=hand_name,
                                                                  conf_list=conf_list_approach_linear,
                                                                  obj_pos=rel_obj_pos,
                                                                  obj_rotmat=rel_obj_rotmat,
                                                                  type='relative')
            jaw_width_list_middle = self.gen_jawwidth_motion(conf_list_middle, jaw_width)
            objpose_list_middle = self.gen_object_motion(component_name=hand_name,
                                                       conf_list=conf_list_middle,
                                                       obj_pos=rel_obj_pos,
                                                       obj_rotmat=rel_obj_rotmat,
                                                       type='relative')
            conf_list = conf_list + conf_list_depart_linear + conf_list_middle + conf_list_approach_linear
            jaw_width_list = jaw_width_list + jaw_width_list_depart_linear + jaw_width_list_middle + jaw_width_list_approach_linear
            objpose_list = objpose_list + objpose_list_depart_linear + objpose_list_middle + objpose_list_approach_linear
            seed_conf = conf_list[-1]
        self.robot_s.release(hand_name, objcm_copy, jaw_width_bk)
        self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
        return conf_list, jaw_width_list, objpose_list

    def gen_approach_linear_with_object(self,
                                        hand_name,
                                        grasp_info,
                                        goal_obj_pos,
                                        goal_obj_rotmat,
                                        approach_direction=None,
                                        approach_distance=.1,
                                        approach_jawwidth=None,
                                        granularity=.03,
                                        obstacle_list=[],
                                        seed_jnt_values=None,
                                        toggle_end_grasp='False',
                                        end_jawwidth=.0):
        """
        wraps the adp gen approach linear function with objects returned together
        :param hand_name:
        :param objcm:
        :param grasp_info:
        :param goal_obj_pos:
        :param goal_obj_rotmat:
        :param approach_direction:
        :param approach_distance:
        :param granularity:
        :param seed_jnt_values:
        :param obstacle_list:
        :return:
        author: weiwei
        date: 20210511
        """
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        goal_jaw_center_pos = goal_obj_rotmat.dot(jaw_center_pos) + goal_obj_pos
        goal_jaw_center_rotmat = goal_obj_rotmat.dot(jaw_center_rotmat)
        if approach_direction is None:
            approach_direction = jaw_center_rotmat[:, 2]
        if approach_jawwidth is None:
            approach_jawwidth = self.robot_s.hnd_dict[hand_name].jaw_width_rng[1]
        conf_list, jawwidth_list = self.gen_approach_linear(hand_name,
                                                            goal_jaw_center_pos,
                                                            goal_jaw_center_rotmat,
                                                            approach_direction=approach_direction,
                                                            approach_distance=approach_distance,
                                                            approach_jawwidth=approach_jawwidth,
                                                            granularity=granularity,
                                                            obstacle_list=obstacle_list,
                                                            seed_jnt_values=seed_jnt_values,
                                                            toggle_end_grasp=toggle_end_grasp,
                                                            end_jawwidth=end_jawwidth)
        if conf_list is None:
            print("Cannot generate linear for approach with object!")
            return None, None, None
        objpose_list = self.gen_object_motion(hand_name, conf_list, goal_obj_pos, goal_obj_rotmat, type='absolute')
        return conf_list, jawwidth_list, objpose_list

    def gen_approach_motion_with_object(self,
                                        hand_name,
                                        grasp_info,
                                        goal_obj_pos,
                                        goal_obj_rotmat,
                                        start_conf=None,
                                        approach_direction=None,
                                        approach_distance=.1,
                                        approach_jawwidth=None,
                                        granularity=.03,
                                        obstacle_list=[],
                                        seed_jnt_values=None,
                                        toggle_end_grasp='False',
                                        end_jawwidth=.0):
        """
        wraps the
        :param hand_name:
        :param objcm:
        :param grasp_info:
        :param goal_obj_pos:
        :param goal_obj_rotmat:
        :param approach_direction:
        :param approach_distance:
        :param granularity:
        :param seed_jnt_values:
        :param obstacle_list:
        :return:
        author: weiwei
        date: 20210511
        """
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        goal_jaw_center_pos = goal_obj_rotmat.dot(jaw_center_pos) + goal_obj_pos
        goal_jaw_center_rotmat = goal_obj_rotmat.dot(jaw_center_rotmat)
        if approach_direction is None:
            approach_direction = jaw_center_rotmat[:, 2]
        if approach_jawwidth is None:
            approach_jawwidth = self.robot_s.hnd_dict[hand_name].jaw_width_rng[1]
        conf_list, jawwidth_list = self.gen_approach_motion(hand_name,
                                                            goal_jaw_center_pos,
                                                            goal_jaw_center_rotmat,
                                                            start_conf=start_conf,
                                                            approach_direction=approach_direction,
                                                            approach_distance=approach_distance,
                                                            approach_jawwidth=approach_jawwidth,
                                                            granularity=granularity,
                                                            obstacle_list=obstacle_list,
                                                            seed_jnt_values=seed_jnt_values,
                                                            toggle_end_grasp=toggle_end_grasp,
                                                            end_jawwidth=end_jawwidth)
        print(conf_list, jawwidth_list)
        if conf_list is None:
            print("Cannot generate motion for approach with object!")
            return None, None, None
        objpose_list = self.gen_object_motion(hand_name, conf_list, goal_obj_pos, goal_obj_rotmat, type='absolute')
        return conf_list, jawwidth_list, objpose_list

    def gen_holding_approach_linear(self,
                                    hnd_name,
                                    objcm,
                                    grasp_info,
                                    goal_obj_pos,
                                    goal_obj_rotmat,
                                    approach_direction=None,
                                    approach_distance=.1,
                                    granularity=.03,
                                    obstacle_list=[],
                                    seed_jnt_values=None):
        """
        :param hnd_name:
        :param objcm:
        :param grasp_info:
        :param goal_obj_pos:
        :param goal_obj_rotmat:
        :param approach_direction:
        :param approach_distance:
        :param granularity:
        :param seed_jnt_values:
        :param obstacle_list:
        :return:
        author: weiwei
        date: 20210511
        """
        # get start object pose
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        if approach_direction is None:
            approach_direction = jaw_center_rotmat[:, 2]
        start_obj_pos = goal_obj_pos - approach_direction * approach_distance
        start_obj_rotmat = goal_obj_rotmat
        # transform them to start and gaol object poses
        start_jaw_center_pos = start_obj_rotmat.dot(jaw_center_pos) + start_obj_pos
        start_jaw_center_rotmat = start_obj_rotmat.dot(jaw_center_rotmat)
        goal_jaw_center_pos = goal_obj_rotmat.dot(jaw_center_pos) + goal_obj_pos
        goal_jaw_center_rotmat = goal_obj_rotmat.dot(jaw_center_rotmat)
        # save current robot jnt values
        jnt_values_bk = self.robot_s.get_jnt_values(hnd_name)
        # find a robot conf that reaches to the start jaw pose, and move the robot there
        start_conf = self.robot_s.ik(hnd_name,
                                     start_jaw_center_pos,
                                     start_jaw_center_rotmat,
                                     seed_jnt_values=seed_jnt_values)
        self.robot_s.fk(start_conf)
        # set a copy of the object to the start pose, hold the object, and move it to goal object pose
        objcm_copy = objcm.copy()
        objcm_copy.set_pos(start_obj_pos)
        objcm_copy.set_rotmat(start_obj_rotmat)
        rel_obj_pos, rel_obj_rotmat = self.robot_s.hold(hnd_name, objcm_copy, jaw_width)
        conf_list = self.inik_slvr.gen_linear_motion(hnd_name,
                                                     start_jaw_center_pos,
                                                     start_jaw_center_rotmat,
                                                     goal_jaw_center_pos,
                                                     goal_jaw_center_rotmat,
                                                     obstacle_list=obstacle_list,
                                                     granularity=granularity,
                                                     seed_jnt_values=start_conf)
        jawwidth_list = self.gen_jawwidth_motion(conf_list, jaw_width)
        objpose_list = self.gen_object_motion(hnd_name, conf_list, rel_obj_pos, rel_obj_rotmat, type='relative')
        # release the object and restore robot configuration
        self.robot_s.release(hnd_name, objcm_copy, jaw_width)
        self.robot_s.fk(jnt_values_bk)
        return conf_list, jawwidth_list, objpose_list

    def gen_holding_approach_motion(self,
                                    component_name,
                                    hnd_name,  # TODO hnd is on  a manipulator
                                    objcm,
                                    grasp_info,
                                    start_obj_pos,
                                    start_obj_rotmat,
                                    goal_obj_pos,
                                    goal_obj_rotmat,
                                    approach_direction=None,  # use np.array([0,0,0]) if do not need linear
                                    approach_distance=.1,
                                    obstacle_list=[],
                                    seed_jnt_values=None,
                                    linear_granularity=.03):
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        if approach_direction is None:
            approach_direction = jaw_center_rotmat[:, 2]
        # object's pre pose
        pre_goal_obj_pos = goal_obj_pos - approach_direction * approach_distance
        pre_goal_obj_rotmat = goal_obj_rotmat
        # jaw centers
        start_jaw_center_pos = start_obj_rotmat.dot(jaw_center_pos) + start_obj_pos
        start_jaw_center_rotmat = start_obj_rotmat.dot(jaw_center_rotmat)
        goal_jaw_center_pos = goal_obj_rotmat.dot(jaw_center_pos) + goal_obj_pos
        goal_jaw_center_rotmat = goal_obj_rotmat.dot(jaw_center_rotmat)
        pre_goal_jaw_center_pos = pre_goal_obj_rotmat.dot(jaw_center_pos) + pre_goal_obj_pos
        pre_goal_jaw_center_rotmat = pre_goal_obj_rotmat.dot(jaw_center_rotmat)
        # save current robot jnt values and move robot to start jaw pose
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        start_conf = self.robot_s.ik(component_name,
                                     start_jaw_center_pos,
                                     start_jaw_center_rotmat,
                                     seed_jnt_values=seed_jnt_values)
        if start_conf is None:
            print("Cannot solve ik at the given start_obj_pos, start_obj_rotmat!")
            return None, None, None
        self.robot_s.fk(start_conf)
        # set a copy of the object to the start pose, hold the object, and move it to goal object pose
        objcm = objcm.copy()
        objcm.set_pos(start_obj_pos)
        objcm.set_rotmat(start_obj_rotmat)
        rel_obj_pos, rel_obj_rotmat = self.robot_s.hold(hnd_name, objcm, jaw_width)
        pre_goal_conf = self.robot_s.ik(component_name,
                                        pre_goal_jaw_center_pos,
                                        pre_goal_jaw_center_rotmat,
                                        seed_jnt_values=start_conf)
        # approach linear
        conf_list_linear = self.inik_slvr.gen_linear_motion(component_name,
                                                            pre_goal_jaw_center_pos,
                                                            pre_goal_jaw_center_rotmat,
                                                            goal_jaw_center_pos,
                                                            goal_jaw_center_rotmat,
                                                            obstacle_list=obstacle_list,
                                                            granularity=linear_granularity,
                                                            seed_jnt_values=pre_goal_conf)
        if conf_list_linear is None:
            print("Cannot generate the linear part of holding approach motion!")
            return None, None, None
        jawwidth_list_linear = self.gen_jawwidth_motion(conf_list_linear, jaw_width)
        objpose_list_linear = self.gen_object_motion(component_name,
                                                     conf_list_linear,
                                                     rel_obj_pos,
                                                     rel_obj_rotmat,
                                                     type='relative')
        conf_list_rrtc = self.rrtc_planner.plan(component_name,
                                                start_conf,
                                                pre_goal_conf,
                                                obstacle_list,
                                                ext_dist=.05,
                                                rand_rate=70,
                                                max_time=300)
        if conf_list_rrtc is None:
            print("Cannot generate the rrtc part of holding approach motion!")
            return None, None, None
        jawwidth_list_rrtc = self.gen_jawwidth_motion(conf_list_rrtc, jaw_width)
        objpose_list_rrtc = self.gen_object_motion(component_name, conf_list_rrtc, rel_obj_pos, rel_obj_rotmat,
                                                   type='relative')
        self.robot_s.release(hnd_name, objcm, jaw_width)
        self.robot_s.fk(jnt_values_bk)
        return conf_list_rrtc + conf_list_linear, \
               jawwidth_list_rrtc + jawwidth_list_linear, \
               objpose_list_rrtc + objpose_list_linear

    def gen_holding_depart_linear(self,
                                  component_name,
                                  hnd_name,  # TODO hnd is on  a manipulator
                                  objcm,
                                  grasp_info,
                                  start_obj_pos,
                                  start_obj_rotmat,
                                  depart_direction=None,
                                  depart_distance=.1,
                                  granularity=.03,
                                  obstacle_list=[],
                                  seed_jnt_values=None):
        """
        :param component_name:
        :param hnd_name:
        :param objcm:
        :param grasp_info:
        :param goal_obj_pos:
        :param goal_obj_rotmat:
        :param approach_direction:
        :param approach_distance:
        :param granularity:
        :param obstacle_list:
        :param seed_jnt_values:
        :return:
        author: weiwei
        date: 20210511
        """
        # get start object pose
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        if depart_direction is None:
            depart_direction = -jaw_center_rotmat[:, 2]
        goal_obj_pos = start_obj_pos + depart_direction * depart_distance
        goal_obj_rotmat = start_obj_rotmat
        # use gen_holding_approach_linear to
        return self.gen_holding_approach_linear(component_name=component_name,
                                                hnd_name=hnd_name,
                                                objcm=objcm,
                                                grasp_info=grasp_info,
                                                goal_obj_pos=goal_obj_pos,
                                                goal_obj_rotmat=goal_obj_rotmat,
                                                approach_direction=depart_direction,
                                                approach_distance=depart_distance,
                                                granularity=granularity,
                                                obstacle_list=obstacle_list,
                                                seed_jnt_values=seed_jnt_values)

    def gen_holding_depart_motion(self,
                                  component_name,
                                  hnd_name,  # TODO hnd is on  a manipulator
                                  objcm,
                                  grasp_info,
                                  start_obj_pos,
                                  start_obj_rotmat,
                                  goal_obj_pos,
                                  goal_obj_rotmat,
                                  depart_direction=None,
                                  depart_distance=.1,
                                  obstacle_list=[],
                                  seed_jnt_values=None,
                                  linear_granularity=.03):
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        if depart_direction is None:
            depart_direction = -jaw_center_rotmat[:, 2]
        # object's post pose
        post_start_obj_pos = start_obj_pos + depart_direction * depart_distance
        post_start_obj_rotmat = start_obj_rotmat
        # jaw centers
        start_jaw_center_pos = start_obj_rotmat.dot(jaw_center_pos) + start_obj_pos
        start_jaw_center_rotmat = start_obj_rotmat.dot(jaw_center_rotmat)
        post_start_jaw_center_pos = post_start_obj_rotmat.dot(jaw_center_pos) + post_start_obj_pos
        post_start_jaw_center_rotmat = post_start_obj_rotmat.dot(jaw_center_rotmat)
        goal_jaw_center_pos = goal_obj_rotmat.dot(jaw_center_pos) + goal_obj_pos
        goal_jaw_center_rotmat = goal_obj_rotmat.dot(jaw_center_rotmat)
        # save current robot jnt values and move robot to start jaw pose
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        start_conf = self.robot_s.ik(component_name,
                                     start_jaw_center_pos,
                                     start_jaw_center_rotmat,
                                     seed_jnt_values=seed_jnt_values)
        if start_conf is None:
            print("Cannot solve ik at the given start_obj_pos, start_obj_rotmat!")
            return None, None, None
        self.robot_s.fk(start_conf)
        # set a copy of the object to the start pose, hold the object, and move it to goal object pose
        objcm = objcm.copy()
        objcm.set_pos(start_obj_pos)
        objcm.set_rotmat(start_obj_rotmat)
        rel_obj_pos, rel_obj_rotmat = self.robot_s.hold(hnd_name, objcm, jaw_width)
        # approach linear
        conf_list_linear = self.inik_slvr.gen_linear_motion(component_name,
                                                            start_jaw_center_pos,
                                                            start_jaw_center_rotmat,
                                                            post_start_jaw_center_pos,
                                                            post_start_jaw_center_rotmat,
                                                            obstacle_list=obstacle_list,
                                                            granularity=linear_granularity,
                                                            seed_jnt_values=seed_jnt_values)
        if conf_list_linear is None:
            print("Cannot generate the linear part of holding approach motion!")
            return None, None, None
        jawwidth_list_linear = self.gen_jawwidth_motion(conf_list_linear, jaw_width)
        objpose_list_linear = self.gen_object_motion(component_name,
                                                     conf_list_linear,
                                                     rel_obj_pos,
                                                     rel_obj_rotmat,
                                                     type='relative')
        post_start_conf = conf_list_linear[-1]
        goal_conf = self.robot_s.ik(component_name,
                                    goal_jaw_center_pos,
                                    goal_jaw_center_rotmat,
                                    seed_jnt_values=post_start_conf)
        conf_list_rrtc = self.rrtc_planner.plan(component_name,
                                                post_start_conf,
                                                goal_conf,
                                                obstacle_list,
                                                ext_dist=.05,
                                                rand_rate=70,
                                                max_time=300)
        if conf_list_rrtc is None:
            print("Cannot generate the rrtc part of holding approach motion!")
            return None, None, None
        jawwidth_list_rrtc = self.gen_jawwidth_motion(conf_list_rrtc, jaw_width)
        objpose_list_rrtc = self.gen_object_motion(component_name, conf_list_rrtc, rel_obj_pos, rel_obj_rotmat,
                                                   type='relative')
        self.robot_s.release(hnd_name, objcm, jaw_width)
        self.robot_s.fk(jnt_values_bk)
        return conf_list_linear + conf_list_rrtc, \
               jawwidth_list_linear + jawwidth_list_rrtc, \
               objpose_list_linear + objpose_list_rrtc

    def gen_holding_approach_holding_depart_linear(self):
        raise NotImplementedError

    def gen_holding_approach_holding_depart_motion(self):
        raise NotImplementedError

    def gen_holding_depart_holding_approach_linear(self):
        raise NotImplementedError

    def gen_holding_depart_holding_approach_motion(self):
        raise NotImplementedError

    def gen_approach_holding_depart_linear(self,
                                           component_name,
                                           hnd_name,
                                           objcm,
                                           grasp_info,
                                           goal_obj_pos,
                                           goal_obj_rotmat,
                                           approach_direction=None,  # np.array([0, 0, -1])
                                           approach_distance=.1,
                                           approach_jawwidth=None,
                                           depart_direction=None,  # np.array([0, 0, 1])
                                           depart_distance=.1,
                                           granularity=.03,
                                           obstacle_list=[],
                                           seed_jnt_values=None):
        """
        :param component_name:
        :param hnd_name:
        :param objcm:
        :param grasp_info:
        :param goal_obj_pos:
        :param goal_obj_rotmat:
        :param approach_direction:
        :param approach_distance:
        :param approach_jawwidth:
        :param depart_direction:
        :param depart_distance:
        :param granularity:
        :param obstacle_list:
        :param seed_jnt_values:
        :return:
        """
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        goal_jaw_center_pos = goal_obj_rotmat.dot(jaw_center_pos) + goal_obj_pos
        goal_jaw_center_rotmat = goal_obj_rotmat.dot(jaw_center_rotmat)
        if approach_direction is None:
            approach_direction = jaw_center_rotmat[:, 2]
        if approach_jawwidth is None:
            approach_jawwidth = self.robot_s.hnd_dict[hnd_name].jaw_width_rng[1]
        approach_conf_list, approach_jawwidth_list = \
            self.gen_approach_linear(component_name=component_name,
                                     goal_tcp_pos=goal_jaw_center_pos,
                                     goal_tcp_rotmat=goal_jaw_center_rotmat,
                                     approach_direction=approach_direction,
                                     approach_distance=approach_distance,
                                     approach_jawwidth=approach_jawwidth,
                                     granularity=granularity,
                                     obstacle_list=obstacle_list,
                                     seed_jnt_values=seed_jnt_values)
        approach_objpose_list = self.gen_object_motion(component_name, goal_obj_pos, goal_obj_rotmat, type="absolute")
        if approach_conf_list is None:
            print('Cannot perform approach linear!')
            return None, None, None
        else:
            if depart_direction is None:
                depart_direction = jaw_center_rotmat[:, 2]
            depart_conf_list, depart_jawwidth_list, depart_objpose_list = \
                self.gen_holding_depart_linear(component_name=component_name,
                                               hnd_name=hnd_name,
                                               objcm=objcm,
                                               grasp_info=grasp_info,
                                               start_obj_pos=goal_obj_pos,
                                               start_obj_rotmat=goal_obj_rotmat,
                                               depart_direction=depart_direction,
                                               depart_distance=depart_distance,
                                               granularity=granularity,
                                               obstacle_list=obstacle_list,
                                               seed_jnt_values=approach_conf_list[-1])
            if depart_conf_list is None:
                print('Cannot perform depart linear!')
                return None, None, None
            else:
                return approach_conf_list + depart_conf_list, \
                       approach_jawwidth_list + depart_jawwidth_list, \
                       approach_objpose_list + depart_objpose_list

    def gen_approach_holding_depart_motion(self,
                                           component_name,
                                           hnd_name,
                                           objcm,
                                           grasp_info,
                                           start_conf,
                                           goal_obj_pos,
                                           goal_obj_rotmat,
                                           approach_direction=None,  # np.array([0, 0, -1])
                                           approach_distance=.1,
                                           approach_jawwidth=None,
                                           depart_direction=None,  # np.array([0, 0, 1])
                                           depart_distance=.1,
                                           granularity=.03,
                                           seed_jnt_values=None):
        """
        :param component_name:
        :param hnd_name:
        :param objcm:
        :param grasp_info:
        :param start_conf:
        :param goal_obj_pos:
        :param goal_obj_rotmat:
        :param approach_direction:
        :param approach_distance:
        :param approach_jawwidth:
        :param depart_direction:
        :param depart_distance:
        :param granularity:
        :param seed_jnt_values:
        :return:
        author: weiwei
        date: 20210511
        """
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        goal_jaw_center_pos = goal_obj_rotmat.dot(jaw_center_pos) + goal_obj_pos
        goal_jaw_center_rotmat = goal_obj_rotmat.dot(jaw_center_rotmat)
        if approach_direction is None:
            approach_direction = jaw_center_rotmat[:, 2]
        if approach_jawwidth is None:
            approach_jawwidth = self.robot_s.hnd_dict[hnd_name].jaw_width_rng[1]
        approach_conf_list, approach_jawwidth_list = \
            self.gen_approach_motion(component_name=component_name,
                                     goal_tcp_pos=goal_jaw_center_pos,
                                     goal_tcp_rotmat=goal_jaw_center_rotmat,
                                     start_conf=start_conf,
                                     approach_direction=approach_direction,
                                     approach_distance=approach_distance,
                                     approach_jawwidth=approach_jawwidth,
                                     granularity=granularity,
                                     obstacle_list=[],
                                     seed_jnt_values=seed_jnt_values)
        approach_objpose_list = self.gen_object_motion(component_name,
                                                       goal_obj_pos,
                                                       goal_obj_rotmat,
                                                       type="absolute")
        if approach_conf_list is None:
            print('Cannot perform approach motion!')
            return None, None, None
        else:
            if depart_direction is None:
                depart_direction = jaw_center_rotmat[:, 2]
            depart_conf_list, depart_jawwidth_list, depart_objpose_list = \
                self.gen_holding_depart_motion(component_name=component_name,
                                               hnd_name=hnd_name,
                                               objcm=objcm,
                                               grasp_info=grasp_info,
                                               start_obj_pos=goal_obj_pos,
                                               start_obj_rotmat=goal_obj_rotmat,
                                               depart_direction=depart_direction,
                                               depart_distance=depart_distance,
                                               granularity=granularity,
                                               seed_jnt_values=approach_conf_list[-1])
            if depart_conf_list is None:
                print('Cannot perform depart action!')
                return None, None, None
            else:
                return approach_conf_list + depart_conf_list, \
                       approach_jawwidth_list + depart_jawwidth_list, \
                       approach_objpose_list + depart_objpose_list

    def gen_pickup_motion(self,
                          component_name,
                          hnd_name,
                          objcm,
                          grasp_info,
                          goal_obj_pos,
                          goal_obj_rotmat,
                          start_conf=None,
                          goal_conf=None,
                          approach_direction=None,  # np.array([0, 0, -1])
                          approach_distance=.1,
                          approach_jawwidth=.05,
                          depart_direction=None,  # np.array([0, 0, 1])
                          depart_distance=.1,
                          granularity=.03,
                          obstacle_list=[],
                          seed_jnt_values=None):
        """
        degenerate into gen_pickup_primitive if both seed_jnt_values and goal_conf are None
        :param component_name:
        :param hnd_name:
        :param objcm:
        :param grasp_info:
        :param goal_obj_pos:
        :param goal_obj_rotmat:
        :param start_conf:
        :param goal_conf:
        :param approach_direction:
        :param approach_distance:
        :param approach_jawwidth:
        :param depart_direction:
        :param depart_distance:
        :param depart_jawwidth:
        :param granularity:
        :param obstacle_list:
        :param seed_jnt_values:
        :return: [conf_list, jawwidth_list, objpose_list, rel_oih_pos, rel_oih_rotmat]
        author: weiwei
        date: 20210125
        """
        objcm = objcm.copy()
        objcm.set_pos(goal_obj_pos)
        objcm.set_rotmat(goal_obj_rotmat)
        depart_jawwidth, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        goal_jaw_center_pos = goal_obj_pos + goal_obj_rotmat.dot(jaw_center_pos)
        goal_jaw_center_rotmat = goal_obj_rotmat.dot(jaw_center_rotmat)
        abs_obj_pos = objcm.get_pos()
        abs_obj_rotmat = objcm.get_rotmat()
        if approach_direction is None:
            approach_direction = goal_jaw_center_rotmat[:, 2]
        approach_conf_list, approach_jawwidth_list = self.gen_approach_motion(component_name,
                                                                              goal_jaw_center_pos,
                                                                              goal_jaw_center_rotmat,
                                                                              start_conf,
                                                                              approach_direction,
                                                                              approach_distance,
                                                                              approach_jawwidth,
                                                                              granularity,
                                                                              obstacle_list,
                                                                              seed_jnt_values)
        if len(approach_conf_list) == 0:
            print('Cannot perform approach action!')
        else:
            approach_objpose_list = self.gen_object_motion(component_name, approach_conf_list, abs_obj_pos,
                                                           abs_obj_rotmat)
            # TODO change to gen_hold_and_moveto_linear
            jnt_values_bk = self.robot_s.get_jnt_values(component_name)
            self.robot_s.fk(component_name, approach_conf_list[-1])
            rel_obj_pos, rel_obj_rotmat = self.robot_s.hold(hnd_name, objcm, depart_jawwidth)
            if depart_direction is None:
                depart_direction = -goal_jaw_center_rotmat[:, 2]
            depart_conf_list, depart_jawwidth_list = self.gen_depart_motion(component_name,
                                                                            goal_jaw_center_pos,
                                                                            goal_jaw_center_rotmat,
                                                                            goal_conf,
                                                                            depart_direction,
                                                                            depart_distance,
                                                                            depart_jawwidth,
                                                                            granularity,
                                                                            obstacle_list,
                                                                            seed_jnt_values=approach_conf_list[-1])
            depart_objpose_list = self.gen_object_motion(component_name, depart_conf_list, rel_obj_pos, rel_obj_rotmat,
                                                         type='relative')
            self.robot_s.release(hnd_name, objcm,
                                 depart_jawwidth)  # we do not maintain inner states, directly return seqs
            self.robot_s.fk(component_name, jnt_values_bk)
            if len(depart_conf_list) == 0:
                print('Cannot perform depart action!')
            else:
                return approach_conf_list + depart_conf_list, \
                       approach_jawwidth_list + depart_jawwidth_list, \
                       approach_objpose_list + depart_objpose_list

    def gen_placedown_linear(self,
                             component_name,
                             hnd_name,
                             objcm,
                             grasp_info,
                             goal_obj_pos,
                             goal_obj_rotmat,
                             approach_direction=None,  # np.array([0, 0, -1])
                             approach_distance=.1,
                             depart_direction=None,  # np.array([0, 0, 1])
                             depart_distance=.1,
                             depart_jawwidth=.0,
                             granularity=.03,
                             obstacle_list=[],
                             seed_jnt_values=None):
        """
        degenerate into gen_pickup_primitive if both seed_jnt_values and goal_conf are None
        :param component_name:
        :param hnd_name:
        :param objcm:
        :param grasp_info:
        :param goal_obj_pos:
        :param goal_obj_rotmat:
        :param seed_jnt_values:
        :param goal_conf:
        :param approach_direction:
        :param approach_distance:
        :param depart_direction:
        :param depart_distance:
        :param depart_jawwidth:
        :param granularity:
        :param obstacle_list:
        :param seed_jnt_values:
        :return:
        author: weiwei
        date: 20210125
        """
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        if approach_direction is None:
            approach_direction = jaw_center_rotmat[:, 2]
        approach_conf_list, approach_jawwidth_list, approach_objpose_list = self.gen_moveto_linear(component_name,
                                                                                                   hnd_name,
                                                                                                   objcm,
                                                                                                   grasp_info,
                                                                                                   goal_obj_pos,
                                                                                                   goal_obj_rotmat,
                                                                                                   approach_direction,
                                                                                                   approach_distance,
                                                                                                   granularity,
                                                                                                   seed_jnt_values)
        if len(approach_conf_list) == 0:
            print('Cannot perform place down action!')
        else:
            goal_jaw_center_pos = goal_obj_rotmat.dot(jaw_center_pos) + goal_obj_pos
            goal_jaw_center_rotmat = goal_obj_rotmat.dot(jaw_center_rotmat)
            if depart_direction is None:
                depart_direction = goal_jaw_center_rotmat[:, 2]
            depart_conf_list, depart_jawwidth_list = self.gen_depart_linear(component_name,
                                                                            goal_jaw_center_pos,
                                                                            goal_jaw_center_rotmat,
                                                                            depart_direction,
                                                                            depart_distance,
                                                                            depart_jawwidth,
                                                                            granularity,
                                                                            obstacle_list,
                                                                            seed_jnt_values=approach_conf_list[-1])
            depart_objpose_list = self.gen_object_motion(component_name,
                                                         depart_conf_list,
                                                         approach_objpose_list[-1][:3, 3],
                                                         approach_objpose_list[-1][:3, :3],
                                                         type='absolute')
            if len(depart_conf_list) == 0:
                print('Cannot perform depart action!')
            else:
                return approach_conf_list + depart_conf_list, \
                       approach_jawwidth_list + depart_jawwidth_list, \
                       approach_objpose_list + depart_objpose_list

    def gen_placedown_motion(self,
                             component_name,
                             hnd_name,
                             objcm,
                             grasp_info,
                             goal_obj_pos,
                             goal_obj_rotmat,
                             start_conf=None,
                             goal_conf=None,
                             approach_direction=None,  # np.array([0, 0, -1])
                             approach_distance=.1,
                             depart_direction=None,  # np.array([0, 0, 1])
                             depart_distance=.1,
                             depart_jawwidth=.0,
                             granularity=.03,
                             obstacle_list=[],
                             seed_jnt_values=None):
        """
        degenerate into gen_pickup_primitive if both seed_jnt_values and goal_conf are None
        :param component_name:
        :param hnd_name:
        :param objcm:
        :param grasp_info:
        :param goal_obj_pos:
        :param goal_obj_rotmat:
        :param start_conf:
        :param goal_conf:
        :param approach_direction:
        :param approach_distance:
        :param depart_direction:
        :param depart_distance:
        :param depart_jawwidth:
        :param granularity:
        :param obstacle_list:
        :param seed_jnt_values:
        :return:
        author: weiwei
        date: 20210125
        """
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        if approach_direction is None:
            approach_direction = jaw_center_rotmat[:, 2]
        approach_conf_list, approach_jawwidth_list, approach_objpose_list = self.gen_holding_approach_motion(
            component_name,
            hnd_name,
            objcm,
            grasp_info,
            goal_obj_pos,
            goal_obj_rotmat,
            approach_direction,
            approach_distance,
            obstacle_list,
            seed_jnt_values)
        if len(approach_conf_list) == 0:
            print('Cannot perform approach action!')
        else:
            goal_jaw_center_pos = goal_obj_rotmat.dot(jaw_center_pos) + goal_obj_pos
            goal_jaw_center_rotmat = goal_obj_rotmat.dot(jaw_center_rotmat)
            if depart_direction is None:
                depart_direction = -goal_jaw_center_rotmat[:, 2]
            depart_conf_list, depart_jawwidth_list = self.gen_depart_motion(component_name,
                                                                            goal_jaw_center_pos,
                                                                            goal_jaw_center_rotmat,
                                                                            goal_conf,
                                                                            depart_direction,
                                                                            depart_distance,
                                                                            depart_jawwidth,
                                                                            granularity,
                                                                            obstacle_list,
                                                                            seed_jnt_values=approach_conf_list[-1])
            depart_objpose_list = self.gen_object_motion(component_name,
                                                         depart_conf_list,
                                                         approach_objpose_list[-1][:3, 3],
                                                         approach_objpose_list[-1][:3, :3],
                                                         type='absolute')
            if len(depart_conf_list) == 0:
                print('Cannot perform depart action!')
            else:
                return approach_conf_list + depart_conf_list, \
                       approach_jawwidth_list + depart_jawwidth_list, \
                       approach_objpose_list + depart_objpose_list
        return [], [], []

    def gen_pick_and_place_motion(self,
                                  manipulator_name,
                                  hand_name,
                                  objcm,
                                  grasp_info_list,
                                  start_conf,
                                  goal_homomat_list):
        """
,
                                  pick_approach_direction_list,
                                  pick_depart_direction_list,
                                  place_approach_direction_list,
                                  place_depart_direction_list
        :param manipulator_name:
        :param hand_name:
        :param grasp_info_list: a list like [[jaw_width, gl_jaw_center_pos, pos, rotmat], ...]
        :param start_conf:
        :param goal_homomat_list: a list of tcp goals like [homomat0, homomat1, ...]
        :return:
        author: weiwei
        date: 20191122, 20200105
        """
        common_grasp_id_list, _ = self.find_common_graspids(manipulator_name,
                                                            hand_name,
                                                            grasp_info_list,
                                                            goal_homomat_list)
        grasp_info = grasp_info_list[common_grasp_id_list[0]]
        conf_list = []
        jawwidth_list = []
        objpose_list = []
        for i in range(len(goal_homomat_list) - 1):
            goal_homomat0 = goal_homomat_list[i]
            obj_pos0 = goal_homomat0[:3, 3]
            obj_rotmat0 = goal_homomat0[:3, :3]
            conf_list0, jawwidth_list0, objpose_list0 = self.gen_pickup_motion(manipulator_name,
                                                                               hand_name,
                                                                               objcm,
                                                                               grasp_info,
                                                                               goal_obj_pos=obj_pos0,
                                                                               goal_obj_rotmat=obj_rotmat0,
                                                                               start_conf=start_conf,
                                                                               approach_direction=np.array([0, 0, -1]),
                                                                               approach_distance=.1,
                                                                               depart_direction=np.array([0, 0, 1]),
                                                                               depart_distance=.2)
            goal_homomat1 = goal_homomat_list[i + 1]
            obj_pos1 = goal_homomat1[:3, 3]
            obj_rotmat1 = goal_homomat1[:3, :3]
            conf_list1, jawwidth_list1, objpose_list1 = self.gen_placedown_motion(manipulator_name,
                                                                                  hand_name,
                                                                                  objcm,
                                                                                  grasp_info,
                                                                                  goal_obj_pos=obj_pos1,
                                                                                  goal_obj_rotmat=obj_rotmat1,
                                                                                  start_conf=conf_list0[-1],
                                                                                  approach_direction=np.array(
                                                                                      [0, 0, -1]),
                                                                                  approach_distance=.1,
                                                                                  depart_direction=np.array(
                                                                                      [.5, .5, 1]),
                                                                                  depart_distance=.2)
            start_conf = conf_list1[-1]
            conf_list = conf_list + conf_list0 + conf_list1
            jawwidth_list = jawwidth_list + jawwidth_list0 + jawwidth_list1
            objpose_list = objpose_list + objpose_list0 + objpose_list1
        return conf_list, jawwidth_list, objpose_list


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
        tmp_objcm.set_rgba([1,0,0,.3])
        tmp_objcm.set_homomat(rm.homomat_from_posrot(goal_pos, goal_rotmat))
        tmp_objcm.attach_to(base)
    grasp_info_list = gutil.load_pickle_file(objcm_name='tubebig', file_name='yumi_tube_big.pickle')
    grasp_info = grasp_info_list[0]
    pp_planner = PickPlacePlanner(robot_s=robot_s)
    # conf_list, jawwidth_list, objpose_list = \
    #     pp_planner.gen_approach_motion_with_object(hand_name,
    #                                                objcm,
    #                                                grasp_info,
    #                                                goal_obj_pos=goal_homomat_list[0][:3, 3],
    #                                                goal_obj_rotmat=goal_homomat_list[0][:3, :3],
    #                                                start_conf=start_conf)
    for grasp_info in grasp_info_list:
        conf_list, jaw_width_list, objpose_list = \
            pp_planner.gen_holding_moveto(hand_name=hand_name,
                                          objcm=objcm,
                                          grasp_info=grasp_info,
                                          obj_pose_list=goal_homomat_list,
                                          depart_direction_list=[np.array([0,0,1])] * len(goal_homomat_list),
                                          approach_direction_list=[np.array([0,0,-1])] * len(goal_homomat_list),
                                          # depart_distance_list=[None] * len(goal_homomat_list),
                                          # approach_distance_list=[None] * len(goal_homomat_list),
                                          depart_distance_list=[.2] * len(goal_homomat_list),
                                          approach_distance_list=[.2] * len(goal_homomat_list),
                                          ad_linear_granularity=.003,
                                          use_rrt = True,
                                          obstacle_list=[],
                                          seed_jnt_values=start_conf)
        print(robot_s.rgt_oih_infos, robot_s.lft_oih_infos)
        if conf_list is not None:
            break
        # pp_planner.gen_approach_motion_with_object(hand_name,
        #                                            objcm,
        #                                            grasp_info,
        #                                            goal_obj_pos=goal_homomat_list[0][:3, 3],
        #                                            goal_obj_rotmat=goal_homomat_list[0][:3, :3],
        #                                            start_conf=start_conf)
    print(conf_list[0])
    # animation
    robot_attached_list = []
    object_attached_list = []
    counter = [0]
    def update(robot_s,
               hand_name,
               objcm,
               robot_path,
               jaw_width_path,
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
        robot_s.jaw_to(hand_name, jaw_width_path[counter[0]])
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
                                     jaw_width_list,
                                     objpose_list,
                                     robot_attached_list,
                                     object_attached_list,
                                     counter],
                          appendTask=True)
    base.run()

    for i, conf in enumerate(conf_list):
        robot_s.fk(manipulator_name, conf)
        robot_s.jaw_to(hand_name, jaw_width_list[i])
        robot_s.gen_meshmodel().attach_to(base)
        tmp_objcm = objcm.copy()
        tmp_objcm.set_homomat(objpose_list[i])
        tmp_objcm.attach_to(base)
    base.run()

    conf_list, jawwidth_list, objpose_list = pp_planner.gen_pick_and_place_motion(manipulator_name,
                                                                                  hand_name,
                                                                                  objcm,
                                                                                  grasp_info_list,
                                                                                  start_conf,
                                                                                  goal_homomat_list)
    for i, conf in enumerate(conf_list):
        robot_s.fk(manipulator_name, conf)
        robot_s.jaw_to(hand_name, jawwidth_list[i])
        robot_s.gen_meshmodel().attach_to(base)
        tmp_objcm = objcm.copy()
        tmp_objcm.set_homomat(objpose_list[i])
        tmp_objcm.attach_to(base)
    # hnd_instance = robot_s.hnd_dict[hand_name].copy()
    # for goal_homomat in goal_homomat_list:
    #     obj_pos = goal_homomat[:3, 3]
    #     obj_rotmat = goal_homomat[:3, :3]
    #     for grasp_id in common_grasp_id_list:
    #         jaw_width, tcp_pos, hnd_pos, hnd_rotmat = grasp_info_list[grasp_id]
    #         new_tcp_pos = obj_rotmat.dot(tcp_pos) + obj_pos
    #         new_hnd_pos = obj_rotmat.dot(hnd_pos) + obj_pos
    #         new_hnd_rotmat = obj_rotmat.dot(hnd_rotmat)
    #         tmp_hnd = hnd_instance.copy()
    #         tmp_hnd.fix_to(new_hnd_pos, new_hnd_rotmat)
    #         tmp_hnd.jaw_to(jaw_width)
    #         tmp_hnd.gen_meshmodel().attach_to(base)
    #         jnt_values = robot_s.ik(hand_name,
    #                                       new_tcp_pos,
    #                                       new_hnd_rotmat)
    #         try:
    #             robot_s.fk(hand_name, jnt_values)
    #             robot_s.gen_meshmodel().attach_to(base)
    #         except:
    #             continue
    # goal_pos = np.array([.55, -.1, .3])
    # goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    # gm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)
    base.run()
